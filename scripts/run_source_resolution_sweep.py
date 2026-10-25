#!/usr/bin/env python3
"""Build, run, audit, and replot the reproducible CC/GL source sweep.

The sweep uses local DUCC, SHTns, RustSHT, and pyspharm-cpp Git sources, plus
a separate conda-forge pyspharm environment. Numerical gates run in a separate
process before any matching timing cells. Raw timing batches remain in one
JSON file per process cell.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sht_bench.source_adapters import MODULES, PLANNING_POLICIES, Workload, capability  # noqa: E402
from sht_bench.conda_environment import (  # noqa: E402
    CONDA_CHANNEL, NUMPY_VERSION, PYTHON_VERSION, PYSPHARM_VERSION,
    create_command as conda_create_command, environment_python as conda_python,
    find_conda, installed_environment_exports, verify_import_path,
    verify_pyspharm_environment,
)
from sht_bench.source_resolution import (  # noqa: E402
    BACKENDS, BACKEND_PRECISIONS, PROFILE_SCHEMA, PRECISIONS, THREADS,
    case_reference, planned_cells, plan_counts, resolution_cases,
)
from sht_bench.source_runner import (  # noqa: E402
    harness_hash, physical_cores, validation_identity, write_json,
)
from sht_bench.source_resources import preflight  # noqa: E402

SCHEMA = 'sht_bench.source_resolution_sweep.v1'
SOURCES = {
    'ducc': ('../ducc', 'HEAD'),
    'shtns': ('../shtns', 'HEAD'),
    'rustsht': ('../RustSHT', 'main'),
    'pyspharm-cpp': ('../pyspharm-cpp', 'HEAD'),
}
MODULES_FOR_BUILD = {
    'ducc': ('ducc0', 'ducc0'),
    'shtns': ('shtns', 'shtns'),
    'rustsht': ('rustsht', 'rustsht'),
    'pyspharm-cpp': ('spharm_cpp', 'pyspharm-cpp'),
    'pyspharm': ('spharm', 'pyspharm'),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _relevant_untracked_source(relative: str) -> bool:
    parts = Path(relative).parts
    if not parts or parts[0] in {'reports','results','docs','benchmarks','tests','.github','target'}:
        return False
    name = parts[-1]
    suffix = Path(name).suffix.lower()
    return name in {'CMakeLists.txt','Cargo.toml','Cargo.lock','pyproject.toml','setup.py','setup.cfg','build.rs','Makefile'} or suffix in {
        '.py','.pyi','.pyx','.c','.cc','.cpp','.h','.hpp','.rs','.f','.f90','.toml','.cmake',
    }


def _git(path: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ['git', '-C', str(path), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode().strip()


def _source_state(path: Path, requested_ref: str) -> dict:
    path = path.expanduser().resolve()
    if not (path / '.git').exists() and not _git(path, 'rev-parse', '--git-dir'):
        raise ValueError(f'not a Git checkout: {path}')
    head = _git(path, 'rev-parse', 'HEAD')
    target = _git(path, 'rev-parse', f'{requested_ref}^{{commit}}')
    branch = _git(path, 'branch', '--show-current') or None
    origin = subprocess.run(
        ['git', '-C', str(path), 'remote', 'get-url', 'origin'],
        text=True, capture_output=True,
    )
    status = _git(path, 'status', '--porcelain=v1', '--untracked-files=all')
    diff = _git(path, 'diff', 'HEAD', '--binary', binary=True)
    untracked = _git(path, 'ls-files', '--others', '--exclude-standard').splitlines()
    untracked_records = []
    # Untracked source changes are checksum-recorded. Large report/data trees
    # stay in their original checkout and are not copied into the build tree.
    for relative in untracked:
        candidate = path / relative
        relevant = _relevant_untracked_source(relative)
        if candidate.is_file() and (relevant or candidate.stat().st_size <= 16 * 1024 * 1024):
            untracked_records.append({'path': relative, 'sha256': _sha(candidate.read_bytes()), 'size_bytes': candidate.stat().st_size,'source_relevant':relevant})
        else:
            untracked_records.append({'path': relative, 'sha256': None, 'size_bytes': None,'source_relevant':relevant})
    return {
        'path': str(path),
        'origin': origin.stdout.strip() if origin.returncode == 0 else None,
        'branch_at_capture': branch,
        'head_sha_at_capture': head,
        'requested_ref': requested_ref,
        'resolved_sha': target,
        'dirty': bool(status),
        'status_porcelain': status,
        'status_sha256': _sha(status.encode()),
        'tracked_diff_sha256': _sha(diff),
        'tracked_diff_bytes': len(diff),
        'untracked_files': untracked_records,
        'untracked_state_sha256': _sha(_canonical(untracked_records)),
        'source_relevant_untracked': [item for item in untracked_records if item['source_relevant']],
        'build_policy': 'Git archive of resolved commit plus tracked diff and source-relevant untracked files when resolved commit equals captured HEAD; report and handoff artifacts are recorded and excluded',
    }


def _tree_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '.git' in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append((relative, _sha(path.read_bytes())))
    return _sha(_canonical(entries))


def _extract_archive(repo: Path, revision: str, destination: Path) -> None:
    archive = _git(repo, 'archive', '--format=tar', revision, binary=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=__import__('io').BytesIO(archive), mode='r:') as bundle:
        bundle.extractall(destination, filter='data')


def _make_source_snapshot(name: str, state: dict, destination: Path) -> dict:
    source = Path(state['path'])
    if state['tracked_diff_bytes'] and state['resolved_sha'] != state['head_sha_at_capture']:
        raise ValueError(f'{name}: tracked local changes cannot be applied to requested ref {state["requested_ref"]}')
    _extract_archive(source, state['resolved_sha'], destination)
    if state['tracked_diff_bytes']:
        diff = _git(source, 'diff', 'HEAD', '--binary', binary=True)
        patch_path = destination.parent / f'{name}-tracked-diff.patch'
        patch_path.write_bytes(diff)
        subprocess.run(['git', 'apply', '--binary', str(patch_path)], cwd=destination, check=True)
    relevant_untracked = state.get('source_relevant_untracked',[])
    if relevant_untracked and state['resolved_sha'] != state['head_sha_at_capture']:
        raise ValueError(f'{name}: source-relevant untracked files cannot be combined with a different requested ref')
    for item in relevant_untracked:
        source_file=source/item['path']
        target_file=destination/item['path']
        target_file.parent.mkdir(parents=True,exist_ok=True)
        if source_file.is_symlink():
            target_file.symlink_to(os.readlink(source_file))
        else:
            shutil.copy2(source_file,target_file)
    submodules = []
    if name == 'pyspharm-cpp':
        nested = source / 'third_party' / 'ducc'
        nested_sha = _git(source, 'rev-parse', f'{state["resolved_sha"]}:third_party/ducc')
        observed_nested = _git(nested, 'rev-parse', 'HEAD')
        if observed_nested != nested_sha:
            raise ValueError(f'pyspharm-cpp nested DUCC checkout {observed_nested} differs from gitlink {nested_sha}')
        nested_status = _git(nested, 'status', '--porcelain=v1', '--untracked-files=all')
        if nested_status:
            raise ValueError('pyspharm-cpp nested DUCC submodule is dirty; preserve it and resolve an immutable submodule snapshot before building')
        nested_destination = destination / 'third_party' / 'ducc'
        _extract_archive(nested, nested_sha, nested_destination)
        submodules.append({
            'path': 'third_party/ducc', 'sha': nested_sha,
            'checkout_path': str(nested.resolve()), 'dirty': False,
        })
    state['submodules'] = submodules
    state['build_snapshot_sha256'] = _tree_digest(destination)
    state['build_snapshot_path'] = str(destination.resolve())
    return state


def _run_logged(command: list[str], *, env: dict[str, str], log, cwd: Path | None = None) -> None:
    log.write('+ ' + ' '.join(map(str, command)) + '\n')
    log.flush()
    subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=log, check=True)


def _compiler_provenance(env: dict[str, str]) -> dict:
    result = {'compiler_environment': {key: env.get(key) for key in (
        'CC','CXX','FC','CFLAGS','CXXFLAGS','FFLAGS','LDFLAGS','RUSTFLAGS',
        'DUCC0_OPTIMIZATION','SPHARM_CPP_BUILD_PROFILE','CMAKE_ARGS','CARGO_BUILD_TARGET',
    )}, 'compilers': {}}
    for name in ('CC','CXX','FC'):
        executable = env.get(name)
        if executable:
            try:
                result['compilers'][name] = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
            except (OSError, subprocess.CalledProcessError) as exc:
                result['compilers'][name] = str(exc)
    for name in ('rustc','cargo','cmake','ninja'):
        try:
            result['compilers'][name] = subprocess.run([name, '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError) as exc:
            result['compilers'][name] = str(exc)
    try:
        result['cpu'] = subprocess.run(['lscpu'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        result['cpu'] = {'error': str(exc)}
    for key, command in {
        'fftw_version':['pkg-config','--modversion','fftw3'],
        'fftw_link_flags':['pkg-config','--cflags','--libs','fftw3'],
    }.items():
        try:
            p = subprocess.run(command, capture_output=True, text=True)
            result[key] = {'command':command,'stdout':p.stdout.strip(),'stderr':p.stderr.strip(),'returncode':p.returncode}
        except OSError as exc:
            result[key] = {'error':str(exc)}
    return result


def _collect_build_inputs(source: Path, backend: str, output: Path) -> list[dict]:
    patterns = {
        'ducc': ('build/**/CMakeCache.txt','build/**/build.ninja','build/**/compile_commands.json'),
        'shtns': ('Makefile','config.log','config.status','sht_config.h'),
        'rustsht': ('Cargo.lock','Cargo.toml','python/Cargo.toml','target/release/.fingerprint/**/*.json','python/target/release/.fingerprint/**/*.json'),
        'pyspharm-cpp': ('build/**/CMakeCache.txt','build/**/build.ninja','build/**/compile_commands.json','build/**/cmake_install.cmake'),
    }
    captured = []
    for pattern in patterns.get(backend, ()):
        for item in sorted(source.glob(pattern)):
            if not item.is_file():
                continue
            relative = item.relative_to(source)
            data = item.read_bytes()
            target = output / 'build-metadata' / backend / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            captured.append({'source_path':str(item.resolve()),'file':str(target.resolve()),'sha256':_sha(data)})
    return captured


def _probe_backend(python: Path, backend: str, env: dict[str, str]) -> dict:
    module, distribution = MODULES_FOR_BUILD[backend]
    code = r'''import contextlib,hashlib,importlib,importlib.metadata as md,json,pathlib,subprocess,sys
module_name,dist_name=sys.argv[1:3]
capture=__import__('io').StringIO()
with contextlib.redirect_stdout(capture): m=importlib.import_module(module_name)
module_path=pathlib.Path(m.__file__).resolve()
if dist_name=='pyspharm':
    import numpy as np
    prefix=pathlib.Path(sys.prefix)
    records=[json.loads(f.read_text()) for f in (prefix/'conda-meta').glob('*.json')]
    named=[str(r.get('name','')).lower() for r in records]
    unexpected=sorted(x for x in named if x.startswith('pyspharm') and x!='pyspharm')
    if unexpected: raise RuntimeError('unexpected conda package identity: '+repr(unexpected))
    package=next((r for r in records if str(r.get('name','')).lower()=='pyspharm'),None)
    if package is None: raise RuntimeError('conda pyspharm package record is missing')
    distributions=[]
    for d in md.distributions():
        name=(d.metadata.get('Name') or '').lower().replace('_','-')
        if name.startswith('pyspharm') and name!='pyspharm': raise RuntimeError('unexpected Python distribution: '+name)
        if name=='pyspharm': distributions.append({'name':name,'version':d.version})
    try: native=importlib.import_module('_spherepack')
    except ImportError as exc: raise RuntimeError('compiled _spherepack extension is missing') from exc
    files=[pathlib.Path(native.__file__).resolve()]
    sh=m.Spharmt(16,8,gridtype='gaussian',legfunc='stored',rsphere=1.0)
    spectrum=np.zeros(10,dtype=np.complex64)
    precision={'input_dtype':'complex64','analysis_dtype':str(sh.grdtospec(np.zeros((8,16),dtype=np.float32),ntrunc=3).dtype),'synthesis_dtype':str(sh.spectogrd(spectrum).dtype),'numpy_abi':np._core._multiarray_umath._get_ndarray_c_version()}
    dist={'name':package.get('name'),'version':package.get('version'),'build':package.get('build'),'build_number':package.get('build_number'),'channel':package.get('channel',package.get('schannel')),'url':package.get('url'),'sha256':package.get('sha256'),'md5':package.get('md5'),'depends':package.get('depends',[]),'subdir':package.get('subdir')}
    out={'import_banner':capture.getvalue(),'module_path':str(module_path),'native_module_path':str(files[0]),'distribution_name':dist_name,'distribution_version':package.get('version'),'conda_record':dist,'python':sys.version,'python_version':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'numpy_abi':precision['numpy_abi'],'python_distributions':distributions,'precision_probe':precision,'binaries':[],'native_cpu_info':None}
else:
    dist=md.distribution(dist_name)
    files=[pathlib.Path(dist.locate_file(f)).resolve() for f in dist.files or [] if str(f).endswith(('.so','.pyd'))]
    out={'import_banner':capture.getvalue(),'module_path':str(module_path),'distribution_name':dist.metadata['Name'],'distribution_version':dist.version,'python':sys.version,'numpy':md.version('numpy'),'wheel_metadata':dist.read_text('WHEEL'),'installer':dist.read_text('INSTALLER'),'binaries':[],'native_cpu_info':None}
for f in files:
    ldd=subprocess.run(['ldd',str(f)],capture_output=True,text=True)
    out['binaries'].append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ldd':ldd.stdout,'ldd_returncode':ldd.returncode})
if module_name=='spharm_cpp' and hasattr(m,'cpu_info'):
    try: out['native_cpu_info']=m.cpu_info()
    except Exception as exc: out['native_cpu_info']={'error':str(exc)}
print(json.dumps(out,default=str))'''
    probe_env = env.copy()
    probe_env.pop('PYTHONPATH', None)
    result = subprocess.check_output([str(python), '-I', '-c', code, module, distribution], env=probe_env, text=True)
    record = json.loads(result)
    if not Path(record['module_path']).is_relative_to(python.parent.parent):
        raise ValueError(f'{backend}: import escaped its isolated environment')
    if not record['binaries']:
        raise ValueError(f'{backend}: no native extension binary found')
    # Keep the virtual-environment path. Resolving this symlink points at the
    # base interpreter and drops pyvenv.cfg/site-packages when workers start.
    record['python_executable'] = str(python.absolute())
    record['freeze'] = subprocess.check_output(['uv','pip','freeze','--python',str(python)],env=env,text=True)
    return record


def build_local_sources(
    output: Path, selected: tuple[str, ...], *, python_version: str,
    source_overrides: dict[str, Path], ref_overrides: dict[str, str],
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError('build output directory already contains data; choose a new output path')
    source_backends = tuple(name for name in selected if name != 'pyspharm')
    source_state = {}
    snapshots = {}
    for backend in source_backends:
        default, default_ref = SOURCES[backend]
        path = source_overrides.get(backend, (ROOT / default).resolve())
        ref = ref_overrides.get(backend, default_ref)
        source_state[backend] = _source_state(path, ref)
        snapshot = output / 'source-snapshots' / backend
        snapshots[backend] = _make_source_snapshot(backend, source_state[backend], snapshot)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name, default in (('CC','gcc'),('CXX','g++'),('FC','gfortran')):
        env.setdefault(name, shutil.which(default) or default)
    env.setdefault('RUSTFLAGS','-C target-cpu=native')
    env.setdefault('DUCC0_OPTIMIZATION','native')
    env.setdefault('SPHARM_CPP_BUILD_PROFILE','multiarch')
    provenance = _compiler_provenance(env)
    manifest = {
        'schema': 1,
        'profile_schema': PROFILE_SCHEMA,
        'source_policy': 'local Git checkouts; native builds use immutable source snapshots under source-snapshots/',
        'sources': snapshots,
        'builds': {},
        'selected_backends': list(selected),
        'conda_policy': {'channel':CONDA_CHANNEL,'python':PYTHON_VERSION,'numpy':NUMPY_VERSION,'pyspharm':PYSPHARM_VERSION},
        **provenance,
    }
    output.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        entry = {'status':'not_requested'}
        manifest['builds'][backend] = entry
        if backend not in selected:
            continue
        venv = output / 'envs' / backend
        python = conda_python(venv) if backend == 'pyspharm' else venv / 'bin' / 'python'
        log_path = output / f'build-{backend}.log'
        with log_path.open('w') as log:
            try:
                if backend == 'pyspharm':
                    conda = find_conda()
                    command = conda_create_command(conda,venv,python=PYTHON_VERSION,numpy=NUMPY_VERSION,pyspharm=PYSPHARM_VERSION)
                    entry.update({'build_kind':'conda-forge','build_origin':CONDA_CHANNEL,'conda_executable':conda,'conda_create_command':command})
                    _run_logged(command,env=env,log=log)
                    package = verify_pyspharm_environment(venv)
                    entry.update({'resolved_conda_package':package,'sha':package.get('sha256') or f"{CONDA_CHANNEL}::pyspharm={package['version']}#{package['build_string']}"})
                    _run_logged(['uv','pip','install','--python',str(python),'-e',str(ROOT),f'numpy=={NUMPY_VERSION}'],env=env,log=log)
                    exports = installed_environment_exports(conda,venv)
                    explicit = output/'pyspharm-conda-explicit.txt'
                    environment = output/'pyspharm-conda-environment.yml'
                    explicit.write_text(exports['explicit'])
                    environment.write_text(exports['environment'])
                    entry['conda_exports'] = {'explicit_file':str(explicit.resolve()),'explicit_sha256':_sha(explicit.read_bytes()),'environment_file':str(environment.resolve())}
                else:
                    _run_logged(['uv','venv','--python',python_version,str(venv)],env=env,log=log)
                    _run_logged(['uv','pip','install','--python',str(python),'-e',str(ROOT),f'numpy=={NUMPY_VERSION}'],env=env,log=log)
                    source = Path(snapshots[backend]['build_snapshot_path'])
                    entry.update(snapshots[backend])
                    entry['build_kind'] = 'local-git-source-snapshot'
                    command = ['uv','pip','install','--python',str(python),'--verbose','--no-cache',str(source)]
                    if backend == 'pyspharm-cpp':
                        command[3:3] = ['--config-setting','cmake.define.SPHARM_CPP_BUILD_PROFILE=multiarch']
                    entry['build_command'] = command
                    build_env = env.copy()
                    if backend == 'pyspharm-cpp':
                        build_env['SPHARM_CPP_BUILD_PROFILE'] = 'multiarch'
                    _run_logged(command,env=build_env,log=log)
                    entry['compiler_metadata_files'] = _collect_build_inputs(source,backend,output)
                module_record = _probe_backend(python,backend,env)
                if backend == 'pyspharm':
                    verify_import_path(venv,module_record['module_path'])
                entry.update(module_record)
                entry['status'] = 'built'
            except (subprocess.SubprocessError,OSError,ValueError,KeyError) as exc:
                entry['status'] = 'unavailable' if backend == 'pyspharm' else 'build_failed'
                entry['error'] = str(exc)
        write_json(output / 'build-manifest.json',manifest)
    write_json(output / 'build-manifest.json',manifest)
    return manifest


def _parse_csv(values: str, *, allowed: tuple[str, ...] | None = None, positive_int: bool = False) -> tuple:
    items = tuple(item.strip() for item in values.split(',') if item.strip())
    if not items or len(set(items)) != len(items):
        raise ValueError('selection must be a non-empty comma-separated list without duplicates')
    if positive_int:
        try:
            numbers = tuple(int(item) for item in items)
        except ValueError as exc:
            raise ValueError('thread counts must be integers') from exc
        if any(number < 1 for number in numbers):
            raise ValueError('thread counts must be positive')
        return numbers
    if allowed is not None and set(items) - set(allowed):
        raise ValueError(f'unknown selection {sorted(set(items)-set(allowed))}; expected values from {allowed}')
    return items


def _cell_id(cell: dict, *, source_sha: str, build_hash: str, harness_sha: str) -> str:
    identity = {
        'backend':cell['backend'], 'workload':cell['workload'],
        'threads_requested':cell['threads_requested'], 'round':cell['round'],
        'source_sha':source_sha,'build_manifest_sha256':build_hash,'harness_sha256':harness_sha,
    }
    return _sha(_canonical(identity))[:24]


def _validation_id(cell: dict, *, source_sha: str, build_hash: str, harness_sha: str) -> tuple[str, dict]:
    pseudo = {
        'backend':cell['backend'], 'source_sha':source_sha,
        'build_manifest_sha256':build_hash,'harness_sha256':harness_sha,
        'threads_requested':cell['threads_requested'],
        'workload':{**cell['workload'],'operation':'synthesis','allocation':'allocating'},
    }
    identity = validation_identity(pseudo)
    return _sha(_canonical(identity))[:24], identity


def _same_task(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    previous = json.loads(path.read_text())
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f'incompatible resume evidence in {path}: field {key!r} changed')
    return True


def _write_outcome(path: Path, task: dict, status: str, reason: str | None = None, **extra) -> None:
    result = {**task, 'status':status, **extra}
    if reason:
        result['reason'] = reason
    write_json(path,result)


def _execute_worker(task: dict, *, path: Path, python: Path, timeout: float, env: dict[str, str]) -> dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    task_path = path.parent.parent / 'tasks' / f'{task["id"]}.json'
    task_path.parent.mkdir(parents=True,exist_ok=True)
    write_json(task_path,task)
    worker_env = env.copy()
    q = task['threads_requested']
    worker_env.update({
        'OMP_NUM_THREADS':str(q), 'RAYON_NUM_THREADS':str(q), 'DUCC0_NUM_THREADS':str(q),
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'SHT_BENCH_REFERENCE_CACHE':str((ROOT / '.builds' / 'source-resolution-reference' / task['harness_sha256']).resolve()),
    })
    log_path = path.with_suffix('.log')
    with log_path.open('w') as log:
        try:
            proc = subprocess.run(
                [str(python),'-m','sht_bench.source_runner','--task',str(task_path),'--result',str(path)],
                env=worker_env,stdout=log,stderr=log,timeout=timeout,
            )
            if not path.exists():
                _write_outcome(path,task,'execution_failed',f'worker exited {proc.returncode} without writing a result')
        except subprocess.TimeoutExpired:
            _write_outcome(path,task,'resource_limited',f'worker exceeded per-cell timeout of {timeout:g} seconds')
    try:
        return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError) as exc:
        _write_outcome(path,task,'malformed_json',str(exc))
        return json.loads(path.read_text())


def _load_or_create_run_config(output: Path, config: dict, *, resume: bool) -> None:
    path = output / 'run-config.json'
    if resume:
        if not path.exists():
            raise ValueError('--resume requires an existing run-config.json')
        if json.loads(path.read_text()) != config:
            raise ValueError('incompatible resume: build, harness, profile, or selected workload changed')
    else:
        if output.exists() and any((output/name).exists() for name in ('run-config.json','workload-plan.json','cells','validations')):
            raise ValueError('output directory already contains run data; select a new path or pass --resume')
        output.mkdir(parents=True,exist_ok=True)
        write_json(path,config)


def _write_immutable_json(path: Path, data: dict, *, resume: bool) -> None:
    if resume and path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f'incompatible resume metadata: {path} changed')
        return
    write_json(path,data)


def _run_sweep(args, cases: list[dict], cells: list[dict], build_path: Path) -> dict:
    output = args.output.resolve()
    if not output.exists() and args.resume:
        raise ValueError('--resume output directory does not exist')
    build_manifest = json.loads(build_path.read_text())
    build_hash = _sha(build_path.read_bytes())
    output.mkdir(parents=True,exist_ok=True)
    local_manifest = output/'build-manifest.json'
    if build_path.resolve() != local_manifest.resolve():
        if local_manifest.exists():
            if local_manifest.read_bytes() != build_path.read_bytes():
                raise ValueError('output build manifest conflicts with the selected immutable build manifest')
        else:
            shutil.copyfile(build_path,local_manifest)
    harness_sha = harness_hash()
    sweep_sha = _sha(Path(__file__).read_bytes())
    build_mode = {
        'build_manifest':str(build_path.resolve()),
        'build_manifest_sha256':build_hash,
        'harness_sha256':harness_sha,
        'sweep_script_sha256':sweep_sha,
        'profile_schema':PROFILE_SCHEMA,
        'grids':list(args.grid_values), 'backends':list(args.backend_values),
        'threads':list(args.thread_values), 'precisions':list(args.precision_values),
        'mmax_modes':list(args.mmax_values), 'rounds':args.rounds,
        'repeat':args.repeat,'min_time':args.min_time,'timeout_s':args.timeout,
        'memory_limit_bytes':None if args.memory_limit_gib is None else int(args.memory_limit_gib * 2**30),
        'endpoints_only':args.smoke,
    }
    _load_or_create_run_config(output,build_mode,resume=args.resume)
    output.mkdir(parents=True,exist_ok=True)
    build_by_backend = build_manifest.get('builds',{})
    for backend,entry in build_by_backend.items():
        if entry.get('status') != 'built':
            continue
        for binary in entry.get('binaries',[]):
            p = Path(binary['path'])
            if not p.exists() or _sha(p.read_bytes()) != binary['sha256']:
                raise ValueError(f'{backend}: installed native binary changed since build')
    for cell in cells:
        build = build_by_backend.get(cell['backend'],{})
        source_sha = build.get('sha',build.get('requirement','unknown'))
        cell['source_sha'] = source_sha
        cell['id'] = _cell_id(cell,source_sha=source_sha,build_hash=build_hash,harness_sha=harness_sha)
        cell['validation_id'] = _validation_id(cell,source_sha=source_sha,build_hash=build_hash,harness_sha=harness_sha)[0]
    profile_record = {'schema':PROFILE_SCHEMA,'profile_version':1,'cases':cases}
    _write_immutable_json(output/'resolution-profile.json',profile_record,resume=args.resume)
    static_counts = plan_counts(iter(cells))
    plan_record = {
        'schema':SCHEMA, 'configuration':build_mode,
        'counts':static_counts,
        'cells':cells,
        'latitude_coordinates_artifact':'resolution-profile.json:cases[].latitude_coordinates_radians',
        'unsupported_policy':'native full triangular truncation only for pyspharm-cpp and pyspharm; half mmax is recorded unsupported, never emulated by zeroing coefficients',
    }
    _write_immutable_json(output/'workload-plan.json',plan_record,resume=args.resume)
    build_hash = build_mode['build_manifest_sha256']
    harness_sha = build_mode['harness_sha256']
    physical, _ = physical_cores()
    memory_cap = build_mode['memory_limit_bytes']
    env = os.environ.copy()
    lock_path = ROOT / '.builds' / 'benchmark.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('w') as lock_handle:
        import fcntl
        try:
            fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another source benchmark controller is running') from exc

        # Generate one independent numerical gate for each backend, geometry,
        # precision, order, spin, and thread setting. Both transform directions
        # and all independent rounds require this exact gate before timing.
        validations = {}
        for cell in cells:
            if cell['static_status'] != 'planned':
                continue
            build = build_by_backend.get(cell['backend'],{})
            if build.get('status') != 'built':
                continue
            validation_id, identity = _validation_id(cell,source_sha=build.get('sha',build.get('requirement','unknown')),build_hash=build_hash,harness_sha=harness_sha)
            validations.setdefault(validation_id,{'id':validation_id,'identity':identity,'backend':cell['backend'],'threads_requested':cell['threads_requested'],'workload':{**cell['workload'],'operation':'synthesis','allocation':'allocating'},'build':build})
        _write_immutable_json(output/'validation-plan.json',{'schema':1,'validation_count':len(validations),'validations':[{'id':v['id'],'identity':v['identity']} for v in validations.values()]},resume=args.resume)
        validation_results = {}
        validation_groups = defaultdict(list)
        for item in validations.values():
            w = item['workload']
            group = (w['grid'],w['nlat'],w['nlon'],w['lmax'],w['mmax'],w['spin'],w['precision'],item['threads_requested'])
            validation_groups[group].append(item)
        group_seed = int(build_hash[:16],16)
        rng = random.Random(group_seed)
        for group in sorted(validation_groups):
            order = validation_groups[group][:]
            rng.shuffle(order)
            for item in order:
                backend = item['backend']; build = item['build']; q = item['threads_requested']
                result_path = output/'validations'/f'{item["id"]}.json'
                identity = item['identity']
                source_sha = build.get('sha',build.get('requirement','unknown'))
                vtask = {
                    'id':item['id'],'backend':backend,'workload':item['workload'],
                    'threads_requested':q,'source_sha':source_sha,
                    'build_mode':build.get('build_kind'),'build_manifest':str(build_path.resolve()),
                    'build_manifest_sha256':build_hash,'harness_sha256':harness_sha,
                    'expected_import_path':build.get('module_path'),
                    'repeat':args.repeat,'min_time':args.min_time,
                    'coefficient_layout':'canonical m-major packed; orthonormal; Condon-Shortley phase; real-field positive-order factor two',
                    'planning_policy':PLANNING_POLICIES[backend],
                    'validation_only':True,'validation_identity':identity,
                }
                if args.resume and result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous.get('validation_identity') != identity:
                        raise ValueError(f'incompatible validation resume: {result_path}')
                    validation_results[item['id']] = previous
                    continue
                if q > len(physical):
                    _write_outcome(result_path,vtask,'resource_limited','requested thread count exceeds available physical cores',validation_identity=identity)
                else:
                    admission = preflight(backend,Workload(**item['workload']),limit_bytes=memory_cap)
                    vtask['resource_preflight'] = admission
                    if admission['resource_limited']:
                        _write_outcome(result_path,vtask,'resource_limited',admission.get('reason'),validation_identity=identity)
                    else:
                        _execute_worker(vtask,path=result_path,python=Path(build['python_executable']),timeout=args.timeout,env=env)
                validation_results[item['id']] = json.loads(result_path.read_text())
                print(f'VALIDATE {backend} {group} {validation_results[item["id"]]["status"]}',flush=True)

        status_counts = defaultdict(int)
        result_groups = defaultdict(list)
        for cell in cells:
            case = cell['resolution_case']
            w = cell['workload']
            group = (w['grid'],w['nlat'],w['nlon'],w['lmax'],w['mmax'],w['spin'],w['precision'],w['operation'],cell['threads_requested'],cell['round'])
            result_groups[group].append(cell)
        for group in sorted(result_groups):
            order = result_groups[group][:]
            rng.shuffle(order)
            for cell in order:
                backend = cell['backend']; build = build_by_backend.get(backend,{})
                source_sha = build.get('sha',build.get('requirement','unknown'))
                identity_id = _cell_id(cell,source_sha=source_sha,build_hash=build_hash,harness_sha=harness_sha)
                result_path = output/'cells'/f'{identity_id}.json'
                validation_id, _ = _validation_id(cell,source_sha=source_sha,build_hash=build_hash,harness_sha=harness_sha)
                task = {
                    **cell,'id':identity_id,'source_sha':source_sha,
                    'build_mode':build.get('build_kind'),'build_manifest':str(build_path.resolve()),
                    'build_manifest_sha256':build_hash,'harness_sha256':harness_sha,
                    'expected_import_path':build.get('module_path'),
                    'repeat':args.repeat,'min_time':args.min_time,
                    'coefficient_layout':'canonical m-major packed; orthonormal; Condon-Shortley phase; real-field positive-order factor two',
                    'planning_policy':PLANNING_POLICIES[backend],
                    'resolution_case':cell['resolution_case'],'validation_id':validation_id,
                }
                if args.resume and result_path.exists():
                    _same_task(result_path,task)
                    previous=json.loads(result_path.read_text())
                    status_counts[previous.get('status','malformed_json')]+=1
                    continue
                if cell['static_status'] == 'unsupported':
                    _write_outcome(result_path,task,'unsupported',cell['static_reason'])
                elif build.get('status') != 'built':
                    _write_outcome(result_path,task,build.get('status','unavailable'),build.get('error','backend build unavailable'))
                elif cell['threads_requested'] > len(physical):
                    _write_outcome(result_path,task,'resource_limited','requested thread count exceeds available physical cores')
                else:
                    validation = validation_results.get(validation_id)
                    if validation is None:
                        _write_outcome(result_path,task,'execution_failed','pre-timing numerical validation outcome is missing')
                    elif validation.get('status') != 'validation_passed':
                        propagated = validation.get('status')
                        status = propagated if propagated in ('unsupported','numerical_validation_failed','resource_limited','build_failed','unavailable','execution_failed') else 'numerical_validation_failed'
                        reason = validation.get('reason') or 'cell was not timed because its matching pre-timing validation gate did not pass'
                        _write_outcome(result_path,task,status,reason,validation=validation.get('validation'),validation_result_path=str((output/'validations'/f'{validation_id}.json').relative_to(output)))
                    else:
                        validation_path = output/'validations'/f'{validation_id}.json'
                        timed_task = {
                            **task,
                            'validation_certificate':str(validation_path.resolve()),
                            'validation_certificate_sha256':_sha(validation_path.read_bytes()),
                        }
                        result = _execute_worker(timed_task,path=result_path,python=Path(build['python_executable']),timeout=args.timeout,env=env)
                        status_counts[result.get('status','malformed_json')]+=1
                        print(f'{identity_id} {backend} {w["grid"]} L{w["lmax"]} m{w["mmax"]} s{w["spin"]} {w["precision"]} q{cell["threads_requested"]} {w["operation"]} r{cell["round"]} {result.get("status")}',flush=True)
                        continue
                status_counts[json.loads(result_path.read_text()).get('status','malformed_json')]+=1
        audit = audit_cells(output,plan_record,build_hash=build_hash,harness_sha=harness_sha)
        write_json(output/'final-audit.json',audit)
        return audit


def audit_cells(output: Path, plan: dict, *, build_hash: str, harness_sha: str) -> dict:
    expected = plan['cells']
    paths = {p.stem:p for p in (output/'cells').glob('*.json')}
    build_manifest = json.loads((output/'build-manifest.json').read_text()) if (output/'build-manifest.json').exists() else None
    results = {}
    issues = []
    expected_by_id = {cell['id']:cell for cell in expected if cell.get('id')}
    for cell in expected:
        if build_manifest is None:
            issues.append('build manifest missing')
            break
        build = build_manifest.get('builds',{}).get(cell['backend'],{})
        source_sha = build.get('sha',build.get('requirement','unknown'))
        key = cell.get('id') or _cell_id(cell,source_sha=source_sha,build_hash=build_hash,harness_sha=harness_sha)
        path = paths.get(key)
        if path is None:
            continue
        try:
            result=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError) as exc:
            issues.append(f'{path.name}: malformed JSON: {exc}')
            continue
        if result.get('id') != key or result.get('build_manifest_sha256') != build_hash or result.get('harness_sha256') != harness_sha:
            issues.append(f'{path.name}: identity/provenance mismatch')
        for field in ('backend','workload','threads_requested','round','source_sha','resolution_case','validation_id'):
            if result.get(field) != cell.get(field):
                issues.append(f'{path.name}: {field} differs from the locked workload plan')
                break
        status = result.get('status')
        if status == 'timed_successfully':
            samples = result.get('samples_s')
            if not isinstance(samples,list) or not samples or any(not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0 for value in samples):
                issues.append(f'{path.name}: successful timing has missing/nonfinite/nonpositive raw samples')
            if result.get('validation',{}).get('status') != 'supported_and_validated':
                issues.append(f'{path.name}: successful timing lacks a passing numerical validation report')
            if cell.get('static_status') == 'unsupported':
                issues.append(f'{path.name}: unsupported API cell was timed')
            certificate_path = result.get('validation_certificate')
            certificate_hash = result.get('validation_certificate_sha256')
            if not certificate_path or not certificate_hash:
                issues.append(f'{path.name}: timing lacks a pre-timing validation certificate')
            else:
                certificate = Path(certificate_path)
                if not certificate.exists() or _sha(certificate.read_bytes()) != certificate_hash:
                    issues.append(f'{path.name}: validation certificate is missing or has a checksum mismatch')
                else:
                    gate = json.loads(certificate.read_text())
                    if gate.get('status') != 'validation_passed' or gate.get('validation_identity') is None:
                        issues.append(f'{path.name}: referenced validation certificate did not pass')
        elif 'samples_s' in result and status != 'timed_successfully':
            issues.append(f'{path.name}: non-success outcome retained timing samples')
        if cell.get('static_status') == 'unsupported' and status != 'unsupported':
            issues.append(f'{path.name}: statically unsupported API cell was not recorded unsupported')
        results[key] = result
    expected_ids = set(expected_by_id)
    if build_manifest is None:
        build_manifest = {}
    unexpected = sorted(set(paths)-expected_ids)
    if unexpected:
        issues.append(f'{len(unexpected)} unexpected cell JSON files')
    counts = defaultdict(int)
    for value in results.values():
        counts[value.get('status','missing_status')]+=1
    missing = sorted(expected_ids-set(results))
    return {
        'schema':1,'expected_cells':len(expected_ids),'recorded_cells':len(results),
        'missing_cells':len(missing),'unexpected_cells':len(unexpected),
        'status_counts':dict(sorted(counts.items())),
        'evidence_issues':issues,'coverage_complete':not missing and not unexpected,
        'evidence_consistent':not issues,'all_requested_cells_accounted_for':not missing and not unexpected and not issues,
        'missing_examples':missing[:20], 'unexpected_examples':unexpected[:20],
        'build_manifest_sha256':build_hash,'harness_sha256':harness_sha,
        'note':'A complete ledger can include unsupported, validation-failed, build-failed, resource-limited, and execution-failed outcomes; only timed_successfully cells contribute performance comparisons.',
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='new raw result directory (never overwritten)')
    p.add_argument('--build-manifest',type=Path,help='use an existing sweep build manifest')
    p.add_argument('--build',action='store_true',help='build isolated environments from the local source checkouts')
    p.add_argument('--python',default='3.14.8',help='uv Python version for isolated backend environments')
    p.add_argument('--ducc-source',type=Path)
    p.add_argument('--shtns-source',type=Path)
    p.add_argument('--rustsht-source',type=Path)
    p.add_argument('--pyspharm-cpp-source',type=Path)
    p.add_argument('--ducc-ref')
    p.add_argument('--shtns-ref')
    p.add_argument('--rustsht-ref',default='main')
    p.add_argument('--pyspharm-cpp-ref')
    p.add_argument('--grid',default='CC,GL',help='CC,GL selection')
    p.add_argument('--backend',default=','.join(BACKENDS),help='comma-separated backend selection')
    p.add_argument('--threads',default=','.join(map(str,THREADS)))
    p.add_argument('--precision',default=','.join(PRECISIONS))
    p.add_argument('--mmax',default='full,half',help='full,half; lmax stays fixed')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--repeat',type=int,default=30)
    p.add_argument('--min-time',type=float,default=.02)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--memory-limit-gib',type=float)
    p.add_argument('--smoke',action='store_true',help='run CC 2.5° and GL N32 cases only')
    p.add_argument('--plan-only',action='store_true',help='print exact API-supported/unsupported outcome counts')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--replot',action='store_true',help='audit/export/plot an existing result directory without transforms')
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.grid_values = _parse_csv(args.grid,allowed=('CC','GL'))
        args.backend_values = _parse_csv(args.backend,allowed=BACKENDS)
        args.thread_values = _parse_csv(args.threads,positive_int=True)
        args.precision_values = _parse_csv(args.precision,allowed=PRECISIONS)
        args.mmax_values = _parse_csv(args.mmax,allowed=('full','half'))
    except ValueError as exc:
        parser().error(str(exc))
    if args.rounds < 1 or args.repeat < 1 or not all(math.isfinite(x) and x > 0 for x in (args.min_time,args.timeout)):
        parser().error('rounds, repeat, min-time, and timeout must be positive; time values must be finite')
    if args.memory_limit_gib is not None and (not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0):
        parser().error('--memory-limit-gib must be finite and positive')
    if args.plan_only and (args.build or args.resume or args.replot):
        parser().error('--plan-only cannot be combined with --build, --resume, or --replot')
    if args.replot:
        if args.output is None or not (args.output/'workload-plan.json').exists():
            parser().error('--replot requires --output with a saved workload plan')
        from sht_bench.source_resolution_report import render
        render(args.output.resolve())
        return 0
    cases = resolution_cases(args.grid_values,endpoints_only=args.smoke)
    cells = list(planned_cells(
        cases,backends=args.backend_values,threads=args.thread_values,
        precisions=args.precision_values,mmax_modes=args.mmax_values,rounds=args.rounds,
    ))
    counts = plan_counts(iter(cells))
    if args.plan_only:
        print(json.dumps({
            'schema':PROFILE_SCHEMA,'cases':[{k:v for k,v in case.items() if k not in ('latitude_coordinates_radians','colatitude_coordinates_radians')} for case in cases],
            'outcomes':counts,'validation_gate_count_upper_bound':len({
                (cell['backend'],cell['workload']['grid'],cell['workload']['nlat'],cell['workload']['nlon'],cell['workload']['lmax'],cell['workload']['mmax'],cell['workload']['spin'],cell['workload']['precision'],cell['threads_requested'])
                for cell in cells if cell['static_status']=='planned'
            }),
            'precision_rule':'SHTns only float64; conda-forge pyspharm uses float32; other selected backends use requested precisions',
            'half_order_rule':'pyspharm-cpp and pyspharm public APIs lack independent native mmax; half-order cells are explicit unsupported outcomes',
        },indent=2))
        return 0
    if args.output is None:
        parser().error('--output is required unless --plan-only is used')
    output = args.output.expanduser().resolve()
    if args.resume and not (output/'run-config.json').exists():
        parser().error('--resume requires an existing run-config.json to validate source/build/harness identity')
    if args.build and args.resume and (output/'build-manifest.json').exists():
        build_path = output/'build-manifest.json'
    elif args.build:
        if args.build_manifest:
            parser().error('--build-manifest cannot be combined with --build')
        source_overrides = {
            'ducc':args.ducc_source or (ROOT/'../ducc'),
            'shtns':args.shtns_source or (ROOT/'../shtns'),
            'rustsht':args.rustsht_source or (ROOT/'../RustSHT'),
            'pyspharm-cpp':args.pyspharm_cpp_source or (ROOT/'../pyspharm-cpp'),
        }
        ref_overrides = {
            'ducc':args.ducc_ref or 'HEAD','shtns':args.shtns_ref or 'HEAD',
            'rustsht':args.rustsht_ref,'pyspharm-cpp':args.pyspharm_cpp_ref or 'HEAD',
        }
        manifest = build_local_sources(output,args.backend_values,python_version=args.python,source_overrides=source_overrides,ref_overrides=ref_overrides)
        build_path = output/'build-manifest.json'
    elif args.build_manifest:
        build_path = args.build_manifest.expanduser().resolve()
    elif args.resume and (output/'build-manifest.json').exists():
        build_path = output/'build-manifest.json'
    else:
        parser().error('execution requires --build or --build-manifest')
    if not build_path.exists():
        parser().error(f'build manifest does not exist: {build_path}')
    audit = _run_sweep(args,cases,cells,build_path)
    from sht_bench.source_resolution_report import render
    audit = render(output)
    print(json.dumps(audit,indent=2))
    return 0 if audit['all_requested_cells_accounted_for'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
