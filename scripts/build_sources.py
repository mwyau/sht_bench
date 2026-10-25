"""Build isolated Git backends and a separate conda-forge Spherepack environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from sht_bench.conda_environment import (
    CONDA_CHANNEL,
    NUMPY_VERSION,
    PYTHON_VERSION,
    PYSPHARM_VERSION,
    create_command as conda_create_command,
    find_conda,
    installed_environment_exports,
    runtime_probe,
    verify_import_path,
    verify_pyspharm_environment,
)
from sht_bench.sources import checkout, git

MODULES = {'ducc': 'ducc0', 'shtns': 'shtns', 'rustsht': 'rustsht', 'pyspharm': 'spharm'}


def collect_build_files(source: Path, backend: str, output: Path) -> list[dict]:
    """Archive generated compiler/linker rules and Cargo dependency/profile inputs."""
    patterns = {
        'ducc': ('build/*/CMakeCache.txt', 'build/*/build.ninja', 'build/*/compile_commands.json'),
        'shtns': ('Makefile', 'config.log', 'config.status', 'sht_config.h'),
        'rustsht': ('Cargo.lock', 'Cargo.toml', 'python/Cargo.toml',
                    'target/release/.fingerprint/rustsht*/lib-*.json',
                    'python/target/release/.fingerprint/rustsht*/lib-*.json',
                    'target/release/.fingerprint/pulp-*/lib-pulp.json',
                    'python/target/release/.fingerprint/pulp-*/lib-pulp.json'),
    }
    artifacts = []
    for pattern in patterns.get(backend, ()):
        for path in sorted(source.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = output / 'build-metadata' / backend / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError(f'compiler metadata already exists with different contents: {destination}')
            destination.write_bytes(data)
            artifacts.append({'source_path':str(path), 'file':str(destination.resolve()),
                              'sha256':hashlib.sha256(data).hexdigest()})
    return artifacts


def run(command, *, env, log):
    log.write('+ ' + ' '.join(map(str, command)) + '\n')
    log.flush()
    subprocess.run(list(map(str, command)), env=env, stdout=log, stderr=log, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.sources'))
    parser.add_argument('--mode', choices=('latest', 'locked'), default='latest')
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--python', default='3.13')
    parser.add_argument('--backend', choices=tuple(MODULES), action='append',
                        help='backend(s) to build; defaults to every supported backend')
    for name in ('ducc', 'shtns', 'rustsht'):
        parser.add_argument(f'--{name}-ref')
    args = parser.parse_args(argv)
    selected_backends = tuple(args.backend or MODULES)
    if (args.output / 'build-manifest.json').exists():
        parser.error('build manifest already exists; choose a new output directory')
    if args.mode == 'locked' and args.lock is None:
        parser.error('--mode locked requires --lock from the previous run')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.lock or args.output / 'sources.lock.json'
    selected_sources = tuple(name for name in selected_backends if name != 'pyspharm')
    sources = (
        checkout(args.root.resolve(), lock, args.mode,
                 {n: getattr(args, n + '_ref') for n in selected_sources if getattr(args, n + '_ref')})
        if selected_sources else {'schema': 1, 'sources': {}}
    )
    env = os.environ.copy()
    for key, default in [('CC', 'gcc'), ('CXX', 'g++'), ('FC', 'gfortran')]:
        env.setdefault(key, shutil.which(default) or default)
    env.setdefault('RUSTFLAGS', '-C target-cpu=native')
    env.setdefault('DUCC0_OPTIMIZATION', 'native')
    manifest = {'schema': 1, 'sources': sources, 'builds': {}, 'compiler_environment': {
        k: env.get(k) for k in ('CC', 'CXX', 'FC', 'CFLAGS', 'CXXFLAGS', 'FFLAGS', 'LDFLAGS', 'RUSTFLAGS', 'DUCC0_OPTIMIZATION', 'CMAKE_ARGS', 'CARGO_BUILD_TARGET')},
        'compilers': {}}
    for name, command in [('cpu', ['lscpu']), ('fftw_version', ['pkg-config', '--modversion', 'fftw3']),
                          ('fftw_flags', ['pkg-config', '--cflags', '--libs', 'fftw3'])]:
        try:
            p = subprocess.run(command, capture_output=True, text=True)
            manifest[name] = {'command':command,'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode}
        except OSError as exc:
            manifest[name] = {'error':str(exc)}
    for compiler in (env['CC'], env['CXX'], env['FC'], 'rustc', 'cargo'):
        try:
            p = subprocess.run([compiler, '--version'], capture_output=True, text=True)
            manifest['compilers'][compiler] = p.stdout + p.stderr
        except OSError as exc:
            manifest['compilers'][compiler] = str(exc)
    for backend in selected_backends:
        module = MODULES[backend]
        build = {'status': 'build_failed'}
        manifest['builds'][backend] = build
        venv = (args.output / 'envs' / backend).resolve()
        python = venv / 'bin/python'
        with (args.output / f'build-{backend}.log').open('w') as log:
            try:
                if backend == 'pyspharm':
                    conda = find_conda()
                    if venv.exists():
                        raise ValueError(f'conda prefix already exists: {venv}')
                    build['build_kind'] = 'conda-forge'
                    build['build_origin'] = CONDA_CHANNEL
                    build['conda_executable'] = conda
                    build['conda_create_command'] = conda_create_command(
                        conda, venv, python=PYTHON_VERSION, numpy=NUMPY_VERSION,
                        pyspharm=PYSPHARM_VERSION,
                    )
                    run(build['conda_create_command'], env=env, log=log)
                    package = verify_pyspharm_environment(venv)
                    build['resolved_conda_package'] = package
                    build['sha'] = package.get('sha256') or (
                        f"{CONDA_CHANNEL}::pyspharm={package['version']}#{package['build_string']}"
                    )
                else:
                    run(['uv', 'venv', '--python', args.python, venv], env=env, log=log)
                run(['uv', 'pip', 'install', '--python', python, '-e', '.[plot]', 'pytest', 'scipy'], env=env, log=log)
                if backend != 'pyspharm':
                    path = args.root.resolve() / backend
                    if git(path, 'rev-parse', 'HEAD') != sources['sources'][backend]['sha']:
                        raise ValueError('source changed after acquisition')
                    build.update(sources['sources'][backend])
                    build['build_kind'] = 'git-source'
                    command = ['uv', 'pip', 'install', '--verbose', '--python', python, '--no-cache', str(path)]
                    build['build_command'] = list(map(str,command))
                    run(command, env=env, log=log)
                # Run outside the checkout with no PYTHONPATH. Hash imported extensions and inspect linkage.
                probe = '''import importlib,json,hashlib,pathlib,sys,subprocess,importlib.metadata as md,io,contextlib
banner=io.StringIO()
with contextlib.redirect_stdout(banner):
 m=importlib.import_module(sys.argv[1])
p=pathlib.Path(m.__file__).resolve()
import numpy as np
distribution_name=sys.argv[2]
if distribution_name=='pyspharm':
 records=[json.loads(f.read_text()) for f in (pathlib.Path(sys.prefix)/'conda-meta').glob('*.json')]
 package=next((r for r in records if r.get('name','').lower()=='pyspharm'),None)
 if package is None or any(r.get('name','').lower().startswith('pyspharm') and r.get('name','').lower()!='pyspharm' for r in records): raise RuntimeError('unexpected conda package identity')
 try: dist=md.distribution('pyspharm')
 except md.PackageNotFoundError: dist=None
 package_dir=p.parent
 files=list(package_dir.glob('*.so'))+list(package_dir.glob('*.pyd'))
 files += list(package_dir.parent.glob('_spherepack*.so'))+list(package_dir.parent.glob('_spherepack*.pyd'))
 if dist is not None: version=dist.version; name=dist.metadata.get('Name')
 else: version=package.get('version'); name=package.get('name')
 python_version='.'.join(map(str,sys.version_info[:3]))
 numpy_abi=np._core._multiarray_umath._get_ndarray_c_version()
 spharm=importlib.import_module('spharm')
 plan=spharm.Spharmt(16,8,gridtype='gaussian',legfunc='stored',rsphere=1.0)
 coeff=plan.grdtospec(np.zeros((8,16),dtype=np.float32),ntrunc=3)
 mapped=plan.spectogrd(np.zeros(10,dtype=np.complex64))
 print(json.dumps({'import_banner':banner.getvalue(),'module_path':str(p),'python':sys.version,'python_version':python_version,'numpy':np.__version__,'numpy_abi':numpy_abi,'distribution_name':name,'distribution_version':version,'conda_record':package,'precision_probe':{'input_dtype':'float32','analysis_dtype':str(coeff.dtype),'synthesis_dtype':str(mapped.dtype)},'binaries':[{'path':str(f.resolve()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ldd':subprocess.run(['ldd',str(f)],capture_output=True,text=True).stdout} for f in files],'conda_packages':[{'name':r.get('name'),'version':r.get('version'),'build':r.get('build'),'channel':r.get('channel',r.get('schannel')),'url':r.get('url'),'sha256':r.get('sha256')} for r in records if r.get('name','').lower() in ('pyspharm','numpy','python','libgfortran','libgfortran5','libgcc')]}))
else:
 dist=md.distribution(distribution_name)
 files=[pathlib.Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).endswith(('.so','.pyd'))]
 print(json.dumps({'import_banner':banner.getvalue(),'module_path':str(p),'python':sys.version,'numpy':md.version('numpy'),'distribution_name':dist.metadata['Name'],'distribution_version':dist.version,'wheel_metadata':dist.read_text('WHEEL'),'installer':dist.read_text('INSTALLER'),'binaries':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ldd':subprocess.run(['ldd',str(f)],capture_output=True,text=True).stdout} for f in files],'direct_url':[(d.metadata['Name'],d.read_text('direct_url.json')) for d in md.distributions() if d.read_text('direct_url.json')]}))'''
                probe_env = env.copy()
                probe_env.pop('PYTHONPATH', None)
                distribution = {'ducc':'ducc0','shtns':'shtns','rustsht':'rustsht','pyspharm':'pyspharm'}[backend]
                probe_output = subprocess.check_output([str(python), '-I', '-c', probe, module, distribution], env=probe_env, text=True)
                observed = next(json.loads(line) for line in probe_output.splitlines() if line.startswith('{'))
                observed['native_import_output'] = '\n'.join(line for line in probe_output.splitlines() if not line.startswith('{'))
                if not Path(observed['module_path']).is_relative_to(venv):
                    raise ValueError('import escaped isolated environment')
                if backend == 'pyspharm':
                    verify_import_path(venv, observed['module_path'])
                    observed['runtime_probe'] = runtime_probe(venv)
                    if observed['runtime_probe']['native_module_path'] not in {
                        binary['path'] for binary in observed['binaries']
                    }:
                        raise ValueError('runtime _spherepack extension differs from the inspected binary')
                if not observed['binaries']:
                    raise ValueError('no native binary found')
                build.update(observed)
                build['python_executable'] = str(python)
                build['status'] = 'built'
                build['freeze'] = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)], text=True)
                if backend == 'pyspharm':
                    exports = installed_environment_exports(conda, venv)
                    explicit_path = args.output / 'pyspharm-conda-explicit.txt'
                    environment_path = args.output / 'pyspharm-conda-environment.yml'
                    explicit_path.write_text(exports['explicit'])
                    environment_path.write_text(exports['environment'])
                    build['conda_exports'] = {
                        'explicit_file': str(explicit_path.resolve()),
                        'explicit_sha256': hashlib.sha256(explicit_path.read_bytes()).hexdigest(),
                        'environment_file': str(environment_path.resolve()),
                    }
                else:
                    build['tracked_source_changes_after_build'] = git(path, 'diff', '--stat', 'HEAD')
                    build['submodules'] = git(path, 'submodule', 'status', '--recursive')
                    build['compiler_metadata_files'] = collect_build_files(path,backend,args.output)
                    if build['tracked_source_changes_after_build']:
                        raise ValueError('build modified tracked upstream sources')
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                build['status'] = 'build_failed'
                build['error'] = str(exc)
                if backend == 'pyspharm':
                    build['status'] = 'unavailable'
        (args.output / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return 0 if all(b['status'] == 'built' for b in manifest['builds'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
