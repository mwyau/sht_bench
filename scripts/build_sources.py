"""Build isolated Git backends and a wheel-only Spherepack environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from sht_bench.sources import checkout, git

MODULES = {'ducc': 'ducc0', 'shtns': 'shtns', 'rustsht': 'rustsht', 'pyspharm': 'spharm'}


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
    for name in ('ducc', 'shtns', 'rustsht'):
        parser.add_argument(f'--{name}-ref')
    args = parser.parse_args(argv)
    if (args.output / 'build-manifest.json').exists():
        parser.error('build manifest already exists; choose a new output directory')
    if args.mode == 'locked' and args.lock is None:
        parser.error('--mode locked requires --lock from the previous run')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.lock or args.output / 'sources.lock.json'
    sources = checkout(args.root.resolve(), lock, args.mode,
                       {n: getattr(args, n + '_ref') for n in ('ducc', 'shtns', 'rustsht') if getattr(args, n + '_ref')})
    env = os.environ.copy()
    for key, default in [('CC', 'gcc'), ('CXX', 'g++'), ('FC', 'gfortran')]:
        env.setdefault(key, shutil.which(default) or default)
    env.setdefault('RUSTFLAGS', '-C target-cpu=native')
    env.setdefault('DUCC0_OPTIMIZATION', 'native')
    manifest = {'schema': 1, 'sources': sources, 'builds': {}, 'compiler_environment': {
        k: env.get(k) for k in ('CC', 'CXX', 'FC', 'CFLAGS', 'CXXFLAGS', 'FFLAGS', 'LDFLAGS', 'RUSTFLAGS', 'DUCC0_OPTIMIZATION')},
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
    for backend, module in MODULES.items():
        build = {'status': 'build_failed'}
        manifest['builds'][backend] = build
        venv = (args.output / 'envs' / backend).resolve()
        python = venv / 'bin/python'
        with (args.output / f'build-{backend}.log').open('w') as log:
            try:
                run(['uv', 'venv', '--python', args.python, venv], env=env, log=log)
                run(['uv', 'pip', 'install', '--python', python, '-e', '.[plot]', 'pytest', 'scipy'], env=env, log=log)
                if backend == 'pyspharm':
                    build['build_kind'] = 'published-wheel'
                    build['requirement'] = 'pyspharm-syl==1.2.1'
                    run(['uv', 'pip', 'install', '--python', python, '--only-binary', 'pyspharm-syl', 'pyspharm-syl==1.2.1'], env=env, log=log)
                else:
                    path = args.root.resolve() / backend
                    if git(path, 'rev-parse', 'HEAD') != sources['sources'][backend]['sha']:
                        raise ValueError('source changed after acquisition')
                    build.update(sources['sources'][backend])
                    build['build_kind'] = 'git-source'
                    run(['uv', 'pip', 'install', '--python', python, '--no-cache', str(path)], env=env, log=log)
                # Run outside the checkout with no PYTHONPATH. Hash imported extensions and inspect linkage.
                probe = '''import importlib,json,hashlib,pathlib,sys,subprocess,importlib.metadata as md,io,contextlib
banner=io.StringIO()
with contextlib.redirect_stdout(banner):
 m=importlib.import_module(sys.argv[1])
p=pathlib.Path(m.__file__).resolve()
dist=md.distribution(sys.argv[2])
files=[pathlib.Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).endswith(('.so','.pyd'))]
print(json.dumps({'import_banner':banner.getvalue(),'module_path':str(p),'python':sys.version,'numpy':md.version('numpy'),'binaries':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ldd':subprocess.run(['ldd',str(f)],capture_output=True,text=True).stdout} for f in files],'direct_url':[(d.metadata['Name'],d.read_text('direct_url.json')) for d in md.distributions() if d.read_text('direct_url.json')]}))'''
                probe_env = env.copy()
                probe_env.pop('PYTHONPATH', None)
                distribution = {'ducc':'ducc0','shtns':'shtns','rustsht':'rustsht','pyspharm':'pyspharm-syl'}[backend]
                probe_output = subprocess.check_output([str(python), '-I', '-c', probe, module, distribution], env=probe_env, text=True)
                observed = next(json.loads(line) for line in probe_output.splitlines() if line.startswith('{'))
                observed['native_import_output'] = '\n'.join(line for line in probe_output.splitlines() if not line.startswith('{'))
                if not Path(observed['module_path']).is_relative_to(venv):
                    raise ValueError('import escaped isolated environment')
                if not observed['binaries']:
                    raise ValueError('no native binary found')
                build.update(observed)
                build['python_executable'] = str(python)
                build['status'] = 'built'
                build['freeze'] = subprocess.check_output(['uv', 'pip', 'freeze', '--python', str(python)], text=True)
                if backend != 'pyspharm':
                    build['tracked_source_changes_after_build'] = git(path, 'diff', '--stat', 'HEAD')
                    build['submodules'] = git(path, 'submodule', 'status', '--recursive')
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
