"""Process-isolated, numerically gated source benchmark matrix."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from .source_adapters import Workload, Adapter, Unsupported, capability, fixtures, PLANNING_POLICIES, MODULES
from .source_validation import validate,validate_operation,reference_map

DEGREES = [1,4,16,31,32,42,63,64,65,85,127,128,159,255,256,319,511,639,720,1023,1279]


def physical_cores():
    """Choose one available logical CPU per physical core, grouped by L3 domain."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    chosen = {}
    domains = {}
    for cpu in allowed:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        try:
            key = ((base/'topology/physical_package_id').read_text().strip(),(base/'topology/core_id').read_text().strip())
            domain = (base/'cache/index3/shared_cpu_list').read_text().strip()
        except OSError:
            key,domain = (cpu,), 'unknown'
        chosen.setdefault(key,cpu)
        domains[cpu] = domain
    return sorted(chosen.values(),key=lambda c:(domains[c],c)),domains


def workloads(degrees, smoke=False):
    from .grids import ATMOSPHERIC_GL_GRIDS,DEFAULT_CC_LMAX
    geometries = set()
    for l in degrees:
        for grid in ('GL','CC','F1'):
            geometries.add((grid,l+(2 if grid=='CC' else 1),2*l+1,l,'minimum'))
        for regular in ('CC','F1'):
            geometries.add((regular,max(32,2*l+2),2*l+1,l,'quadrature'))
        geometries.add(('GL',l+2,2*l+1,l,'extended'))
        if l<31:
            geometries.add(('GL',32,2*l+1,l,'simd'))
        if l in ATMOSPHERIC_GL_GRIDS:
            nlat,nlon,_ = ATMOSPHERIC_GL_GRIDS[l]
            geometries.add(('GL',nlat,nlon,l,'atmospheric'))
        if l in DEFAULT_CC_LMAX:
            geometries.add(('CC',2*l+1,4*l,l,'regular'))
    if not smoke:
        for l in DEFAULT_CC_LMAX:
            geometries.add(('CC',2*l+1,4*l,l,'regular'))
    for grid,nlat,nlon,l,group in sorted(geometries):
        for m in sorted({l,max(1,l//4)}):
            for spin in (0,1):
                for precision in ('float64','float32'):
                    yield Workload(grid,nlat,nlon,l,m,spin=spin,precision=precision,geometry_group=group)


def identity(backend,w,q,round_index,sha):
    data = dict(backend=backend,workload=w.record(),threads_requested=q,round=round_index,source_sha=sha)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _json_safe(value):
    """Replace nonfinite metric values before writing auditable JSON."""
    if isinstance(value,float):
        return value if math.isfinite(value) else None
    if isinstance(value,dict):
        return {key:_json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_safe(item) for item in value]
    return value


def harness_hash():
    from bench_common import timing,environment
    paths = [Path(__file__),*[Path(__file__).with_name(name) for name in
             ('source_adapters.py','source_validation.py','source_batch.py','source_resources.py','grids.py')],
             Path(timing.__file__),Path(environment.__file__)]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def resident_bytes():
    try:
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,IndexError):
        return None


def validation_identity(task):
    """Identity for a pre-timing gate shared by analysis/synthesis cells."""
    w = task['workload']
    workload = {key:w[key] for key in ('grid','nlat','nlon','lmax','mmax','spin','precision','batch','phi0')}
    return {
        'backend':task['backend'],
        'source_sha':task['source_sha'],
        'build_manifest_sha256':task['build_manifest_sha256'],
        'harness_sha256':task['harness_sha256'],
        'threads_requested':task['threads_requested'],
        'workload':workload,
    }


def load_validation_certificate(task):
    """Reject timing unless an exact backend/workload/thread gate passed first."""
    path = Path(task['validation_certificate'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != task.get('validation_certificate_sha256'):
        raise ValueError('validation certificate checksum changed')
    certificate = json.loads(raw)
    if certificate.get('validation_identity') != validation_identity(task):
        raise ValueError('validation certificate belongs to another backend/build/workload/thread')
    return certificate


def worker(args):
    import numpy as np
    from bench_common.timing import measure
    from bench_common.environment import collect_environment
    task = json.loads(args.task.read_text())
    w = Workload(**task['workload'])
    result = {**task,'status':'build_failed'}
    try:
        if task.get('harness_sha256') != harness_hash():
            raise ValueError('benchmark implementation changed after matrix launch')
        cores,domains = physical_cores()
        q = task['threads_requested']
        if q > len(cores):
            raise Unsupported('requested threads exceed available physical cores')
        if hasattr(os,'sched_setaffinity'):
            os.sched_setaffinity(0,cores[:q])
        result['environment'] = collect_environment(include_cpu_controls=True)
        result['environment']['l3_domains'] = {str(c):domains[c] for c in cores[:q]}
        result['environment']['load_average'] = list(os.getloadavg()) if hasattr(os,'getloadavg') else None
        result['environment']['measurement_label'] = 'provisional: shared host; power/frequency not controlled'
        import importlib
        t = time.perf_counter()
        importlib.import_module(MODULES[task['backend']])
        result['module_import_s'] = time.perf_counter()-t
        from .source_resources import apply_address_space_limit
        result['allocation_limit'] = apply_address_space_limit(task.get('resource_preflight',{}).get('memory_budget_bytes'))
        result['rss_before_setup_bytes'] = resident_bytes()
        t = time.perf_counter()
        adapter = Adapter(task['backend'],w,q)
        if Path(adapter.module.__file__).resolve() != Path(task['expected_import_path']).resolve():
            raise ValueError('import path differs from verified build manifest')
        result['setup_s'] = time.perf_counter()-t
        result['rss_after_setup_bytes'] = resident_bytes()
        result['persistent_state_note'] = 'resident process delta after constructor/grid setup; native plan allocation is not separately queryable'
        result['setup_policy'] = adapter.setup_policy
        result['setup_components_s'] = adapter.setup_components_s
        result['native_setup_s'] = adapter.native_setup_s
        result['native_api'] = adapter.native_api
        result['native_conventions'] = adapter.native_conventions
        if w.batch>1 and task['backend']=='ducc':
            result['batch_validation_policy'] = 'native batched synthesis; independent per-field analysis_2d calls only for validation, outside timing'
        result['threads_effective'] = adapter.effective_threads
        result['thread_control'] = adapter.thread_control
        result['import_path'] = adapter.module.__file__
        if task.get('validation_only'):
            result['validation_identity'] = validation_identity(task)
            result['validation'] = validate(adapter)
            result['status'] = ('validation_passed' if result['validation']['status'] == 'supported_and_validated'
                                else 'numerical_validation_failed')
            result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
            result['memory_policy'] = 'process peak RSS including setup and numerical validation'
            write_json(args.result,_json_safe(result))
            return 0 if result['status'] == 'validation_passed' else 1

        certificate = load_validation_certificate(task) if task.get('validation_certificate') else None
        if certificate is not None:
            result['validation'] = certificate.get('validation')
            result['validation_certificate_sha256'] = task['validation_certificate_sha256']
        if certificate is not None and certificate.get('status') != 'validation_passed':
            result.update(status='numerical_validation_failed', reason='matching pre-timing validation gate failed')
        else:
            a = next(fixtures(w))[1]
            native = adapter.encode(a)
            first_synthesis = adapter.operation(native,None,operation='synthesis')
            t = time.perf_counter()
            first_map = first_synthesis()
            result['first_synthesis_s'] = time.perf_counter()-t
            m = adapter.canonical_synthesis_output(first_map)
            first_analysis = adapter.operation(native,m,operation='analysis')
            t = time.perf_counter()
            first_analysis()
            result['first_analysis_s'] = time.perf_counter()-t
            result['first_call_policy'] = 'native allocating A/S calls; argument preparation and canonical output conversion outside timing'
            if w.batch>1 and task['backend']=='ducc':
                result['first_call_policy'] += '; first_analysis_s is a validation-only loop of individual analysis_2d calls, not native batch analysis'
            del first_map,first_synthesis,first_analysis
            if certificate is None:
                result['validation'] = validate(adapter)
            if result['validation']['status'] != 'supported_and_validated':
                result['status'] = 'numerical_validation_failed'
            else:
                m = np.ascontiguousarray(reference_map(w,a),dtype=w.precision)
                output = None
                if w.allocation == 'preallocated':
                    output = np.empty_like(native if w.operation=='analysis' else m)
                t = time.perf_counter()
                fn = adapter.operation(native,m,output)
                result['operation_prepare_s'] = time.perf_counter()-t
                result['operation_prepare_policy'] = 'native input/output preparation; coefficient conversion and fixture generation are outside transform timing'
                if w.operation not in ('analysis','synthesis'):
                    t = time.perf_counter()
                    fn()
                    result['first_requested_operation_s'] = time.perf_counter()-t
                # Secondary operation gates are checked separately before timing.
                if w.operation not in ('analysis','synthesis'):
                    result['operation_validation'] = validate_operation(adapter,a,m,fn)
                if not result.get('operation_validation',{'passed':True})['passed']:
                    result['status'] = 'numerical_validation_failed'
                else:
                    if output is not None:
                        from .source_validation import errors,acceptable
                        expected = adapter.analysis(m) if w.operation=='analysis' else adapter.synthesis(native)
                        observed = fn()
                        result['preallocated_validation'] = errors(observed,expected)
                        if not acceptable(result['preallocated_validation'],result['validation']['tolerance']):
                            result['status'] = 'numerical_validation_failed'
                            write_json(args.result,result)
                            return 1
                    iterations,samples = measure(fn,warmup=2,repeat=task['repeat'],min_time=task['min_time'])
                    result.update(status='timed_successfully',samples_s=samples,iterations=iterations,
                                  median_s=float(np.median(samples)),p10_s=float(np.percentile(samples,10)),
                                  p90_s=float(np.percentile(samples,90)),std_s=float(np.std(samples)),
                                  transforms_per_second=w.batch/float(np.median(samples)),
                                  grid_points_per_second=w.batch*w.nlat*w.nlon/float(np.median(samples)))
        result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        result['memory_policy'] = 'process peak RSS including setup, validation and transform buffers'
    except Unsupported as exc:
        result.update(status='unsupported',reason=str(exc))
    except MemoryError as exc:
        result.update(status='resource_limited',reason=str(exc) or 'worker allocation failed (MemoryError); see recorded memory observation and address-space limit')
    except (Exception,) as exc:
        import traceback
        result.update(status='execution_failed',reason=str(exc),traceback=traceback.format_exc())
    write_json(args.result,_json_safe(result))
    return 0 if result['status'] in ('unsupported','timed_successfully','validation_passed') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-manifest',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--degrees',default=','.join(map(str,DEGREES)))
    parser.add_argument('--threads',default='1,2,4,8,16')
    parser.add_argument('--backends',default='ducc,shtns,rustsht,pyspharm')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--repeat',type=int,default=30)
    parser.add_argument('--min-time',type=float,default=.02)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--memory-limit-gib',type=float,help='cap additional worker virtual allocations; automatic observed available memory is also recorded and bounds this cap')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--secondary',action='store_true')
    parser.add_argument('--task',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--result',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.task: return worker(args)
    if not args.output or not args.build_manifest: parser.error('--output and --build-manifest required')
    if args.rounds<1 or args.repeat<1 or not math.isfinite(args.min_time) or args.min_time<=0 or not math.isfinite(args.timeout) or args.timeout<=0:
        parser.error('positive rounds/repeat and finite positive min-time/timeout required')
    if args.memory_limit_gib is not None and (not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib<=0):
        parser.error('--memory-limit-gib must be finite and positive')
    if args.memory_limit_gib is not None and not 1/2**30 <= args.memory_limit_gib <= sys.maxsize/2**30:
        parser.error('--memory-limit-gib must represent between one byte and the platform address-space bound')
    memory_limit_bytes = None if args.memory_limit_gib is None else int(args.memory_limit_gib*2**30)
    try:
        requested_degrees = [int(value) for value in args.degrees.split(',')]
        requested_threads = [int(value) for value in args.threads.split(',')]
    except ValueError:
        parser.error('degrees and threads must be comma-separated integers')
    if any(value<1 for value in requested_degrees+requested_threads) or len(set(requested_degrees))!=len(requested_degrees) or len(set(requested_threads))!=len(requested_threads):
        parser.error('degrees and threads must contain unique positive integers')
    requested_backends = args.backends.split(',')
    if len(set(requested_backends))!=len(requested_backends) or set(requested_backends)-set(PLANNING_POLICIES):
        parser.error('backends must contain unique supported names')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error('output exists; choose a new directory or --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    cells = args.output/'cells'
    cells.mkdir(exist_ok=True)
    build = json.loads(args.build_manifest.read_text())
    build_hash = hashlib.sha256(args.build_manifest.read_bytes()).hexdigest()
    code_hash = harness_hash()
    run_config = {'build_manifest_sha256':build_hash,'harness_sha256':code_hash,
                  'degrees':args.degrees,'threads':args.threads,'backends':args.backends,
                  'rounds':args.rounds,'repeat':args.repeat,'min_time':args.min_time,
                  'timeout':args.timeout,'smoke':args.smoke,'secondary':args.secondary,'memory_limit_bytes':memory_limit_bytes}
    config_path = args.output/'run-config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError('incompatible resume configuration or changed build/harness')
    write_json(config_path,run_config)
    # A repository-wide advisory lock serializes benchmark controllers across outputs.
    import fcntl
    lock_path = Path('.builds/benchmark.lock')
    lock_path.parent.mkdir(exist_ok=True)
    lock_handle = lock_path.open('w')
    try:
        fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('another source benchmark controller is running')
    for backend,b in build['builds'].items():
        if b['status']=='built':
            for binary in b['binaries']:
                if hashlib.sha256(Path(binary['path']).read_bytes()).hexdigest()!=binary['sha256']:
                    raise ValueError(f'{backend}: installed binary changed since build')
    cores,_ = physical_cores()
    qs = [int(q) for q in args.threads.split(',')]
    backends = args.backends.split(',')
    degrees = [int(l) for l in args.degrees.split(',')]
    configs = list(workloads(degrees,args.smoke))
    if args.secondary:
        configs += [replace(w,operation=op) for w in list(configs) for op in ('adjoint_analysis','adjoint_synthesis','gradient','deriv1') if w.spin==0 or op.startswith('adjoint')]
        configs += [replace(w,batch=4) for w in list(configs) if w.operation in ('analysis','synthesis')]
        configs += [replace(w,phi0=origin,nlon=w.nlon+delta) for w in list(configs)
                    if w.batch==1 and w.operation=='synthesis'
                    for delta,origin in ((1,0.0),(0,.37),(1,.37))]
    write_json(args.output/'workload-plan.json',{'workloads':[w.record() for w in configs],
               'backends':backends,'threads':qs,'rounds':args.rounds,
               'operation_expansion':'synthesis entries expand to analysis and synthesis; every entry expands to allocating and preallocated'})
    ledger = []
    # Alternate backend order between independent process rounds.
    for round_index in range(args.rounds):
        order = backends if round_index%2==0 else list(reversed(backends))
        for wbase in configs:
            ops = ('analysis','synthesis') if wbase.operation=='synthesis' else (wbase.operation,)
            for op in ops:
                for allocation in ('allocating','preallocated'):
                    w = replace(wbase,operation=op,allocation=allocation)
                    for q in qs:
                        for backend in order:
                            b = build['builds'].get(backend,{'status':'unavailable'})
                            sha = b.get('sha',b.get('requirement','unavailable'))
                            key = identity(backend,w,q,round_index,sha+build_hash+code_hash)
                            path = cells/f'{key}.json'
                            task = dict(id=key,backend=backend,workload=w.record(),threads_requested=q,round=round_index,
                                        source_sha=sha,build_mode=b.get('build_kind'),build_manifest=str(args.build_manifest.resolve()),
                                        build_manifest_sha256=build_hash,harness_sha256=code_hash,expected_import_path=b.get('module_path'),
                                        repeat=args.repeat,min_time=args.min_time,coefficient_layout='canonical m-major packed; orthonormal; CS phase; real-field positive-order factor two',
                                        planning_policy=PLANNING_POLICIES[backend],allocation_policy=allocation)
                            ledger.append({'id':key,'file':str(path.relative_to(args.output))})
                            if args.resume and path.exists():
                                previous=json.loads(path.read_text())
                                if any(previous.get(k)!=v for k,v in task.items()):
                                    raise ValueError(f'incompatible resume: {path}')
                                continue
                            reason = capability(backend,w)
                            if reason:
                                write_json(path,{**task,'status':'unsupported','reason':reason})
                                continue
                            if q>len(cores):
                                write_json(path,{**task,'status':'resource_limited','reason':'exceeds available physical core count'})
                                continue
                            if b['status'] != 'built':
                                write_json(path,{**task,'status':b['status'],'reason':b.get('error','backend unavailable')})
                                continue
                            from .source_resources import preflight
                            admission = preflight(backend,w,limit_bytes=memory_limit_bytes)
                            task['resource_preflight'] = admission
                            if admission['resource_limited']:
                                write_json(path,{**task,'status':'resource_limited','reason':admission['reason']})
                                continue
                            taskfile = args.output/'task.json'
                            write_json(taskfile,task)
                            env = os.environ.copy()
                            env['SHT_BENCH_REFERENCE_CACHE'] = str((Path('.builds/reference')/code_hash).resolve())
                            for variable in ('OMP_NUM_THREADS','RAYON_NUM_THREADS','DUCC0_NUM_THREADS'):
                                env[variable]=str(q)
                            for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                                env[variable]='1'
                            with (cells/f'{key}.log').open('w') as log:
                                try:
                                    p = subprocess.run([b['python_executable'],'-m','sht_bench.source_runner','--task',str(taskfile),'--result',str(path)],env=env,stdout=log,stderr=log,timeout=args.timeout)
                                    if not path.exists():
                                        write_json(path,{**task,'status':'execution_failed','reason':f'worker exited {p.returncode}; see log'})
                                except subprocess.TimeoutExpired:
                                    write_json(path,{**task,'status':'resource_limited','reason':f'worker exceeded {args.timeout}s'})
                            print(f'{key} {backend} {w.grid} L{w.lmax} m{w.mmax} s{w.spin} q{q} {op} '+json.loads(path.read_text())['status'],flush=True)
        write_json(args.output/'ledger.json',{'schema':1,'cells':ledger,'configuration':vars(args)|{'output':str(args.output),'build_manifest':str(args.build_manifest),'task':None,'result':None}})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
