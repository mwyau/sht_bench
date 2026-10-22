"""Observable memory limits and stored Spherepack plan-size lower bounds."""
from __future__ import annotations

from pathlib import Path


def available_memory():
    """Read host available memory and finite remaining cgroup limits in bytes."""
    result = {'host_available_bytes':None, 'cgroup_limits':[]}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                result['host_available_bytes'] = int(line.split()[1])*1024
    except (OSError,ValueError):
        pass
    roots = [Path('/sys/fs/cgroup')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                path = Path('/sys/fs/cgroup') / line.split('::',1)[1].lstrip('/')
                if path.exists():
                    while path.is_relative_to('/sys/fs/cgroup'):
                        roots.append(path)
                        if path == Path('/sys/fs/cgroup'):break
                        path = path.parent
    except OSError:
        pass
    for root in dict.fromkeys(roots):
        try:
            limit = (root/'memory.max').read_text().strip()
            if limit == 'max':continue
            current = int((root/'memory.current').read_text().strip())
            result['cgroup_limits'].append({'path':str(root), 'limit_bytes':int(limit), 'current_bytes':current,
                                             'remaining_bytes':max(0,int(limit)-current)})
        except (OSError,ValueError):
            continue
    values = [result['host_available_bytes']] if result['host_available_bytes'] is not None else []
    values += [entry['remaining_bytes'] for entry in result['cgroup_limits']]
    result['available_bytes'] = min(values) if values else None
    return result


def spherepack_stored_arrays(nlat, nlon, grid):
    """Sizes of all four persistent float32 arrays in pyspharm-syl 1.2.1 Spharmt.

    The public constructor initializes scalar and vector tables for every plan.
    These are lower bounds: working arrays, fields and coefficient buffers are
    additional allocations. Integer formulas reproduce the wrapper's positive
    array-length expressions without floating-point overflow.
    """
    if nlat < 3 or nlon < 4 or grid not in ('GL','CC'):
        raise ValueError('Spherepack stored plan requires GL/CC, nlat>=3 and nlon>=4')
    n1 = min(nlat,(nlon+1)//2 if nlon%2 else (nlon+2)//2)
    n2 = (nlat+1)//2
    if grid == 'CC':
        scalar = (n1*n2*(2*nlat-n1+1))//2+nlon+15
        vector = n1*n2*(2*nlat-n1+1)+nlon+15
        lengths = {'wshaes':scalar,'wshses':scalar,'wvhaes':vector,'wvhses':vector}
    else:
        scalar = nlat*(3*(n1+n2)-2)+((n1-1)*(n2*(2*nlat-n1)-3*n1))//2+nlon+15
        analysis = ((nlat+1)*(nlat+1)*nlat)//2+nlon+15
        synthesis = n1*n2*(2*nlat-n1+1)+nlon+15+2*nlat
        lengths = {'wshags':scalar,'wshsgs':scalar,'wvhags':analysis,'wvhsgs':synthesis}
    return {'array_lengths':lengths,'itemsize_bytes':4,'persistent_lower_bound_bytes':sum(lengths.values())*4,
            'method':'pyspharm-syl 1.2.1 spharm.py Spharmt legfunc=stored array-length formulas'}


def preflight(backend, workload, *, limit_bytes=None):
    observation = available_memory()
    available = observation['available_bytes']
    budget = min(available,limit_bytes) if available is not None and limit_bytes is not None else (available if limit_bytes is None else limit_bytes)
    result = {'memory_observation':observation,'memory_budget_bytes':budget,'persistent_lower_bound_bytes':None,
              'resource_limited':False}
    if backend == 'pyspharm' and workload.grid in ('GL','CC') and workload.nlat>=3 and workload.nlon>=4:
        result.update(spherepack_stored_arrays(workload.nlat,workload.nlon,workload.grid))
        if budget is not None and result['persistent_lower_bound_bytes']>budget:
            result['resource_limited'] = True
            result['reason'] = 'native stored-plan persistent arrays alone exceed observed available memory or explicit memory budget'
    return result


def apply_address_space_limit(additional_bytes):
    """Bound additional worker virtual allocations after existing imports.

    The baseline virtual size includes loaded libraries. This is a process
    allocation limit, not an RSS measurement or an estimate of native plan size.
    """
    if additional_bytes is None:return None
    import resource
    try:
        baseline = int(Path('/proc/self/statm').read_text().split()[0])*__import__('os').sysconf('SC_PAGE_SIZE')
        soft,hard = resource.getrlimit(resource.RLIMIT_AS)
        target = baseline+additional_bytes
        if hard != resource.RLIM_INFINITY:target=min(target,hard)
        if soft != resource.RLIM_INFINITY:target=min(target,soft)
        resource.setrlimit(resource.RLIMIT_AS,(target,hard))
        return {'baseline_virtual_bytes':baseline,'soft_limit_bytes':target,'scope':'worker virtual address space, including loaded libraries'}
    except (OSError,ValueError,AttributeError) as exc:
        return {'error':str(exc)}
