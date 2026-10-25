"""Check source-matrix coverage and the evidence behind each valid timing."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics

from .source_adapters import Workload

STATUSES = {
    'timed_successfully', 'unsupported', 'unavailable', 'build_failed',
    'numerical_validation_failed', 'resource_limited', 'execution_failed',
}


def case_key(backend, workload, threads, process_round):
    return (backend, json.dumps(workload, sort_keys=True), threads, process_round)


def expected_cases(plan):
    """Expand explicit plan entries into backend/operation/allocation/process cells."""
    seen = set()
    allocations = plan.get('allocations', ('allocating', 'preallocated'))
    for fields in plan['workloads']:
        base = Workload(**fields)
        operations = ('analysis', 'synthesis') if base.operation == 'synthesis' else (base.operation,)
        for operation in operations:
            for allocation in allocations:
                w = replace(base, operation=operation, allocation=allocation)
                for backend in plan['backends']:
                    for threads in plan['threads']:
                        for process_round in range(plan['rounds']):
                            key = case_key(backend, w.record(), threads, process_round)
                            if key in seen:
                                raise ValueError('workload plan contains duplicate expanded cells')
                            seen.add(key)
                            yield key


def _config_values(config, key, default, *, converter=str):
    value = config.get(key, default)
    if isinstance(value, str):
        values = [part.strip() for part in value.split(',') if part.strip()]
    else:
        values = list(value)
    return [converter(part) for part in values]


def required_base_workloads(config):
    """Independently reconstruct the requested subset of standard geometries."""
    from .grids import ATMOSPHERIC_GL_GRIDS, DEFAULT_CC_LMAX
    grids = set(_config_values(config, 'grids', 'GL,CC,F1'))
    precisions = set(_config_values(config, 'precisions', 'float64,float32'))
    spins = set(_config_values(config, 'spins', '0,1', converter=int))
    orders = set(_config_values(config, 'orders', 'full,quarter'))
    geometry_groups = (None if config.get('geometry_groups') is None else
                       set(_config_values(config, 'geometry_groups', '')))
    geometries = set()
    degrees = _config_values(config, 'degrees', '', converter=int)
    for degree in degrees:
        for grid in ('GL','CC','F1'):
            geometries.add((grid,degree+(2 if grid=='CC' else 1),2*degree+1,degree,'minimum'))
        for grid in ('CC','F1'):
            geometries.add((grid,max(32,2*degree+2),2*degree+1,degree,'quadrature'))
        geometries.add(('GL',degree+2,2*degree+1,degree,'extended'))
        if degree < 31:
            geometries.add(('GL',32,2*degree+1,degree,'simd'))
        if degree in ATMOSPHERIC_GL_GRIDS:
            nlat,nlon,_ = ATMOSPHERIC_GL_GRIDS[degree]
            geometries.add(('GL',nlat,nlon,degree,'atmospheric'))
        if degree in DEFAULT_CC_LMAX:
            geometries.add(('CC',2*degree+1,4*degree,degree,'regular'))
    if not config['smoke']:
        for degree in DEFAULT_CC_LMAX:
            geometries.add(('CC',2*degree+1,4*degree,degree,'regular'))
    base = []
    for grid,nlat,nlon,degree,group in geometries:
        if grid not in grids or (geometry_groups is not None and group not in geometry_groups):
            continue
        orders_for_degree = set()
        if 'full' in orders:
            orders_for_degree.add(degree)
        if 'quarter' in orders:
            quarter = max(1,degree//4)
            if quarter < degree:
                orders_for_degree.add(quarter)
        for order in orders_for_degree:
            for spin in spins:
                for precision in precisions:
                    base.append(Workload(grid,nlat,nlon,degree,order,spin=spin,precision=precision,geometry_group=group).record())
    if config.get('batches') is not None:
        requested_batches = sorted({int(value) for value in config['batches'].split(',')})
        base = [replace(Workload(**fields),batch=batch).record()
                for fields in base for batch in requested_batches]
    yield from base


def required_plan_workloads(config):
    """Require the named experiment dimensions independently of its stored plan."""
    base = list(required_base_workloads(config))
    yield from base
    if not config.get('secondary', False):
        return
    for fields in base:
        w = Workload(**fields)
        for operation in ('adjoint_analysis','adjoint_synthesis','gradient','deriv1'):
            if w.spin == 0 or operation.startswith('adjoint'):
                yield replace(w,operation=operation).record()
        yield replace(w,batch=4).record()
        for nlon,phi0 in ((w.nlon+1,0.0),(w.nlon,0.37),(w.nlon+1,0.37)):
            yield replace(w,nlon=nlon,phi0=phi0).record()


def finite_number(value, *, positive=False):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def check_metrics(metrics, tolerance):
    if not isinstance(metrics, dict):
        return False
    return (all(finite_number(metrics.get(k)) for k in ('relative_l2', 'max_absolute', 'scaled_error'))
            and metrics['relative_l2'] <= tolerance and metrics['scaled_error'] <= tolerance)


def required_fixtures(w):
    names = {'random-triangular'}
    modes = {(max(1, w.spin), 0), (1, min(1, w.mmax)), (w.lmax, 0), (w.lmax, w.mmax)}
    for degree, order in modes:
        for component in range(w.components):
            for part in ('real', 'imag') if order else ('real',):
                names.add(f'mode-{degree}-{order}-{component}-{part}')
    if not w.spin:
        names.add('Y00')
    return names


def check_timing(row, config, build):
    """Recompute acceptance and timing summaries; do not trust a status label alone."""
    import numpy as np
    problems = []
    w = Workload(**row['workload'])
    tolerance = 5e-5 if w.precision == 'float32' else 2e-10
    validation = row.get('validation', {})
    fixtures = validation.get('fixtures', [])
    if validation.get('status') != 'supported_and_validated' or validation.get('tolerance') != tolerance:
        problems.append('invalid numerical gate status/tolerance')
    if {f.get('name') for f in fixtures} != required_fixtures(w):
        problems.append('missing or unexpected validation fixtures')
    if len(fixtures) != len({f.get('name') for f in fixtures}):
        problems.append('duplicate validation fixtures')
    for fixture in fixtures:
        if fixture.get('passed') is not True or fixture.get('dtype_passed') is not True:
            problems.append('fixture or precision gate failed')
        stages = ['synthesis', 'coefficient_recovery', 'map_roundtrip']
        if fixture.get('name') == 'random-triangular':
            stages += ['full_grid_synthesis', 'common_reference_analysis']
        if any(not check_metrics(fixture.get(stage), tolerance) for stage in stages):
            problems.append(f"invalid numerical metrics: {fixture.get('name')}")
    if w.allocation == 'preallocated' and not check_metrics(row.get('preallocated_validation'), tolerance):
        problems.append('invalid preallocated-output gate')
    if w.operation not in ('analysis', 'synthesis'):
        operation = row.get('operation_validation', {})
        if operation.get('passed') is not True or operation.get('tolerance') != tolerance:
            problems.append('missing secondary operation gate')
        if w.operation in ('gradient', 'deriv1'):
            if not check_metrics(operation.get('independent_synthesis'), tolerance):
                problems.append('invalid independent derivative synthesis metrics')
        else:
            lhs, rhs = operation.get('adjoint_lhs'), operation.get('adjoint_rhs')
            if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (lhs, rhs)):
                problems.append('invalid adjoint inner products')
            else:
                error = abs(lhs-rhs)/max(abs(lhs), abs(rhs), np.finfo(float).tiny)
                if error > tolerance or not math.isclose(error, operation.get('scaled_residual', -1), rel_tol=1e-10, abs_tol=1e-15):
                    problems.append('adjoint identity does not pass its recorded tolerance')
    samples = row.get('samples_s', [])
    if len(samples) != config['repeat'] or not samples or not all(finite_number(s, positive=True) for s in samples):
        problems.append('missing or invalid raw timing samples')
    else:
        summaries = {'median_s': statistics.median(samples), 'p10_s': float(np.percentile(samples, 10)),
                     'p90_s': float(np.percentile(samples, 90)), 'std_s': float(np.std(samples))}
        for name, value in summaries.items():
            if not finite_number(row.get(name)) or not math.isclose(row[name], value, rel_tol=1e-12, abs_tol=1e-15):
                problems.append(f'incorrect timing summary: {name}')
    if not isinstance(row.get('iterations'), int) or row['iterations'] < 1:
        problems.append('missing calibrated iteration count')
    for name in ('setup_s', 'first_synthesis_s', 'first_analysis_s', 'operation_prepare_s', 'peak_rss_bytes'):
        if not finite_number(row.get(name)):
            problems.append(f'missing separate setup/first-call/memory measurement: {name}')
    if row.get('native_api') is not None:
        from .source_adapters import PLANNING_POLICIES
        if row.get('planning_policy') != PLANNING_POLICIES.get(row['backend']):
            problems.append('native API planning policy differs from adapter definition')
        components = row.get('setup_components_s',{})
        if not isinstance(components,dict) or not all(finite_number(value) for value in components.values()):
            problems.append('invalid separate native setup components')
        else:
            native_sum = sum(value for key,value in components.items() if key.startswith('native_'))
            native_total = row.get('native_setup_s')
            if row['backend']=='ducc':
                if native_total is not None or native_sum:
                    problems.append('DUCC public-call setup cannot be labelled an external reusable plan')
            elif not finite_number(native_total) or not math.isclose(native_total,native_sum,rel_tol=1e-12,abs_tol=1e-15):
                problems.append('native setup total does not match separate constructor/grid components')
            if finite_number(row.get('setup_s')) and sum(components.values())>row['setup_s']+1e-12:
                problems.append('setup components exceed total adapter setup measurement')
    q = row['threads_requested']
    environment = row.get('environment', {})
    affinity = environment.get('process_affinity')
    if not affinity or len(affinity) != q or len(set(affinity)) != q:
        problems.append('recorded affinity does not match physical-core thread request')
    if environment.get('physical_cpus') is not None and q > environment['physical_cpus']:
        problems.append('physical-core experiment exceeds host physical core count')
    if set(environment.get('l3_domains', {})) != {str(cpu) for cpu in affinity or []}:
        problems.append('missing L3-domain affinity')
    backend = build['builds'].get(row['backend'], {})
    if backend.get('status') != 'built' or row.get('import_path') != backend.get('module_path'):
        problems.append('timing does not identify its verified isolated import')
    if row.get('threads_effective') is None and not row.get('thread_control'):
        problems.append('unknown effective threading lacks runtime-control description')
    return problems


def audit(directory, *, build_manifest=None, cell_paths=None):
    config = json.loads((directory/'run-config.json').read_text())
    plan = json.loads((directory/'workload-plan.json').read_text())
    expected = set(expected_cases(plan))
    recorded = 0
    issues = []
    try:
        planned = {json.dumps(w,sort_keys=True) for w in plan['workloads']}
        required = {json.dumps(w,sort_keys=True) for w in required_plan_workloads(config)}
        if required-planned:
            issues.append({'problem':'workload plan omits required geometry or experiment cases', 'missing_required_workloads':len(required-planned)})
        if planned-required:
            issues.append({'problem':'workload plan includes unrequested geometry or experiment cases', 'unexpected_workloads':len(planned-required)})
        expected_allocations = sorted(_config_values(config, 'allocations', 'allocating,preallocated'))
        actual_allocations = sorted(plan.get('allocations', ('allocating', 'preallocated')))
        if (plan['rounds']!=config['rounds']
                or plan['threads']!=_config_values(config, 'threads', '', converter=int)
                or plan['backends']!=_config_values(config, 'backends', '')
                or actual_allocations!=expected_allocations):
            issues.append({'problem':'plan process/backend/thread dimensions differ from run configuration'})
    except (ValueError, TypeError, KeyError) as exc:
        issues.append({'problem':f'cannot establish required base workload scope: {exc}'})
    present = set()
    counts = Counter()
    manifest_path = build_manifest
    manifest_referenced_by_cell = manifest_path is None
    paths = sorted((directory/'cells').glob('*.json')) if cell_paths is None else cell_paths
    if manifest_referenced_by_cell:
        for path in paths:
            try:
                candidate = json.loads(path.read_text()).get('build_manifest')
                if candidate:
                    manifest_path = Path(candidate)
                    break
            except (OSError,ValueError,TypeError,AttributeError):
                continue
    build = None
    if manifest_path is None:
        issues.append({'problem':'no build manifest supplied or referenced by cells'})
    else:
        try:
            manifest_path = Path(manifest_path)
            if manifest_referenced_by_cell and not manifest_path.is_absolute():
                manifest_path = directory/manifest_path
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config['build_manifest_sha256']:
                issues.append({'problem':'referenced build manifest digest changed'})
            build = json.loads(manifest_path.read_text())
            if not isinstance(build.get('builds'),dict):
                raise ValueError('build manifest lacks a backend mapping')
        except (OSError,ValueError,AttributeError) as exc:
            build = None
            issues.append({'problem':f'build manifest cannot be read: {exc}'})
    # Validate one raw cell at a time; full fixture payloads need not stay in RAM.
    for path in paths:
        try:
            row = json.loads(path.read_text())
            key = case_key(row['backend'], row['workload'], row['threads_requested'], row['round'])
            if key in present:
                issues.append({'file':path.name, 'problem':'duplicate workload/process identity'})
            present.add(key)
            if key not in expected:
                issues.append({'file':path.name, 'problem':'cell not requested by workload plan'})
            if row.get('harness_sha256') != config['harness_sha256'] or row.get('build_manifest_sha256') != config['build_manifest_sha256']:
                issues.append({'file':path.name, 'problem':'cell uses a different harness or build manifest'})
            identity = dict(backend=row['backend'],workload=row['workload'],threads_requested=row['threads_requested'],
                            round=row['round'],source_sha=row['source_sha']+config['build_manifest_sha256']+config['harness_sha256'])
            digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            if row.get('id')!=digest or path.stem!=digest:
                issues.append({'file':path.name, 'problem':'cell filename/id differs from its workload/source identity'})
            if row.get('repeat') != config['repeat'] or row.get('min_time') != config['min_time']:
                issues.append({'file':path.name, 'problem':'cell timing policy differs from run configuration'})
            if row.get('status') not in STATUSES:
                issues.append({'file':path.name, 'problem':'unknown cell status'})
            if row.get('status') in ('unsupported','unavailable','build_failed','execution_failed','resource_limited') and not row.get('reason'):
                issues.append({'file':path.name, 'problem':'outcome lacks an explicit reason'})
            counts[row.get('status', 'missing_status')] += 1
            recorded += 1
            if build is not None:
                backend = build['builds'].get(row['backend'], {})
                sha = backend.get('sha', backend.get('requirement', 'unavailable'))
                if row.get('source_sha') != sha or row.get('build_mode') != backend.get('build_kind'):
                    issues.append({'file':path.name, 'problem':'source revision or build mode differs from manifest'})
                if row.get('status') == 'timed_successfully':
                    try:
                        for problem in check_timing(row, config, build):
                            issues.append({'file':path.name, 'problem':problem})
                    except (ValueError, TypeError, KeyError) as exc:
                        issues.append({'file':path.name, 'problem':f'incomplete timing evidence: {exc}'})
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append({'file':path.name, 'problem':f'malformed cell: {exc}'})
    missing = expected-present
    return {'schema':1, 'expected_cells':len(expected), 'recorded_cells':recorded,
            'missing_cells':len(missing), 'unexpected_cells':len(present-expected),
            'status_counts':dict(sorted(counts.items())), 'evidence_issues':issues,
            'coverage_complete':not missing and not (present-expected),
            'evidence_consistent':not issues,
            'all_requested_cells_accounted_for':not missing and not issues,
            'missing_case_examples':[list(key) for key in sorted(missing)[:20]],
            'note':'Coverage includes explicit unsupported and numerical-failure outcomes; it does not imply universal numerical validity or performance superiority.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--build-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    result = audit(args.directory, build_manifest=args.build_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['evidence_consistent'] and (not args.require_complete or result['coverage_complete']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
