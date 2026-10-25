"""Audit and render resolution-sweep evidence without rerunning transforms."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median, pstdev

from .colors import BACKEND_COLORS

BACKEND_LABELS = {
    'ducc':'DUCC', 'shtns':'SHTns', 'rustsht':'RustSHT',
    'pyspharm-cpp':'pyspharm-cpp (DUCC-derived)', 'pyspharm-syl':'pyspharm-syl (PyPI wheel)',
}
BUILD_STYLES = {'local-git-source-snapshot':'--', 'conda-forge':'-.', 'pypi-wheel':'-'}
BUILD_CAPTION = 'pyspharm-syl: PyPI wheel'
MARKERS = ('o','s','^','D','v','P','X','<','>')
OUTCOME_FIELDS = (
    'id','backend','grid','case_id','resolution','nlat','nlon','lmax','mmax','mmax_mode',
    'precision','spin','operation','threads_requested','round','status','reason',
    'median_s','p10_s','p90_s','std_s','iterations','samples_count','transforms_per_second',
    'grid_points_per_second','setup_s','native_setup_s','first_synthesis_s','first_analysis_s',
    'rss_before_setup_bytes','rss_after_setup_bytes','peak_rss_bytes','threads_effective',
    'persistent_state_delta_bytes','setup_components_s_json','persistent_state_note',
    'thread_control','source_sha','source_snapshot_sha256','build_mode','build_origin','package_version',
    'artifact_sha256','python_version','numpy_version','import_path','raw_result_path',
)


def _read(path: Path):
    return json.loads(path.read_text())


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], records: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in records:
            writer.writerow({key:_csv_value(row.get(key)) for key in fields})


def _csv_value(value):
    if isinstance(value,(dict,list,tuple)):
        return json.dumps(value,sort_keys=True,separators=(',',':'))
    if value is None:
        return ''
    return value


def _audit(output: Path, plan: dict, build_hash: str, harness_hash: str) -> tuple[dict,dict[str,dict],dict[str,dict]]:
    expected = {cell['id']:cell for cell in plan['cells']}
    files = {path.stem:path for path in (output/'cells').glob('*.json')}
    results = {}
    issues = []
    build_manifest_path=output/'build-manifest.json'
    build_manifest=_read(build_manifest_path) if build_manifest_path.exists() else {}
    for backend,entry in build_manifest.get('builds',{}).items():
        if entry.get('status')!='built' or entry.get('build_kind')!='local-git-source-snapshot':
            continue
        revision=entry.get('resolved_sha')
        snapshot_hash=entry.get('build_snapshot_sha256')
        if entry.get('sha')!=revision or not isinstance(revision,str) or len(revision)!=40:
            issues.append(f'{backend} build is missing its exact 40-character Git revision')
        if not isinstance(snapshot_hash,str) or len(snapshot_hash)!=64:
            issues.append(f'{backend} build is missing its tracked source-snapshot checksum')
    syl=build_manifest.get('builds',{}).get('pyspharm-syl')
    if syl is not None and syl.get('status') == 'built':
        wheel=syl.get('pypi_wheel',{})
        wheel_path=Path(wheel.get('local_path',''))
        if syl.get('build_kind')!='pypi-wheel' or syl.get('build_origin')!='PyPI':
            issues.append('pyspharm-syl build is not identified as a PyPI wheel')
        if syl.get('package')!='pyspharm-syl' or syl.get('package_version')!='1.2.1':
            issues.append('pyspharm-syl package identity/version differs from the locked release')
        if syl.get('distribution_name')!='pyspharm-syl' or syl.get('distribution_version')!='1.2.1':
            issues.append('installed pyspharm-syl distribution identity/version is inconsistent')
        if syl.get('python_version')!='3.13.15' or syl.get('numpy')!='2.5.3':
            issues.append('pyspharm-syl environment differs from the locked Python/NumPy versions')
        if 'Tag: cp313-cp313-' not in (syl.get('wheel_metadata') or ''):
            issues.append('pyspharm-syl installed WHEEL metadata is not CPython 3.13')
        if not wheel.get('filename','').endswith('.whl') or not wheel.get('url','').startswith('https://files.pythonhosted.org/'):
            issues.append('pyspharm-syl artifact is not a resolved PyPI wheel')
        if not wheel_path.is_file():
            issues.append('resolved pyspharm-syl wheel artifact is missing from the run output')
        else:
            actual=hashlib.sha256(wheel_path.read_bytes()).hexdigest()
            if actual!=wheel.get('sha256') or actual!=syl.get('sha'):
                issues.append('pyspharm-syl wheel artifact checksum differs from the build manifest')
        if not syl.get('module_path') or not syl.get('native_module_path'):
            issues.append('pyspharm-syl import or compiled extension path is missing')
    if set(files)-set(expected):
        issues.append(f'{len(set(files)-set(expected))} unexpected result files')
    for identifier, cell in expected.items():
        path = files.get(identifier)
        if path is None:
            continue
        try:
            result = _read(path)
        except (OSError,json.JSONDecodeError) as exc:
            issues.append(f'{path.name}: malformed JSON: {exc}')
            continue
        for field, value in (
            ('id',identifier),('backend',cell['backend']),('workload',cell['workload']),
            ('threads_requested',cell['threads_requested']),('round',cell['round']),
            ('source_sha',cell['source_sha']),('resolution_case',cell['resolution_case']),
            ('validation_id',cell['validation_id']),('build_manifest_sha256',build_hash),
            ('harness_sha256',harness_hash),
        ):
            if result.get(field) != value:
                issues.append(f'{path.name}: {field} differs from the planned identity')
                break
        status = result.get('status')
        samples = result.get('samples_s')
        if status == 'timed_successfully':
            if not isinstance(samples,list) or not samples or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x <= 0 for x in samples):
                issues.append(f'{path.name}: successful cell lacks finite positive raw timing samples')
            elif not math.isclose(float(result.get('median_s',float('nan'))),median(samples),rel_tol=1e-12,abs_tol=0.0):
                issues.append(f'{path.name}: reported median does not match retained raw timing samples')
            if result.get('validation',{}).get('status') != 'supported_and_validated':
                issues.append(f'{path.name}: timed result lacks a passing independent numerical gate')
            if cell['static_status'] == 'unsupported':
                issues.append(f'{path.name}: unsupported cell contains a successful timing')
            cert_path = result.get('validation_certificate')
            cert_hash = result.get('validation_certificate_sha256')
            if not cert_path or not cert_hash or not Path(cert_path).is_file():
                issues.append(f'{path.name}: pre-timing validation certificate is missing')
            else:
                if hashlib.sha256(Path(cert_path).read_bytes()).hexdigest() != cert_hash:
                    issues.append(f'{path.name}: pre-timing validation checksum mismatch')
                else:
                    gate=_read(Path(cert_path))
                    from .source_runner import validation_identity
                    expected_identity=validation_identity({
                        'backend':cell['backend'],'source_sha':cell['source_sha'],
                        'build_manifest_sha256':build_hash,'harness_sha256':harness_hash,
                        'threads_requested':cell['threads_requested'],'workload':cell['workload'],
                    })
                    if gate.get('status')!='validation_passed' or gate.get('validation_identity')!=expected_identity:
                        issues.append(f'{path.name}: referenced numerical gate does not match its exact backend/workload/thread')
        elif samples is not None:
            issues.append(f'{path.name}: non-success cell has raw timing samples')
        if cell['static_status'] == 'unsupported' and status != 'unsupported':
            issues.append(f'{path.name}: native API limitation was not kept as unsupported')
        results[identifier] = result
    expected_validation = {}
    validation_path = output/'validation-plan.json'
    if validation_path.exists():
        validation_plan = _read(validation_path)
        expected_validation = {item['id']:item['identity'] for item in validation_plan['validations']}
    validation_files = {path.stem:path for path in (output/'validations').glob('*.json')}
    validations = {}
    for identifier, identity in expected_validation.items():
        path = validation_files.get(identifier)
        if path is None:
            continue
        try:
            result = _read(path)
        except (OSError,json.JSONDecodeError) as exc:
            issues.append(f'{path.name}: malformed validation JSON: {exc}')
            continue
        if result.get('validation_identity') != identity:
            issues.append(f'{path.name}: validation identity differs from validation plan')
        if result.get('id') != identifier:
            issues.append(f'{path.name}: validation filename/id mismatch')
        if result.get('status') == 'validation_passed' and result.get('validation',{}).get('status') != 'supported_and_validated':
            issues.append(f'{path.name}: validation status contradicts fixture results')
        fixtures=result.get('validation',{}).get('fixtures',[])
        if result.get('status') == 'validation_passed' and any(not fixture.get('passed',False) for fixture in fixtures):
            issues.append(f'{path.name}: passed validation contains a failed fixture')
        if result.get('status') == 'numerical_validation_failed' and result.get('validation',{}).get('status') == 'supported_and_validated':
            issues.append(f'{path.name}: numerical failure status contradicts fixture results')
        validations[identifier] = result
    unexpected_validations = sorted(set(validation_files)-set(expected_validation))
    if unexpected_validations:
        issues.append(f'{len(unexpected_validations)} unexpected validation files')
    missing_validations = sorted(set(expected_validation)-set(validations))
    if missing_validations:
        issues.append(f'{len(missing_validations)} required numerical validation outcomes are missing')
    status_counts = defaultdict(int)
    for result in results.values():
        status_counts[result.get('status','missing_status')]+=1
    missing = sorted(set(expected)-set(results))
    unexpected = sorted(set(files)-set(expected))
    audit = {
        'schema':1,'expected_cells':len(expected),'recorded_cells':len(results),
        'missing_cells':len(missing),'unexpected_cells':len(unexpected),
        'status_counts':dict(sorted(status_counts.items())),
        'evidence_issues':issues,'coverage_complete':not missing and not unexpected,
        'evidence_consistent':not issues,
        'all_requested_cells_accounted_for':not missing and not unexpected and not issues,
        'missing_examples':missing[:20],'unexpected_examples':unexpected[:20],
        'expected_validation_gates':len(expected_validation),'recorded_validation_gates':len(validations),
        'missing_validation_gates':len(missing_validations),
        'validation_status_counts':dict((key,sum(1 for value in validations.values() if value.get('status')==key)) for key in sorted({str(value.get('status')) for value in validations.values()})),
        'build_manifest_sha256':build_hash,'harness_sha256':harness_hash,
        'note':'Only cells with passing independent reference gates and successful raw timing samples contribute to performance comparisons.',
    }
    return audit,results,validations


def _cell_record(cell: dict, result: dict, path: Path | None, build: dict) -> dict:
    w = cell['workload']
    c = cell['resolution_case']
    before=result.get('rss_before_setup_bytes');after=result.get('rss_after_setup_bytes')
    return {
        **{field:result.get(field) for field in OUTCOME_FIELDS},
        'backend':cell['backend'],'grid':w['grid'],'case_id':c['case_id'],
        'resolution':c['resolution'],'nlat':w['nlat'],'nlon':w['nlon'],
        'lmax':w['lmax'],'mmax':w['mmax'],'mmax_mode':cell['mmax_mode'],
        'precision':w['precision'],'spin':w['spin'],'operation':w['operation'],
        'threads_requested':cell['threads_requested'],'round':cell['round'],
        'status':result.get('status'),'reason':result.get('reason'),
        'samples_count':len(result.get('samples_s',[])) if isinstance(result.get('samples_s'),list) else 0,
        'persistent_state_delta_bytes':after-before if isinstance(after,(int,float)) and isinstance(before,(int,float)) else None,
        'setup_components_s_json':result.get('setup_components_s'),
        'persistent_state_note':result.get('persistent_state_note'),
        'build_origin':build.get('build_origin'),
        'source_snapshot_sha256':build.get('build_snapshot_sha256'),
        'package_version':build.get('package_version') or build.get('distribution_version'),
        'artifact_sha256':build.get('pypi_wheel',{}).get('sha256'),
        'python_version':build.get('python_version'),
        'numpy_version':build.get('numpy'),
        'raw_result_path':str(path) if path is not None else '',
    }


def _median_or_none(values: list) -> float | None:
    finite_values = [value for value in values
                     if isinstance(value,(int,float)) and math.isfinite(value)]
    return median(finite_values) if finite_values else None


def _summaries(records: list[dict], results: dict[str,dict]) -> list[dict]:
    keys = ('backend','build_mode','build_origin','package_version','artifact_sha256','python_version','numpy_version','source_sha','source_snapshot_sha256',
            'grid','case_id','resolution','nlat','nlon','lmax','mmax','mmax_mode','precision','spin','operation','threads_requested')
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(k) for k in keys)].append(record)
    output = []
    for key,rows in sorted(grouped.items(),key=lambda pair:tuple(str(x) for x in pair[0])):
        successful = [row for row in rows if row['status']=='timed_successfully']
        raw = [sample for row in successful for sample in results[row['id']].get('samples_s',[])]
        process_medians = [row['median_s'] for row in successful if isinstance(row.get('median_s'),(int,float))]
        setup_values = [r.get('setup_s') for r in successful]
        native_setup_values = [r.get('native_setup_s') for r in successful]
        first_analysis_values = [r.get('first_analysis_s') for r in successful]
        first_synthesis_values = [r.get('first_synthesis_s') for r in successful]
        row = dict(zip(keys,key))
        statuses = defaultdict(int)
        for item in rows:
            statuses[item['status']]+=1
        row.update({
            'rounds_planned':len(rows),'rounds_successful':len(successful),
            'status_counts':dict(sorted(statuses.items())),
            'median_s':median(raw) if raw else None,
            'p10_s':_percentile(raw,.1) if raw else None,
            'p90_s':_percentile(raw,.9) if raw else None,
            'process_median_s':median(process_medians) if process_medians else None,
            'process_medians_s':process_medians,
            'process_cv_percent':(100*pstdev(process_medians)/median(process_medians)) if len(process_medians)>1 and median(process_medians)>0 else 0.0 if process_medians else None,
            'process_stability':'unstable (>20% process-median CV)' if len(process_medians)>1 and pstdev(process_medians)/median(process_medians)>.20 else 'no threshold flag',
            'setup_s_median':_median_or_none(setup_values),
            'native_setup_s_median':_median_or_none(native_setup_values),
            'first_analysis_s_median':_median_or_none(first_analysis_values),
            'first_synthesis_s_median':_median_or_none(first_synthesis_values),
            'peak_rss_bytes_max':max([r['peak_rss_bytes'] for r in successful if isinstance(r.get('peak_rss_bytes'),(int,float))],default=None),
            'transforms_per_second':1/median(raw) if raw else None,
            'grid_points_per_second':(row['nlat']*row['nlon'])/median(raw) if raw else None,
        })
        output.append(row)
    return output


def _percentile(values: list[float], fraction: float) -> float:
    ordered=sorted(values)
    if len(ordered)==1:return float(ordered[0])
    position=(len(ordered)-1)*fraction
    lower=math.floor(position); upper=math.ceil(position)
    if lower==upper:return float(ordered[lower])
    return float(ordered[lower]*(upper-position)+ordered[upper]*(position-lower))


def _accuracy_rows(validations: dict[str,dict], profile: dict, builds: dict[str,dict]) -> tuple[list[dict],dict[tuple,dict]]:
    case_lookup={(case['grid'],case['nlat'],case['nlon'],case['lmax']):case for case in profile['cases']}
    rows=[]
    maxima={}
    for identifier,result in validations.items():
        identity=result.get('validation_identity',{})
        w=identity.get('workload',{})
        case=case_lookup.get((w.get('grid'),w.get('nlat'),w.get('nlon'),w.get('lmax')),{})
        validation=result.get('validation') or {}
        build=builds.get(identity.get('backend'),{})
        base={
            'validation_id':identifier,'backend':identity.get('backend'),'grid':w.get('grid'),
            'source_sha':identity.get('source_sha'),'build_mode':build.get('build_kind'),
            'source_snapshot_sha256':build.get('build_snapshot_sha256'),
            'build_origin':build.get('build_origin'),'package_version':build.get('package_version') or build.get('distribution_version'),
            'artifact_sha256':build.get('pypi_wheel',{}).get('sha256'),
            'python_version':build.get('python_version'),'numpy_version':build.get('numpy'),
            'case_id':case.get('case_id'),'resolution':case.get('resolution'),
            'nlat':w.get('nlat'),'nlon':w.get('nlon'),'lmax':w.get('lmax'),
            'mmax':w.get('mmax'),'precision':w.get('precision'),'spin':w.get('spin'),
            'threads_requested':identity.get('threads_requested'),'status':result.get('status'),
            'tolerance':validation.get('tolerance'),
        }
        metrics=[]
        for fixture in validation.get('fixtures',[]):
            for metric_name,metric in fixture.items():
                if not isinstance(metric,dict) or not any(key in metric for key in ('relative_l2','max_absolute','scaled_error')):
                    continue
                rows.append({**base,'fixture':fixture.get('name'),'metric':metric_name,
                             'relative_l2':metric.get('relative_l2'),'max_absolute':metric.get('max_absolute'),
                             'scaled_error':metric.get('scaled_error'),'passed':fixture.get('passed')})
                for key in ('relative_l2','max_absolute','scaled_error'):
                    value=metric.get(key)
                    if isinstance(value,(int,float)) and math.isfinite(value):
                        metrics.append((key,float(value)))
        max_relative=max((value for key,value in metrics if key=='relative_l2'),default=None)
        max_absolute=max((value for key,value in metrics if key=='max_absolute'),default=None)
        max_scaled=max((value for key,value in metrics if key=='scaled_error'),default=None)
        maxima[(identity.get('backend'),w.get('grid'),case.get('case_id'),w.get('lmax'),w.get('mmax'),w.get('precision'),w.get('spin'),identity.get('threads_requested'))]={
            **base,'max_relative_l2':max_relative,'max_absolute':max_absolute,'max_scaled_error':max_scaled,
        }
    return rows,maxima


def _raw_comparisons(records: list[dict]) -> tuple[list[dict],list[dict],list[dict]]:
    matched_keys=('grid','case_id','nlat','nlon','lmax','mmax','mmax_mode','precision','spin','operation','threads_requested','round')
    paired=defaultdict(dict)
    for row in records:
        if row['status']!='timed_successfully':continue
        paired[tuple(row.get(key) for key in matched_keys)][row['backend']]=row
    ratios=[]
    for key,backends in paired.items():
        ducc=backends.get('ducc')
        if not ducc or not isinstance(ducc.get('median_s'),(int,float)):continue
        for backend,row in backends.items():
            if backend=='ducc' or not isinstance(row.get('median_s'),(int,float)):continue
            if row.get('python_version')!=ducc.get('python_version') or row.get('numpy_version')!=ducc.get('numpy_version'):
                continue
            base=dict(zip(matched_keys,key))
            ratios.append({**base,'backend':backend,'ducc_median_s':ducc['median_s'],'backend_median_s':row['median_s'],
                           'ratio_to_ducc':row['median_s']/ducc['median_s'],
                           'speedup_vs_backend':ducc['median_s']/row['median_s'],
                           'absolute_difference_ms':1000*(row['median_s']-ducc['median_s']),
                           'backend_build_mode':row.get('build_mode'),'backend_build_origin':row.get('build_origin'),
                           'backend_package_version':row.get('package_version'),'backend_artifact_sha256':row.get('artifact_sha256'),
                           'backend_source_sha':row.get('source_sha'),'backend_python_version':row.get('python_version'),
                           'backend_source_snapshot_sha256':row.get('source_snapshot_sha256'),
                           'backend_numpy_version':row.get('numpy_version'),
                           'ducc_build_mode':ducc.get('build_mode'),'ducc_source_sha':ducc.get('source_sha'),
                           'ducc_source_snapshot_sha256':ducc.get('source_snapshot_sha256'),
                           'ducc_python_version':ducc.get('python_version'),'ducc_numpy_version':ducc.get('numpy_version'),
                           'matched':True})
    provenance_keys=('build_mode','build_origin','package_version','artifact_sha256','python_version','numpy_version','source_sha','source_snapshot_sha256')
    thread_keys=('backend',*provenance_keys,'grid','case_id','resolution','nlat','nlon','lmax','mmax','mmax_mode','precision','spin','operation','round')
    threads=defaultdict(dict)
    for row in records:
        if row['status']=='timed_successfully':
            threads[tuple(row.get(key) for key in thread_keys)][row['threads_requested']]=row
    scaling=[]
    for key,values in threads.items():
        baseline=values.get(1)
        # A q=1 row is only a baseline; emit scaling when this same exact
        # workload also has at least one successful q>1 observation.
        if (not baseline or not isinstance(baseline.get('median_s'),(int,float))
                or not any(q > 1 for q in values)):
            continue
        for q,row in sorted(values.items()):
            speedup=baseline['median_s']/row['median_s']
            scaling.append({**dict(zip(thread_keys,key)),'threads_requested':q,'median_s':row['median_s'],
                            'speedup_vs_q1':speedup,'parallel_efficiency':speedup/q})
    order_keys=('backend',*provenance_keys,'grid','case_id','nlat','nlon','lmax','precision','spin','operation','threads_requested','round')
    orders=defaultdict(dict)
    for row in records:
        if row['status']=='timed_successfully':
            orders[tuple(row.get(key) for key in order_keys)][row['mmax_mode']]=row
    order=[]
    for key,values in orders.items():
        full=values.get('full');half=values.get('half')
        if full and half and isinstance(full.get('median_s'),(int,float)) and isinstance(half.get('median_s'),(int,float)):
            order.append({**dict(zip(order_keys,key)),'full_median_s':full['median_s'],'half_median_s':half['median_s'],
                          'half_over_full':half['median_s']/full['median_s'],'full_over_half_speedup':full['median_s']/half['median_s'],
                          'absolute_difference_ms':1000*(half['median_s']-full['median_s'])})
    return ratios,scaling,order


def _matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _plot_resolution(report: Path, records: list[dict], cases: list[dict], precision: str) -> None:
    plt=_matplotlib()
    fig,axes=plt.subplots(4,2,figsize=(14,15),constrained_layout=True)
    facet_specs=((0,'analysis','Scalar analysis'),(0,'synthesis','Scalar synthesis'),(1,'analysis','Vector analysis'),(1,'synthesis','Vector synthesis'))
    for row_index,(spin,operation,title) in enumerate(facet_specs):
        for col_index,grid in enumerate(('CC','GL')):
            ax=axes[row_index,col_index]
            chosen=[case for case in cases if case['grid']==grid]
            labels=[case['resolution'] if grid=='CC' else f"N={case['N']}" for case in chosen]
            x=list(range(len(chosen)))
            for backend in dict.fromkeys(record['backend'] for record in records):
                points=[]
                for case in chosen:
                    match=[r for r in records if r['backend']==backend and r['grid']==grid and r['case_id']==case['case_id'] and r['precision']==precision and r['mmax_mode']=='full' and r['spin']==spin and r['operation']==operation and r['threads_requested']==1 and r['status']=='timed_successfully']
                    values=[r['median_s'] for r in match if isinstance(r.get('median_s'),(int,float))]
                    points.append(median(values)*1000 if values else float('nan'))
                if all(math.isnan(value) for value in points):continue
                build=next((r.get('build_mode') for r in records if r['backend']==backend),None)
                ax.plot(x,points,color=BACKEND_COLORS[backend],marker='o',linestyle=BUILD_STYLES.get(build,'-'),label=BACKEND_LABELS.get(backend,backend))
            ax.set_yscale('log');ax.grid(True,which='both',alpha=.25)
            ax.set_xticks(x,labels,rotation=25 if grid=='CC' else 0)
            ax.set_title(f'{title} — {grid}')
            ax.set_ylabel('Median warm call (ms)' if col_index==0 else '')
            ax.set_xlabel('Nominal CC resolution' if grid=='CC' else 'Gaussian sample N')
            if row_index==0 and col_index==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'{precision}: full mmax, batch 1, q1; source/conda build style, fixed backend colors; {BUILD_CAPTION}')
    fig.savefig(report/f'resolution-{precision}.png',dpi=150)
    plt.close(fig)


def _plot_order(report: Path, order: list[dict], cases: list[dict], precision: str) -> None:
    plt=_matplotlib()
    fig,axes=plt.subplots(4,2,figsize=(14,15),constrained_layout=True)
    facet_specs=((0,'analysis','Scalar analysis'),(0,'synthesis','Scalar synthesis'),(1,'analysis','Vector analysis'),(1,'synthesis','Vector synthesis'))
    for row_index,(spin,operation,title) in enumerate(facet_specs):
        for col_index,grid in enumerate(('CC','GL')):
            ax=axes[row_index,col_index]
            chosen=[case for case in cases if case['grid']==grid]
            labels=[case['resolution'] if grid=='CC' else f"N={case['N']}" for case in chosen]
            x=list(range(len(chosen)))
            for backend in BACKEND_LABELS:
                points=[]
                for case in chosen:
                    values=[row['half_over_full'] for row in order if row['backend']==backend and row['grid']==grid and row['case_id']==case['case_id'] and row['precision']==precision and row['spin']==spin and row['operation']==operation and row['threads_requested']==1]
                    points.append(median(values) if values else float('nan'))
                if all(math.isnan(value) for value in points):continue
                ax.plot(x,points,color=BACKEND_COLORS[backend],marker='o',label=BACKEND_LABELS[backend])
            ax.axhline(1,color='#555555',linewidth=.8,linestyle=':');ax.grid(True,alpha=.25)
            ax.set_xticks(x,labels,rotation=25 if grid=='CC' else 0)
            ax.set_title(f'{title} — {grid}');ax.set_ylabel('half/full warm latency' if col_index==0 else '')
            ax.set_xlabel('Nominal CC resolution' if grid=='CC' else 'Gaussian sample N')
            if row_index==0 and col_index==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'{precision}: native full versus half mmax, batch 1, q1 (unsupported APIs have no ratio); {BUILD_CAPTION}')
    fig.savefig(report/f'order-{precision}.png',dpi=150);plt.close(fig)


def _plot_matched(report: Path, ratios: list[dict], cases: list[dict], precision: str, *, difference: bool=False) -> None:
    plt=_matplotlib()
    fig,axes=plt.subplots(4,2,figsize=(14,15),constrained_layout=True)
    facets=((0,'analysis','Scalar analysis'),(0,'synthesis','Scalar synthesis'),(1,'analysis','Vector analysis'),(1,'synthesis','Vector synthesis'))
    for row_index,(spin,operation,title) in enumerate(facets):
        for col_index,grid in enumerate(('CC','GL')):
            ax=axes[row_index,col_index]
            chosen=[case for case in cases if case['grid']==grid]
            labels=[case['resolution'] if grid=='CC' else f"N={case['N']}" for case in chosen]
            x=list(range(len(chosen)))
            for backend in BACKEND_LABELS:
                points=[]
                for case in chosen:
                    values=[item['absolute_difference_ms' if difference else 'ratio_to_ducc'] for item in ratios if item['backend']==backend and item['grid']==grid and item['case_id']==case['case_id'] and item['precision']==precision and item['mmax_mode']=='full' and item['spin']==spin and item['operation']==operation and item['threads_requested']==1]
                    points.append(median(values) if values else float('nan'))
                if all(math.isnan(value) for value in points):continue
                ax.plot(x,points,color=BACKEND_COLORS[backend],marker='o',label=BACKEND_LABELS[backend])
            if difference:
                ax.axhline(0,color='#555555',linewidth=.8,linestyle=':');ax.set_ylabel('Backend − DUCC (ms)' if col_index==0 else '')
            else:
                ax.axhline(1,color='#555555',linewidth=.8,linestyle=':');ax.set_yscale('log');ax.set_ylabel('Latency ratio to DUCC' if col_index==0 else '')
            ax.grid(True,which='both',alpha=.25);ax.set_xticks(x,labels,rotation=25 if grid=='CC' else 0)
            ax.set_title(f'{title} — {grid}');ax.set_xlabel('Nominal CC resolution' if grid=='CC' else 'Gaussian sample N')
            if row_index==0 and col_index==0:ax.legend(fontsize=8,ncol=2)
    suffix='differences' if difference else 'ratios'
    fig.suptitle(f'{precision}: matched DUCC comparisons, full mmax, q1; paired process rounds; {BUILD_CAPTION}')
    fig.savefig(report/f'{suffix}-{precision}.png',dpi=150);plt.close(fig)


def _plot_threads(report: Path, records: list[dict], cases: list[dict], precision: str) -> None:
    plt=_matplotlib()
    fig,axes=plt.subplots(4,2,figsize=(15,16),constrained_layout=True)
    facets=((0,'analysis','Scalar analysis'),(0,'synthesis','Scalar synthesis'),(1,'analysis','Vector analysis'),(1,'synthesis','Vector synthesis'))
    thread_counts=sorted({r['threads_requested'] for r in records if r['precision']==precision and r['status']=='timed_successfully'})
    case_markers={case['case_id']:MARKERS[index%len(MARKERS)] for index,case in enumerate(cases)}
    for row_index,(spin,operation,title) in enumerate(facets):
        for col_index,grid in enumerate(('CC','GL')):
            ax=axes[row_index,col_index]
            for case in (item for item in cases if item['grid']==grid):
                for backend in BACKEND_LABELS:
                    points=[];qs=[]
                    backend_records=[r for r in records if r['backend']==backend and r['case_id']==case['case_id'] and r['precision']==precision and r['mmax_mode']=='full' and r['spin']==spin and r['operation']==operation and r['status']=='timed_successfully']
                    for q in sorted({r['threads_requested'] for r in backend_records}):
                        values=[r['median_s'] for r in backend_records if r['threads_requested']==q]
                        qs.append(q);points.append(median(values)*1000)
                    if len(qs)<2:continue
                    build=backend_records[0].get('build_mode')
                    ax.plot(qs,points,color=BACKEND_COLORS[backend],marker=case_markers[case['case_id']],linestyle=BUILD_STYLES.get(build,'-'),alpha=.75)
            ax.set_xscale('log',base=2);ax.set_xticks(thread_counts,[str(q) for q in thread_counts]);ax.set_yscale('log')
            ax.grid(True,which='both',alpha=.25);ax.set_title(f'{title} — {grid}')
            ax.set_ylabel('Median warm call (ms)' if col_index==0 else '');ax.set_xlabel('Requested physical-core threads')
    from matplotlib.lines import Line2D
    backend_handles=[Line2D([0],[0],color=BACKEND_COLORS[name],label=BACKEND_LABELS[name]) for name in BACKEND_LABELS]
    marker_handles=[Line2D([0],[0],color='#444444',marker=case_markers[case['case_id']],linestyle='None',label=case['case_id']) for case in cases]
    fig.legend(handles=backend_handles+marker_handles,loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle(f'{precision}: threads {", ".join(map(str,thread_counts))} at every profile case, full mmax; marker identifies resolution; {BUILD_CAPTION}')
    fig.savefig(report/f'threads-{precision}.png',dpi=150);plt.close(fig)


def _plot_overheads(report: Path, records: list[dict], precision: str) -> None:
    plt=_matplotlib()
    facets=((0,'analysis','Scalar analysis'),(0,'synthesis','Scalar synthesis'),(1,'analysis','Vector analysis'),(1,'synthesis','Vector synthesis'))
    fig,axes=plt.subplots(4,2,figsize=(14,15),constrained_layout=True)
    for row_index,(spin,operation,title) in enumerate(facets):
        for col_index,grid in enumerate(('CC','GL')):
            ax=axes[row_index,col_index]
            points=[r for r in records if r['grid']==grid and r['precision']==precision and r['mmax_mode']=='full' and r['threads_requested']==1 and r['spin']==spin and r['operation']==operation and r['status']=='timed_successfully']
            for backend in BACKEND_LABELS:
                selected=[r for r in points if r['backend']==backend and isinstance(r.get('median_s'),(int,float)) and isinstance(r.get('setup_s'),(int,float))]
                if not selected:continue
                x=[r['median_s']*1000 for r in selected]
                y=[r['setup_s'] for r in selected]
                build=selected[0].get('build_mode')
                ax.scatter(x,y,color=BACKEND_COLORS[backend],marker='o',facecolors='none' if build=='conda-forge' else BACKEND_COLORS[backend],label=BACKEND_LABELS[backend])
                for point in selected:
                    if point.get('first_analysis_s') and point.get('first_synthesis_s'):
                        first=point['first_analysis_s'] if operation=='analysis' else point['first_synthesis_s']
                        ax.annotate(point['case_id'],(point['median_s']*1000,point['setup_s']),xytext=(2,2),textcoords='offset points',fontsize=5,alpha=.65)
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.25)
            ax.set_title(f'{title} — {grid}');ax.set_xlabel('Warm transform median (ms)');ax.set_ylabel('Constructor/grid setup (s)' if col_index==0 else '')
            if row_index==0 and col_index==0:ax.legend(fontsize=8,ncol=2)
    fig.suptitle(f'{precision}: setup cost versus warm latency, full mmax, q1; open markers are reserved for conda builds; {BUILD_CAPTION}')
    fig.savefig(report/f'setup-vs-warm-{precision}.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    for row_index,grid in enumerate(('CC','GL')):
        for col_index,spin in enumerate((0,1)):
            ax=axes[row_index,col_index]
            for record in records:
                if record['grid']!=grid or record['precision']!=precision or record['mmax_mode']!='full' or record['threads_requested']!=1 or record['spin']!=spin or record['status']!='timed_successfully':continue
                first=record.get('first_analysis_s') if record['operation']=='analysis' else record.get('first_synthesis_s')
                warm=record.get('median_s')
                if not isinstance(first,(int,float)) or not isinstance(warm,(int,float)):continue
                build=record.get('build_mode')
                marker='o' if record['operation']=='analysis' else '^'
                ax.scatter(warm*1000,first*1000,color=BACKEND_COLORS[record['backend']],marker=marker,facecolors='none' if build=='conda-forge' else BACKEND_COLORS[record['backend']])
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.25)
            ax.set_title(f'{grid}, {"vector" if spin else "scalar"}')
            ax.set_xlabel('Warm requested operation (ms)');ax.set_ylabel('First native call (ms)')
    fig.suptitle(f'{precision}: first-call versus warm latency, full mmax, q1; circles analysis, triangles synthesis; {BUILD_CAPTION}')
    fig.savefig(report/f'first-call-vs-warm-{precision}.png',dpi=150);plt.close(fig)


def _plot_accuracy(report: Path, records: list[dict], cases: list[dict], maxima: dict, precision: str) -> None:
    plt=_matplotlib()
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    for row_index,grid in enumerate(('CC','GL')):
        for col_index,spin in enumerate((0,1)):
            ax=axes[row_index,col_index]
            for record in records:
                if record['grid']!=grid or record['precision']!=precision or record['mmax_mode']!='full' or record['threads_requested']!=1 or record['spin']!=spin or record['status']!='timed_successfully':continue
                key=(record['backend'],grid,record['case_id'],record['lmax'],record['mmax'],precision,spin,1)
                accuracy=maxima.get(key)
                if not accuracy or not isinstance(accuracy.get('max_relative_l2'),(int,float)):continue
                marker='o' if record['operation']=='analysis' else '^'
                build=record.get('build_mode')
                ax.scatter(record['median_s']*1000,accuracy['max_relative_l2'],color=BACKEND_COLORS[record['backend']],marker=marker,facecolors='none' if build=='conda-forge' else BACKEND_COLORS[record['backend']])
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.25)
            ax.set_title(f'{grid}, {"vector" if spin else "scalar"}')
            ax.set_xlabel('Warm transform median (ms)');ax.set_ylabel('Maximum fixture relative L2 error')
    fig.suptitle(f'{precision}: numerical error versus latency; full mmax, q1; circles analysis, triangles synthesis; {BUILD_CAPTION}')
    fig.savefig(report/f'error-vs-latency-{precision}.png',dpi=150);plt.close(fig)


def _optional_plot_availability(records: list[dict], ratios: list[dict], order: list[dict], precision: str) -> dict[str, bool]:
    """Identify secondary plots that have actual matched data for this precision."""
    thread_counts = {
        int(record['threads_requested'])
        for record in records
        if record.get('precision') == precision and record.get('status') == 'timed_successfully'
    }
    return {
        'order': any(row.get('precision') == precision for row in order),
        'matched': any(row.get('precision') == precision for row in ratios),
        'threads': len(thread_counts) >= 2,
    }


def render(output: Path) -> dict:
    """Re-audit saved cells and rebuild summary tables and figures only."""
    global BUILD_CAPTION
    output=Path(output).resolve()
    plan=_read(output/'workload-plan.json')
    profile=_read(output/'resolution-profile.json')
    BACKEND_LABELS['pyspharm-syl']='pyspharm-syl (PyPI wheel)'
    BUILD_CAPTION='pyspharm-syl: PyPI wheel'
    build_manifest_path=output/'build-manifest.json'
    build_manifest={}
    if build_manifest_path.exists():
        build_manifest=_read(build_manifest_path)
        wheel_build=build_manifest.get('builds',{}).get('pyspharm-syl',{})
        wheel=wheel_build.get('pypi_wheel',{})
        version=wheel_build.get('package_version')
        filename=wheel.get('filename')
        digest=wheel.get('sha256')
        if version and filename and digest:
            BACKEND_LABELS['pyspharm-syl']=f'pyspharm-syl (PyPI wheel {version})'
            BUILD_CAPTION=f'pyspharm-syl {version} PyPI wheel {filename} sha256:{digest[:12]}'
    config=plan['configuration']
    audit,results,validations=_audit(output,plan,config['build_manifest_sha256'],config['harness_sha256'])
    (output/'final-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    report=output/'report';report.mkdir(exist_ok=True)
    cells_by_id={cell['id']:cell for cell in plan['cells']}
    records=[]
    for identifier,cell in cells_by_id.items():
        path=output/'cells'/f'{identifier}.json'
        result=results.get(identifier)
        if result is None:
            result={'id':identifier,'status':'pending','reason':'no outcome file was recorded'}
        build=build_manifest.get('builds',{}).get(cell['backend'],{})
        records.append(_cell_record(cell,result,path if path.exists() else None,build))
    records.sort(key=lambda r:(r['grid'],r['nlat'],r['backend'],r['precision'],r['spin'],r['operation'],r['mmax'],r['threads_requested'],r['round']))
    summaries=_summaries(records,results)
    accuracy_rows,maxima=_accuracy_rows(validations,profile,build_manifest.get('builds',{}))
    ratios,scaling,order=_raw_comparisons(records)
    _write_csv(report/'outcomes.csv',list(OUTCOME_FIELDS),records)
    summary_fields=sorted({key for row in summaries for key in row})
    _write_csv(report/'summary.csv',summary_fields,summaries)
    _write_csv(report/'accuracy.csv',('validation_id','backend','source_sha','source_snapshot_sha256','build_mode','build_origin','package_version','artifact_sha256','python_version','numpy_version','grid','case_id','resolution','nlat','nlon','lmax','mmax','precision','spin','threads_requested','status','tolerance','fixture','metric','relative_l2','max_absolute','scaled_error','passed'),accuracy_rows)
    _write_csv(report/'matched-ratios.csv',list(ratios[0]) if ratios else ('backend','grid','case_id','nlat','nlon','lmax','mmax','mmax_mode','precision','spin','operation','threads_requested','round','ratio_to_ducc','speedup_vs_backend','absolute_difference_ms'),ratios)
    _write_csv(report/'thread-scaling.csv',list(scaling[0]) if scaling else ('backend','grid','case_id','precision','spin','operation','threads_requested','speedup_vs_q1','parallel_efficiency'),scaling)
    _write_csv(report/'order-comparison.csv',list(order[0]) if order else ('backend','grid','case_id','precision','spin','operation','half_over_full','full_over_half_speedup'),order)
    optional_plots = {}
    for precision in config['precisions']:
        availability = _optional_plot_availability(records, ratios, order, precision)
        optional_plots[precision] = availability
        _plot_resolution(report,records,profile['cases'],precision)
        if availability['order']:
            _plot_order(report,order,profile['cases'],precision)
        else:
            (report/f'order-{precision}.png').unlink(missing_ok=True)
        if availability['matched']:
            _plot_matched(report,ratios,profile['cases'],precision)
            _plot_matched(report,ratios,profile['cases'],precision,difference=True)
        else:
            (report/f'ratios-{precision}.png').unlink(missing_ok=True)
            (report/f'differences-{precision}.png').unlink(missing_ok=True)
        if availability['threads']:
            _plot_threads(report,records,profile['cases'],precision)
        else:
            (report/f'threads-{precision}.png').unlink(missing_ok=True)
        _plot_overheads(report,records,precision)
        _plot_accuracy(report,records,profile['cases'],maxima,precision)
    readme=(
        '# CC/GL source-resolution sweep evidence\n\n'
        f'- Audit: `../final-audit.json` ({audit["recorded_cells"]}/{audit["expected_cells"]} outcomes recorded; complete={audit["coverage_complete"]}; consistent={audit["evidence_consistent"]}).\n'
        f'- Passing numerical gates: {audit["validation_status_counts"].get("validation_passed",0)}/{audit["expected_validation_gates"]} recorded.\n'
        '- Every successful cell retains its independent-process raw timing samples in `../cells/*.json`.\n'
        '- Accuracy records use an independent normalized associated-Legendre recurrence; no timings from failed gates are included in comparisons.\n'
        '- Ratios are matched on grid dimensions, L, mmax, precision, spin, operation, and thread count.\n'
        f'- Optional matched figures by precision: {json.dumps(optional_plots, sort_keys=True)}; figures are omitted when no matched outcomes exist.\n'
        f'- Backend colors are fixed; line style and markers identify build origin. The Spherepack series uses {BUILD_CAPTION}.\n'
        '- CC and GL are shown in separate facets. GL N=320 (640×1280) is not paired as an identical workload with CC 0.25° (721×1440, L=360).\n'
        '- `pyspharm-cpp` is DUCC-derived and is distinguished by its API and package/build metadata.\n'
    )
    (report/'README.md').write_text(readme)
    return audit
