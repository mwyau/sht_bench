"""Regenerate figures and matched comparisons from source-matrix raw cells."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
from itertools import chain
import json
from pathlib import Path
import statistics
from .colors import BACKEND_COLORS

COLORS={name:BACKEND_COLORS[name] for name in ('ducc','shtns','rustsht','pyspharm')}
LINE_STYLES={'git-source':'--','published-wheel':'-'}
MARKERS={'git-source':'s','published-wheel':'o'}


def revision_label(value):
    return value[:10] if len(value)==40 and all(c in '0123456789abcdef' for c in value) else value


def figure_gallery(output, figures):
    """Bound the number of embedded figures per Markdown page."""
    pages=[f'figures-{i+1:04d}.md' for i in range((len(figures)+47)//48)] if len(figures)>48 else []
    expected_pages=set(pages)
    for stale in output.glob('figures-*.md'):
        if stale.name not in expected_pages:
            stale.unlink()
    if len(figures)<=48:
        return [f'[![{name}]({name})]({name})\n' for name in figures],[]
    index=['## Figures','',f'{len(figures)} figures; each page contains at most 48 previews.','']
    for i,page in enumerate(pages):
        first=i*48
        selected=figures[first:first+48]
        navigation=['[Results index](README.md)']
        if i:navigation.append(f'[Previous]({pages[i-1]})')
        if i+1<len(pages):navigation.append(f'[Next]({pages[i+1]})')
        lines=[f'# Figures {first+1}–{first+len(selected)}','', ' | '.join(navigation),'']
        lines.extend(f'[![{name}]({name})]({name})\n' for name in selected)
        (output/page).write_text('\n'.join(lines)+'\n')
        index.append(f'- [Figures {first+1}–{first+len(selected)}]({page})')
    return index,pages


def matched_key(r):
    return (json.dumps(r['workload'],sort_keys=True),r['threads_requested'],r['round'])


def csv_file(path,rows):
    rows=iter(rows)
    first=next(rows,None)
    if first is None:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(first))
        writer.writeheader()
        writer.writerows(chain((first,),rows))


def collect_for_report(directory,output,*,cell_paths=None):
    """Stream full accuracy/outcome exports and retain only valid plot fields."""
    fields=('id','backend','source_sha','build_mode','workload','threads_requested','round',
            'median_s','p10_s','p90_s','setup_s','setup_policy','native_setup_s','setup_components_s',
            'native_api','native_conventions','first_call_policy','planning_policy','threads_effective',
            'thread_control','operation_prepare_s','operation_prepare_policy','first_requested_operation_s',
            'first_synthesis_s','first_analysis_s','peak_rss_bytes')
    stages=('synthesis','full_grid_synthesis','common_reference_analysis','coefficient_recovery','map_roundtrip')
    valid=[]
    hashes=set()
    builds=set()
    counts=Counter()
    recorded=0
    provenance_path=None
    with (output/'accuracy.json').open('w') as accuracy, (output/'outcomes.csv').open('w',newline='') as outcomes:
        accuracy.write('[\n')
        first_metric=True
        writer=None
        paths=sorted((directory/'cells').glob('*.json')) if cell_paths is None else cell_paths
        for path in paths:
            r=json.loads(path.read_text())
            recorded+=1
            counts[r['status']]+=1
            hashes.add(r.get('harness_sha256'))
            builds.add(r.get('build_manifest_sha256'))
            if provenance_path is None and r.get('build_manifest'):
                provenance_path=Path(r['build_manifest'])
            outcome=dict(id=r['id'],backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],
                         **r['workload'],threads=r['threads_requested'],round=r['round'],status=r['status'],
                         reason=r.get('reason',''),raw_cell=f"cells/{r['id']}.json")
            if writer is None:
                writer=csv.DictWriter(outcomes,fieldnames=list(outcome))
                writer.writeheader()
            writer.writerow(outcome)
            maximum_error=0.0
            for fixture in r.get('validation',{}).get('fixtures',[]):
                for stage in stages:
                    if stage not in fixture:continue
                    metric=dict(id=r['id'],backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],
                                **r['workload'],threads=r['threads_requested'],round=r['round'],
                                fixture=fixture['name'],stage=stage,passed=fixture['passed'],**fixture[stage])
                    if r['workload']['operation'] not in ('analysis','synthesis'):
                        metric['validation_scope']='base_analysis_synthesis'
                    if not first_metric:accuracy.write(',\n')
                    json.dump(metric,accuracy,allow_nan=False)
                    first_metric=False
                    if r['status']=='timed_successfully':
                        maximum_error=max(maximum_error,fixture[stage]['relative_l2'])
            operation=r.get('operation_validation',{})
            if operation:
                identity=dict(id=r['id'],backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],
                              **r['workload'],threads=r['threads_requested'],round=r['round'],
                              fixture='random-triangular',passed=operation['passed'],validation_scope='requested_operation',
                              tolerance=operation['tolerance'])
                if 'independent_synthesis' in operation:
                    metric=dict(**identity,stage='requested_derivative_synthesis',**operation['independent_synthesis'])
                    if r['status']=='timed_successfully':
                        maximum_error=max(maximum_error,operation['independent_synthesis']['relative_l2'])
                else:
                    metric=dict(**identity,stage='requested_adjoint_identity',
                                **{key:operation[key] for key in ('adjoint_lhs','adjoint_rhs','scaled_residual')})
                if not first_metric:accuracy.write(',\n')
                json.dump(metric,accuracy,allow_nan=False)
                first_metric=False
            if r['status']=='timed_successfully':
                compact={key:r[key] for key in fields if key in r}
                compact['maximum_validation_relative_l2']=maximum_error
                valid.append(compact)
        accuracy.write('\n]\n')
    return valid,counts,recorded,hashes,builds,provenance_path


def report(directory,output,*,plots=True):
    from .source_audit import audit
    cell_paths=sorted((directory/'cells').glob('*.json'))
    evidence = audit(directory,cell_paths=cell_paths)
    output.mkdir(parents=True,exist_ok=True)
    (output/'evidence-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not evidence['evidence_consistent']:
        raise ValueError('source timing evidence is inconsistent; see evidence-audit.json')
    valid,counts,recorded,hashes,builds,provenance_path=collect_for_report(directory,output,cell_paths=cell_paths)
    if provenance_path is not None:
        (output/'build-provenance.json').write_bytes(provenance_path.read_bytes())
    if len(hashes)>1 or len(builds)>1:
        raise ValueError('report directory mixes benchmark implementations or build manifests')
    output.mkdir(parents=True,exist_ok=True)
    index={(r['backend'],matched_key(r)):r for r in valid}
    comparisons=[]
    for r in valid:
        if r['backend'] not in ('rustsht','shtns'): continue
        reference=index.get(('ducc',matched_key(r)))
        if reference is None: continue
        comparisons.append(dict(backend=r['backend'],reference='ducc',source_sha=r['source_sha'],reference_sha=reference['source_sha'],
                                build_mode=r['build_mode'],reference_build_mode=reference['build_mode'],
                                **r['workload'],threads=r['threads_requested'],round=r['round'],
                                latency_ms=1000*r['median_s'],reference_ms=1000*reference['median_s'],
                                ratio=r['median_s']/reference['median_s'],excess_ms=1000*(r['median_s']-reference['median_s'])))
    csv_file(output/'matched-ratios.csv',comparisons)
    paired=defaultdict(list)
    for row in comparisons:
        fields={k:v for k,v in row.items() if k not in ('round','latency_ms','reference_ms','ratio','excess_ms')}
        paired[json.dumps(fields,sort_keys=True)].append(row)
    paired_summary=[]
    for key,group in sorted(paired.items()):
        paired_summary.append(dict(**json.loads(key),process_pairs=len(group),
            median_ratio=statistics.median(r['ratio'] for r in group),
            minimum_ratio=min(r['ratio'] for r in group),maximum_ratio=max(r['ratio'] for r in group),
            median_excess_ms=statistics.median(r['excess_ms'] for r in group),
            median_latency_ms=statistics.median(r['latency_ms'] for r in group),
            median_reference_ms=statistics.median(r['reference_ms'] for r in group)))
    csv_file(output/'paired-summary.csv',paired_summary)
    serial={(r['backend'],r['source_sha'],json.dumps(r['workload'],sort_keys=True),r['round']):r
            for r in valid if r['threads_requested']==1}
    scaling=[]
    for r in valid:
        baseline=serial.get((r['backend'],r['source_sha'],json.dumps(r['workload'],sort_keys=True),r['round']))
        if baseline:
            speedup=baseline['median_s']/r['median_s']
            scaling.append(dict(backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],**r['workload'],threads=r['threads_requested'],
                                round=r['round'],parallel_speedup=speedup,parallel_efficiency=speedup/r['threads_requested']))
    csv_file(output/'thread-scaling.csv',scaling)
    flat=(dict(id=r['id'],backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],**r['workload'],
               threads=r['threads_requested'],round=r['round'],median_ms=1000*r['median_s'],p10_ms=1000*r['p10_s'],p90_ms=1000*r['p90_s'],
               setup_ms=1000*r['setup_s'],setup_policy=r['setup_policy'],
               native_setup_ms=1000*r['native_setup_s'] if r.get('native_setup_s') is not None else '',
               setup_components_s=json.dumps(r.get('setup_components_s',{}),sort_keys=True),native_api=r.get('native_api',''),
               native_conventions=json.dumps(r.get('native_conventions',{}),sort_keys=True),first_call_policy=r.get('first_call_policy','adapter-level A/S diagnostic; see recorded harness'),
               planning_policy=r['planning_policy'],threads_effective=r.get('threads_effective'),thread_control=r['thread_control'],
               operation_prepare_ms=1000*r['operation_prepare_s'],operation_prepare_policy=r.get('operation_prepare_policy',''),
               first_requested_operation_ms=1000*r['first_requested_operation_s'] if r.get('first_requested_operation_s') is not None else '',
               first_synthesis_ms=1000*r['first_synthesis_s'],first_analysis_ms=1000*r['first_analysis_s'],peak_rss_bytes=r['peak_rss_bytes']) for r in valid)
    csv_file(output/'timings.csv',flat)
    figures=[]
    if valid and plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups=defaultdict(list)
        for r in valid:
            w=r['workload']
            order='full' if w['mmax']==w['lmax'] else 'quarter'
            key=(w['grid'],w['spin'],w['precision'],w['allocation'],w['operation'],order,w['geometry_group'],w['batch'],w['phi0'],w['nlon']%2)
            groups[key].append(r)
        for key,group in sorted(groups.items()):
            grid,spin,precision,allocation,op,order,geometry,batch,phi0,parity=key
            longitude='odd' if parity else 'even'
            base=f'{grid}-spin{spin}-{precision}-{allocation}-{op}-{order}-{geometry}-batch{batch}-phi{phi0}-{longitude}'
            dimensions=defaultdict(set)
            for row in group:
                w=row['workload']
                dimensions[w['lmax']].add((w['nlat'],w['nlon'],w['mmax']))
            if any(len(shapes)>1 for shapes in dimensions.values()):
                raise ValueError(f'multiple geometries share a plot facet: {base}; use distinct geometry_group labels')
            title=f'{grid} spin {spin}, {precision}, {op}, {allocation}\n{order} order, {geometry}, batch {batch}, phi0={phi0}, {longitude} nlon'
            for q in sorted({r['threads_requested'] for r in group}):
                selected=[r for r in group if r['threads_requested']==q]
                for quantity,label,scale in [('median_s','Warm native call (ms)',1000),('setup_s','Adapter setup including coordinate checks (ms)',1000),
                                             ('native_setup_s','Base A/S constructor/grid/precomputation (ms)',1000),
                                             ('peak_rss_bytes','Process peak RSS (MiB)',1/2**20)]:
                    measured=[r for r in selected if r.get(quantity) is not None and r[quantity]>0]
                    if not measured:continue
                    fig,ax=plt.subplots(figsize=(7,4))
                    for backend in COLORS:
                        backend_rows=[r for r in measured if r['backend']==backend]
                        if not backend_rows: continue
                        shas={r['source_sha'] for r in backend_rows}
                        for sha in sorted(shas):
                            subset=[r for r in backend_rows if r['source_sha']==sha]
                            ls=sorted({r['workload']['lmax'] for r in subset})
                            ys=[statistics.median(r[quantity]*scale for r in subset if r['workload']['lmax']==l) for l in ls]
                            mode=subset[0]['build_mode']
                            ax.plot(ls,ys,marker='o',linestyle=LINE_STYLES.get(mode,':'),color=COLORS[backend],label=f'{backend} {mode} {revision_label(sha)}')
                    ax.set(xlabel='Inclusive degree L',ylabel=label,title=f'{title}, q{q}')
                    ax.set_xscale('log');ax.set_yscale('log');ax.legend(fontsize=7);ax.grid(alpha=.2)
                    fig.tight_layout()
                    name=f'{base}-q{q}-{quantity}.png'
                    fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
            # Thread scaling retains the exact geometry and degree.
            for l in sorted({r['workload']['lmax'] for r in group}):
                selected=[r for r in group if r['workload']['lmax']==l]
                if len({r['threads_requested'] for r in selected})<2: continue
                fig,ax=plt.subplots(figsize=(7,4))
                for backend in COLORS:
                    subset=[r for r in selected if r['backend']==backend]
                    if not subset:continue
                    qs=sorted({r['threads_requested'] for r in subset})
                    ys=[statistics.median(r['median_s']*1000 for r in subset if r['threads_requested']==q) for q in qs]
                    mode=subset[0]['build_mode']
                    label=f"{backend} {mode} {revision_label(subset[0]['source_sha'])}"
                    ax.plot(qs,ys,marker='o',linestyle=LINE_STYLES.get(mode,':'),color=COLORS[backend],label=label)
                ax.set(xlabel='Requested physical-core threads',ylabel='Warm transform (ms)',title=f'{base}, L{l}')
                ax.set_xscale('log',base=2);ax.set_yscale('log');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
                name=f'{base}-L{l}-threads.png';fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
            for q in sorted({r['threads_requested'] for r in group}):
                fig,ax=plt.subplots(figsize=(7,4))
                for backend in COLORS:
                    subset=[r for r in group if r['backend']==backend and r['threads_requested']==q]
                    if not subset:continue
                    degrees=sorted({r['workload']['lmax'] for r in subset})
                    xs=[];ys=[]
                    for degree in degrees:
                        case=[r for r in subset if r['workload']['lmax']==degree]
                        metrics=[r['maximum_validation_relative_l2'] for r in case]
                        xs.append(statistics.median(1000*r['median_s'] for r in case))
                        ys.append(max(max(metrics,default=0),1e-18))
                    label=f"{backend} {subset[0]['build_mode']} {revision_label(subset[0]['source_sha'])}"
                    ax.scatter(xs,ys,s=16,marker=MARKERS.get(subset[0]['build_mode'],'^'),color=COLORS[backend],label=label)
                    for degree,x,y in zip(degrees,xs,ys):
                        ax.annotate(str(degree),(x,y),fontsize=6,xytext=(3,3),textcoords='offset points')
                ax.set(xlabel='Warm native call (ms)',ylabel='Maximum relative L2 validation error',title=f'{title}, q{q}\nAnnotations give degree L; zero errors displayed at 1e-18')
                ax.set_xscale('log');ax.set_yscale('log');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
                name=f'{base}-q{q}-error-vs-latency.png';fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
    gallery_lines,gallery_pages=figure_gallery(output,figures)
    expected=evidence['expected_cells']
    pending=evidence['missing_cells']
    manifest={'status_counts':dict(counts),'expected_cells':expected,'pending_cells':pending,
              'raw_cells':recorded,'timed_cells':len(valid),'matched_pairs':len(comparisons),'figures':figures,
              'figure_pages':gallery_pages,
              'harness_sha256':next(iter(hashes),None),'build_manifest_sha256':next(iter(builds),None),
              'coverage_complete':evidence['coverage_complete'],'evidence_consistent':evidence['evidence_consistent'],
              'source_directory':str(directory.resolve()),'plots_requested':plots,
              'command':f'uv run --extra plot python -m sht_bench.source_report {directory} --output {output}'+('' if plots else ' --tables-only')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Source benchmark results','',f'{recorded} recorded outcomes of {expected}; {pending} pending. {len(valid)} timed after numerical validation.','',
           'Measurements are provisional on a shared host. Ratios pair the same grid, bandwidth, precision, allocation, operation, thread request, and process round.','',
           '| Outcome | Cells |','| --- | ---: |',*[f'| {k} | {v} |' for k,v in sorted(counts.items())],'',
           '`outcomes.csv` identifies every recorded status and its raw cell. `build-provenance.json` preserves the referenced build manifest. Raw cells contain every timing sample, numerical metrics, CPU controls, and build-manifest reference. `matched-ratios.csv` records RustSHT/DUCC and SHTns/DUCC ratios and absolute latency excess in milliseconds; `paired-summary.csv` summarizes those process pairs. `thread-scaling.csv` pairs each thread request with its q1 process round. `accuracy.json` includes successful and failed validation fixtures. Peak RSS includes setup and validation buffers.','',
           'For secondary workloads, accuracy entries distinguish base A/S gates from requested derivative synthesis and adjoint identity checks. Adjoint scaled residuals are separate from relative L2 synthesis errors.','']
    lines.extend(gallery_lines)
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tables-only',action='store_true',help='export audited ledgers and matched tables without rendering figures')
    args=parser.parse_args(argv)
    report(args.directory,args.output,plots=not args.tables_only)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
