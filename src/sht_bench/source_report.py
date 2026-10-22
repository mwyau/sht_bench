"""Regenerate figures and matched comparisons from source-matrix raw cells."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics

COLORS={'ducc':'#0072B2','shtns':'#D55E00','rustsht':'#009E73','pyspharm':'#CC79A7'}


def matched_key(r):
    return (json.dumps(r['workload'],sort_keys=True),r['threads_requested'],r['round'])


def csv_file(path,rows):
    if not rows: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(directory,output):
    rows=[json.loads(p.read_text()) for p in sorted((directory/'cells').glob('*.json'))]
    hashes={r.get('harness_sha256') for r in rows}
    builds={r.get('build_manifest_sha256') for r in rows}
    if len(hashes)>1 or len(builds)>1:
        raise ValueError('report directory mixes benchmark implementations or build manifests')
    output.mkdir(parents=True,exist_ok=True)
    counts=Counter(r['status'] for r in rows)
    valid=[r for r in rows if r['status']=='timed_successfully']
    index={(r['backend'],matched_key(r)):r for r in valid}
    comparisons=[]
    for r in valid:
        if r['backend'] not in ('rustsht','shtns'): continue
        reference=index.get(('ducc',matched_key(r)))
        if reference is None: continue
        comparisons.append(dict(backend=r['backend'],reference='ducc',source_sha=r['source_sha'],reference_sha=reference['source_sha'],
                                **r['workload'],threads=r['threads_requested'],round=r['round'],
                                latency_ms=1000*r['median_s'],reference_ms=1000*reference['median_s'],
                                ratio=r['median_s']/reference['median_s'],excess_ms=1000*(r['median_s']-reference['median_s'])))
    csv_file(output/'matched-ratios.csv',comparisons)
    flat=[dict(backend=r['backend'],source_sha=r['source_sha'],build_mode=r['build_mode'],**r['workload'],
               threads=r['threads_requested'],round=r['round'],median_ms=1000*r['median_s'],p10_ms=1000*r['p10_s'],p90_ms=1000*r['p90_s'],
               setup_ms=1000*r['setup_s'],first_synthesis_ms=1000*r['first_synthesis_s'],first_analysis_ms=1000*r['first_analysis_s'],peak_rss_bytes=r['peak_rss_bytes']) for r in valid]
    csv_file(output/'timings.csv',flat)
    accuracy=[]
    for r in rows:
        for f in r.get('validation',{}).get('fixtures',[]):
            for stage in ('synthesis','full_grid_synthesis','common_reference_analysis','coefficient_recovery','map_roundtrip'):
                if stage not in f: continue
                accuracy.append(dict(id=r['id'],backend=r['backend'],source_sha=r['source_sha'],**r['workload'],threads=r['threads_requested'],
                                     fixture=f['name'],stage=stage,passed=f['passed'],**f[stage]))
    # Failed shape mismatches can add fields; retain every metric in JSON.
    (output/'accuracy.json').write_text(json.dumps(accuracy,indent=2)+'\n')
    figures=[]
    if valid:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups=defaultdict(list)
        for r in valid:
            w=r['workload']
            order='full' if w['mmax']==w['lmax'] else 'quarter'
            key=(w['grid'],w['spin'],w['precision'],w['allocation'],w['operation'],order,w['geometry_group'],w['batch'],w['phi0'])
            groups[key].append(r)
        for key,group in sorted(groups.items()):
            grid,spin,precision,allocation,op,order,geometry,batch,phi0=key
            base=f'{grid}-spin{spin}-{precision}-{allocation}-{op}-{order}-{geometry}-batch{batch}-phi{phi0}'
            for q in sorted({r['threads_requested'] for r in group}):
                selected=[r for r in group if r['threads_requested']==q]
                for quantity,label,scale in [('median_s','Warm transform (ms)',1000),('setup_s','Constructor and grid setup (ms)',1000),('peak_rss_bytes','Process peak RSS (MiB)',1/2**20)]:
                    fig,ax=plt.subplots(figsize=(7,4))
                    for backend in COLORS:
                        backend_rows=[r for r in selected if r['backend']==backend]
                        if not backend_rows: continue
                        shas={r['source_sha'] for r in backend_rows}
                        for sha in sorted(shas):
                            subset=[r for r in backend_rows if r['source_sha']==sha]
                            ls=sorted({r['workload']['lmax'] for r in subset})
                            ys=[statistics.median(r[quantity]*scale for r in subset if r['workload']['lmax']==l) for l in ls]
                            ax.plot(ls,ys,marker='o',linestyle='-' if backend=='pyspharm' else '--',color=COLORS[backend],label=f'{backend} {sha[:10]}')
                    ax.set(xlabel='Inclusive degree L',ylabel=label,title=f'{grid} spin {spin}, {precision}, {op}, q{q}\n{allocation}, {order} order, {geometry}')
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
                    ax.plot(qs,ys,marker='o',linestyle='-' if backend=='pyspharm' else '--',color=COLORS[backend],label=backend)
                ax.set(xlabel='Requested physical-core threads',ylabel='Warm transform (ms)',title=f'{base}, L{l}')
                ax.set_xscale('log',base=2);ax.set_yscale('log');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
                name=f'{base}-L{l}-threads.png';fig.savefig(output/name,dpi=150);plt.close(fig);figures.append(name)
    manifest={'status_counts':dict(counts),'raw_cells':len(rows),'timed_cells':len(valid),'matched_pairs':len(comparisons),'figures':figures,
              'source_directory':str(directory.resolve()),'command':f'uv run --extra plot python -m sht_bench.source_report {directory} --output {output}'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Source benchmark results','',f'{len(rows)} recorded outcomes; {len(valid)} timed after numerical validation.','',
           'Measurements are provisional on a shared host. Ratios pair the same grid, bandwidth, precision, allocation, operation, thread request, and process round.','',
           '| Outcome | Cells |','| --- | ---: |',*[f'| {k} | {v} |' for k,v in sorted(counts.items())],'',
           'Raw cells contain every timing sample, numerical metrics, CPU controls, and build-manifest reference. `matched-ratios.csv` records RustSHT/DUCC and SHTns/DUCC ratios and absolute latency excess in milliseconds. `accuracy.json` includes successful and failed validation fixtures. Peak RSS includes setup and validation buffers.','']
    for name in figures:
        lines.append(f'[![{name}]({name})]({name})\n')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    report(args.directory,args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
