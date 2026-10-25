from __future__ import annotations

from sht_bench.source_resolution_report import (
    _optional_plot_availability,
    _raw_comparisons,
    _summaries,
)


def _record(backend, *, q=1, mmax=36, mode='full', nlon=144, median_s=.001, operation='analysis'):
    return {
        'backend':backend,'grid':'CC','case_id':'CC-2.5deg','resolution':'2.5°',
        'nlat':73,'nlon':nlon,'lmax':36,'mmax':mmax,'mmax_mode':mode,
        'precision':'float64','spin':0,'operation':operation,'threads_requested':q,
        'round':0,'status':'timed_successfully','median_s':median_s,
        'build_mode':'local-git-source-snapshot',
    }


def test_ratios_require_exact_workload_and_thread_match_and_keep_ms_difference():
    rows=[
        _record('ducc',median_s=.001),
        _record('pyspharm-cpp',median_s=.0015),
        _record('rustsht',nlon=256,median_s=.0005),
        _record('shtns',q=2,median_s=.0003),
    ]
    ratios,scaling,orders=_raw_comparisons(rows)
    assert len(ratios)==1
    assert ratios[0]['backend']=='pyspharm-cpp'
    assert ratios[0]['ratio_to_ducc']==1.5
    assert ratios[0]['absolute_difference_ms']==.5
    assert scaling==[]
    assert orders==[]


def test_thread_and_order_derivations_keep_native_order_support_distinct():
    rows=[
        _record('ducc',q=1,mmax=36,mode='full',median_s=.004),
        _record('ducc',q=2,mmax=36,mode='full',median_s=.002),
        _record('ducc',q=1,mmax=18,mode='half',median_s=.003),
        _record('pyspharm-cpp',q=1,mmax=36,mode='full',median_s=.005),
    ]
    ratios,scaling,orders=_raw_comparisons(rows)
    assert ratios and all(row['backend']=='pyspharm-cpp' for row in ratios)
    q2=next(row for row in scaling if row['backend']=='ducc' and row['threads_requested']==2)
    assert q2['speedup_vs_q1']==2
    order=next(row for row in orders if row['backend']=='ducc')
    assert order['half_over_full']==.75
    assert all(row['backend']!='pyspharm-cpp' for row in orders)


def test_summaries_allow_missing_native_setup_timing():
    record=_record('ducc',median_s=.001)
    record.update({
        'id':'ducc-cell','setup_s':.01,'native_setup_s':None,
        'first_analysis_s':.002,'first_synthesis_s':.003,
        'peak_rss_bytes':1000,
    })
    summary=_summaries([record],{'ducc-cell':{'samples_s':[.001,.0012]}})[0]
    assert summary['native_setup_s_median'] is None
    assert summary['setup_s_median']==.01


def test_single_backend_q1_run_omits_empty_comparison_figures():
    records=[_record('pyspharm')]
    assert _optional_plot_availability(records,[],[],'float64') == {
        'order':False,'matched':False,'threads':False,
    }


def test_optional_figures_require_matching_precision_and_multiple_threads():
    records=[_record('ducc',q=1),_record('ducc',q=4)]
    ratios=[{'precision':'float32'}]
    order=[{'precision':'float32'}]
    assert _optional_plot_availability(records,ratios,order,'float64') == {
        'order':False,'matched':False,'threads':True,
    }
    assert _optional_plot_availability(records,ratios,order,'float32') == {
        'order':True,'matched':True,'threads':False,
    }
