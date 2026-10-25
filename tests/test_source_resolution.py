from __future__ import annotations

import numpy as np

from sht_bench.source_resolution import (
    BACKENDS, CC_PROFILE, GL_PROFILE, case_reference, planned_cells,
    plan_counts, resolution_cases,
)


def test_resolution_profile_has_exact_cc_and_gl_geometry_and_coordinates():
    cases = resolution_cases()
    cc = [case for case in cases if case['grid'] == 'CC']
    gl = [case for case in cases if case['grid'] == 'GL']
    assert [(case['nlat'],case['nlon'],case['lmax']) for case in cc] == [
        (item['nlat'],item['nlon'],item['lmax']) for item in CC_PROFILE
    ]
    assert [(case['N'],case['nlat'],case['nlon'],case['lmax']) for case in gl] == [
        (item['N'],item['nlat'],item['nlon'],item['lmax']) for item in GL_PROFILE
    ]
    assert len(cc) == len(gl) == 7
    assert cc[0]['latitude_coordinates_radians'][0] == np.pi / 2
    assert cc[0]['latitude_coordinates_radians'][-1] == -np.pi / 2
    assert gl[0]['nominal_longitude_resolution_deg'] == 90 / 32
    assert gl[-1]['case_id'] == 'GL-N320'
    assert gl[-1]['nlat'] * gl[-1]['nlon'] == 640 * 1280
    assert case_reference(cc[-1])['latitude_coordinates_sha256']
    assert case_reference(gl[-1])['colatitude_coordinates_sha256']


def test_resolution_plan_keeps_lmax_and_requests_native_full_and_half_mmax():
    cells = list(planned_cells(resolution_cases(),rounds=1))
    by_case = {}
    for cell in cells:
        if cell['backend'] == 'ducc':
            by_case.setdefault((cell['resolution_case']['case_id'],cell['mmax_mode']),cell['workload'])
    for case in resolution_cases():
        full=by_case[(case['case_id'],'full')]
        half=by_case[(case['case_id'],'half')]
        assert full['lmax'] == half['lmax'] == case['lmax']
        assert full['mmax'] == case['lmax']
        assert half['mmax'] == case['lmax'] // 2


def test_default_plan_counts_backend_api_exclusions_explicitly():
    counts=plan_counts(planned_cells(resolution_cases()))
    assert counts['total'] == 12_096
    assert counts['planned'] == 10_248
    assert counts['unsupported'] == 1_848
    assert tuple(counts['by_backend']) == BACKENDS
    assert counts['by_backend']['pyspharm-cpp']['unsupported'] > 0
    assert counts['by_backend']['pyspharm']['unsupported'] > 0
    assert counts['by_backend']['shtns']['unsupported'] == 0


def test_endpoint_smoke_keeps_only_small_cc_and_gl_cases():
    cases=resolution_cases(endpoints_only=True)
    assert [(case['grid'],case['case_id']) for case in cases] == [('CC','CC-2.5deg'),('GL','GL-N32')]


def test_source_resolution_cli_accepts_pyspharm_plan_without_building(tmp_path, capsys):
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_source_resolution_sweep.py'
    main = runpy.run_path(str(script), run_name='sht_bench_source_resolution_test')['main']

    result = main([
        '--plan-only', '--grid', 'CC', '--backend', 'pyspharm',
        '--threads', '1', '--precision', 'float32', '--mmax', 'full', '--rounds', '1',
    ])
    import json

    plan = json.loads(capsys.readouterr().out)
    assert result == 0
    assert tuple(plan['outcomes']['by_backend']) == ('pyspharm',)
    assert plan['outcomes']['total'] == 28
    assert plan['precision_rule'].startswith('SHTns only float64; conda-forge pyspharm')
