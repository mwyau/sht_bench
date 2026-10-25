from __future__ import annotations

import hashlib
import io
import json
import runpy
from pathlib import Path

import pytest
from packaging.tags import Tag


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_source_resolution_sweep.py'
SWEEP = runpy.run_path(str(SCRIPT), run_name='sht_bench_resolution_sweep_test')


def test_wheel_resolution_selects_compatible_pypi_binary_and_verifies_hash(tmp_path,monkeypatch):
    content=b'wheel fixture bytes'
    digest=hashlib.sha256(content).hexdigest()
    response_json={
        'urls':[
            {'packagetype':'bdist_wheel','filename':'pyspharm_syl-1.2.1-cp314-cp314-manylinux_2_34_x86_64.whl',
             'digests':{'sha256':'0'*64},'size':1,'url':'https://files.pythonhosted.org/cp314.whl'},
            {'packagetype':'bdist_wheel','filename':'pyspharm_syl-1.2.1-cp313-cp313-manylinux_2_34_x86_64.whl',
             'digests':{'sha256':digest},'size':len(content),'url':'https://files.pythonhosted.org/cp313.whl'},
        ],
    }
    responses=[json.dumps(response_json).encode(),content]

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self,*args):
            self.close()

    def fake_urlopen(url,timeout):
        return Response(responses.pop(0))

    namespace=SWEEP['_resolve_pyspharm_syl_wheel'].__globals__
    monkeypatch.setitem(namespace,'urlopen',fake_urlopen)
    monkeypatch.setitem(namespace,'sys_tags',lambda:[Tag('cp313','cp313','manylinux_2_34_x86_64')])
    result=SWEEP['_resolve_pyspharm_syl_wheel'](tmp_path)
    assert result['filename']=='pyspharm_syl-1.2.1-cp313-cp313-manylinux_2_34_x86_64.whl'
    assert result['sha256']==digest
    assert Path(result['local_path']).read_bytes()==content


def test_pyspharm_probe_rejects_distribution_substitution():
    valid={
        'distribution_name':'pyspharm-syl','distribution_version':'1.2.1',
        'pyspharm_distributions':['pyspharm-syl'],'wheel_metadata':'Tag: cp313-cp313-manylinux_2_34_x86_64',
        'installer':'uv',
    }
    SWEEP['_validate_pyspharm_syl_probe'](valid)
    substituted={**valid,'distribution_name':'pyspharm','pyspharm_distributions':['pyspharm']}
    with pytest.raises(ValueError,match='locked pyspharm-syl'):
        SWEEP['_validate_pyspharm_syl_probe'](substituted)
    extra={**valid,'pyspharm_distributions':['pyspharm-syl','pyspharm']}
    with pytest.raises(ValueError,match='substitution'):
        SWEEP['_validate_pyspharm_syl_probe'](extra)

