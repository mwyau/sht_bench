"""Native DUCC batch synthesis on explicit rings.

Analysis calls here are individual 2d transforms for numerical validation only;
they are not a native batched analysis workload.
"""
from __future__ import annotations

import numpy as np


class DuccBatchSynthesis:
    def __init__(self, module, workload, threads):
        from .source_validation import nodes
        self.module = module
        self.workload = workload
        self.threads = threads
        self.rings = {
            'theta':np.ascontiguousarray(nodes(workload),dtype=np.float64),
            'nphi':np.full(workload.nlat,workload.nlon,dtype=np.uintp),
            'phi0':np.full(workload.nlat,workload.phi0,dtype=np.float64),
            'ringstart':np.arange(workload.nlat,dtype=np.uintp)*workload.nlon,
        }

    def synthesis(self, alm, out=None):
        w = self.workload
        kwargs = {} if out is None else {'map':out.reshape(w.batch,w.components,-1)}
        result = self.module.sht.synthesis(
            alm=alm,spin=w.spin,lmax=w.lmax,mmax=w.mmax,
            nthreads=self.threads,**self.rings,**kwargs)
        return result.reshape(w.batch,w.components,w.nlat,w.nlon)

    def analysis_for_validation(self, field):
        w = self.workload
        return np.stack([
            self.module.sht.analysis_2d(
                map=part,spin=w.spin,lmax=w.lmax,mmax=w.mmax,
                geometry=w.grid,phi0=w.phi0,nthreads=self.threads)
            for part in field
        ])
