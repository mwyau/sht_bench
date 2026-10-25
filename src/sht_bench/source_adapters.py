"""Adapters for explicit source-matrix workloads.

Canonical coefficients are orthonormal, include the Condon–Shortley phase,
use complex positive orders with real-field factor two, and are packed by m.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib
import time
import numpy as np

PLANNING_POLICIES = {
    'rustsht':'reusable ShtPlan constructed outside timing; one-plan workspace serialized; Rayon intra-transform pool',
    'ducc':'public DUCC transform call includes any API-internal setup/planning; no external reusable plan is exposed',
    'shtns':'reusable SHTns state constructed/configured outside transform timing',
    'pyspharm':'conda-forge pyspharm Spharmt legfunc=stored state constructed outside transform timing',
    'pyspharm-syl':'PyPI pyspharm-syl wheel Spharmt legfunc=stored state constructed outside transform timing',
    'pyspharm-cpp':'native pyspharm-cpp Spharmt legfunc=computed state constructed outside transform timing',
}

MODULES = {
    'ducc':'ducc0', 'rustsht':'rustsht', 'shtns':'shtns',
    'pyspharm':'spharm', 'pyspharm-syl':'spharm', 'pyspharm-cpp':'spharm_cpp',
}
PYSPHARM_BACKENDS = ('pyspharm',)
PYSPHARM_SYL_BACKENDS = ('pyspharm-syl',)
SPHARM_BACKENDS = (*PYSPHARM_BACKENDS, *PYSPHARM_SYL_BACKENDS)


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Workload:
    grid: str
    nlat: int
    nlon: int
    lmax: int
    mmax: int
    spin: int = 0
    precision: str = 'float64'
    allocation: str = 'allocating'
    batch: int = 1
    phi0: float = 0.0
    operation: str = 'synthesis'
    geometry_group: str = 'minimum'

    def __post_init__(self):
        if self.grid not in ('GL', 'CC', 'F1'):
            raise ValueError('unknown grid')
        if not 1 <= self.lmax or not 0 <= self.mmax <= self.lmax:
            raise ValueError('invalid inclusive degree/order')
        if self.nlat < self.lmax + (2 if self.grid == 'CC' else 1):
            raise ValueError('latitude count below exact analysis bound')
        if self.nlon <= 2 * self.mmax:
            raise ValueError('longitude count aliases positive orders')
        if self.spin not in (0, 1) or self.precision not in ('float32', 'float64'):
            raise ValueError('invalid spin/precision')
        if self.batch < 1 or self.allocation not in ('allocating', 'preallocated'):
            raise ValueError('invalid batch/allocation')

    @property
    def size(self):
        return sum(self.lmax - m + 1 for m in range(self.mmax + 1))

    @property
    def components(self):
        return 1 if self.spin == 0 else 2

    @property
    def cdtype(self):
        return np.complex64 if self.precision == 'float32' else np.complex128

    def record(self):
        return asdict(self)


def indices(w):
    return [(l, m) for m in range(w.mmax + 1) for l in range(m, w.lmax + 1)]


def _fixtures(w):
    rng = np.random.default_rng(20261022 + w.lmax)
    a = (rng.normal(size=(w.components, w.size)) + 1j*rng.normal(size=(w.components, w.size))) / (w.lmax+1)
    a[:, :w.lmax+1] = a[:, :w.lmax+1].real
    if w.spin:
        a[:, 0] = 0
    yield 'random-triangular', a.astype(w.cdtype)
    modes = {(max(1, w.spin), 0), (1, min(1,w.mmax)), (w.lmax, 0), (w.lmax, w.mmax)}
    ix = indices(w)
    for l, m in sorted(modes):
        for component in range(w.components):
            for part in ('real', 'imag') if m else ('real',):
                a = np.zeros((w.components,w.size), dtype=w.cdtype)
                a[component, ix.index((l,m))] = 1 if part == 'real' else 1j
                yield f'mode-{l}-{m}-{component}-{part}', a
    if not w.spin:
        a = np.zeros((1,w.size), dtype=w.cdtype)
        a[0,0] = 1
        yield 'Y00', a


def fixtures(w):
    for name,a in _fixtures(w):
        if w.batch>1:
            a=np.stack([a*np.array((i+1)/w.batch,dtype=w.precision) for i in range(w.batch)])
        yield name,a


def capability(backend, w):
    if backend not in MODULES:
        return 'unknown backend'
    if w.batch != 1 and backend not in SPHARM_BACKENDS and backend != 'pyspharm-cpp' and not (backend=='ducc' and w.operation=='synthesis'):
        return 'selected reusable-plan / 2d API has no native batch dimension'
    if backend in (*SPHARM_BACKENDS, 'pyspharm-cpp') and w.grid == 'F1':
        return 'no native F1 path'
    if backend in ('shtns', *SPHARM_BACKENDS, 'pyspharm-cpp') and w.phi0:
        return 'native longitude-origin control unavailable'
    if backend == 'pyspharm' and w.grid == 'GL' and w.lmax == 16 and w.nlat == 32 and w.nlon == 33:
        return 'Spherepack vhsgsi initialization returns ierror=3 for GL 32x33 at L16 in the conda-forge build'
    if backend in SPHARM_BACKENDS:
        if w.nlat < 3 or w.nlon < 4: return 'Spherepack Spharmt requires nlat>=3 and nlon>=4'
        if w.precision != 'float32':
            origin = 'conda-forge pyspharm 1.0.9' if backend == 'pyspharm' else 'PyPI pyspharm-syl 1.2.1'
            return f'{origin} (Spherepack) runtime uses float32/complex64'
        if w.mmax != w.lmax: return 'public API has full triangular truncation only'
        if w.allocation != 'allocating': return 'no public output-buffer API'
        if w.operation not in ('analysis','synthesis','gradient','deriv1'): return 'no native requested operation'
        if backend == 'pyspharm' and w.grid == 'GL' and w.spin == 1 and w.nlat <= w.lmax + 1:
            return 'GL spin-1 validation requires nlat>=lmax+2; highest-degree modes vanish at nlat=lmax+1'
    if backend == 'pyspharm-cpp':
        if w.nlat < 3 or w.nlon < 4: return 'pyspharm-cpp Spharmt requires nlat>=3 and nlon>=4'
        if w.mmax != w.lmax: return 'public Spharmt API has full triangular truncation only'
        if w.allocation != 'allocating': return 'no public output-buffer API'
        if w.operation not in ('analysis','synthesis'): return 'no native requested operation'
    if backend == 'shtns':
        if w.precision != 'float64': return 'Python API uses float64/complex128'
        if w.operation not in ('analysis','synthesis','gradient','deriv1'): return 'no selected native adjoint API'
        if w.nlat < 32: return 'SHTns SIMD grid requires at least 32 latitude nodes'
    if backend == 'rustsht' and w.precision != 'float64':
        if w.allocation == 'preallocated': return 'ShtPlan32 exposes no output-buffer methods'
        if w.operation not in ('analysis','synthesis','deriv1'): return 'ShtPlan32 lacks the requested operation'
    if w.operation not in ('analysis','synthesis','adjoint_analysis','adjoint_synthesis','gradient','deriv1'):
        return 'unknown operation'
    if w.operation in ('gradient','deriv1') and w.spin != 0:
        return 'derivative input must be scalar'
    if w.allocation == 'preallocated' and w.operation not in ('analysis','synthesis'):
        return 'requested operation has no output-buffer adapter'
    return None


class Adapter:
    def __init__(self, backend, w, threads):
        reason = capability(backend,w)
        if reason: raise Unsupported(reason)
        self.backend, self.w, self.threads = backend,w,threads
        self.effective_threads = None
        self.thread_control = 'not queryable'
        self.setup_components_s = {}
        self.native_setup_s = None
        self.native_api = None
        self.planning_policy = PLANNING_POLICIES[backend]
        self.module = importlib.import_module(MODULES[backend])
        self.encode = lambda a: np.ascontiguousarray(a)
        self.decode = lambda a: np.asarray(a)
        if backend == 'rustsht':
            plan_type = self.module.ShtPlan
            if w.precision == 'float32':
                if not hasattr(self.module._rustsht,'ShtPlan32'):
                    raise Unsupported('selected RustSHT revision does not expose ShtPlan32')
                plan_type = self.module._rustsht.ShtPlan32
            started = time.perf_counter()
            self.plan = plan_type(w.grid,w.nlat,w.nlon,w.lmax,w.mmax,spin=w.spin,phi0=w.phi0)
            self.setup_components_s['native_constructor_grid_plan'] = time.perf_counter()-started
            self.native_setup_s = self.setup_components_s['native_constructor_grid_plan']
            self.setup_policy = 'public ShtPlan constructor includes grid and plan; no finer public split; lazy workspace may appear in first-call measurements'
            self.native_api = plan_type.__name__
            self.thread_control = 'Rayon global pool; RAYON_NUM_THREADS set before import; effective count not exposed'
        elif backend == 'ducc':
            self.setup_policy = 'no reusable public 2d plan; adapter metadata preparation measured separately'
            self.native_api = 'ducc0.sht.*_2d'
            if w.batch>1:
                from .source_batch import DuccBatchSynthesis
                started = time.perf_counter()
                self.batch_plan = DuccBatchSynthesis(self.module,w,threads)
                self.setup_components_s['batch_ring_coordinate_metadata'] = time.perf_counter()-started
                self.native_api = 'ducc0.sht.synthesis native batch on explicit rings'
            self.thread_control = 'explicit per-call nthreads; effective worker count not exposed'
        elif backend == 'shtns':
            s = self.module
            started = time.perf_counter()
            self.plan = s.sht(w.lmax,w.mmax,mres=1,norm=s.sht_orthonormal,nthreads=threads)
            self.setup_components_s['native_constructor'] = time.perf_counter()-started
            flag={'GL':s.sht_gauss,'CC':s.sht_reg_poles,'F1':s.sht_reg_fast}[w.grid]
            started = time.perf_counter()
            self.plan.set_grid(nlat=w.nlat,nphi=w.nlon,flags=flag|s.SHT_PHI_CONTIGUOUS,polar_opt=0.0)
            self.setup_components_s['native_grid_precomputation'] = time.perf_counter()-started
            self.native_setup_s = self.setup_components_s['native_constructor']+self.setup_components_s['native_grid_precomputation']
            self.setup_policy = 'public sht constructor and set_grid/precomputation separately measured; independent coordinate checks excluded from native setup'
            self.native_api = 'shtns.sht'
            self.thread_control = 'explicit constructor nthreads; OMP_NUM_THREADS set before import; effective worker count not exposed'
            started = time.perf_counter()
            if (self.plan.nlat,self.plan.nphi) != (w.nlat,w.nlon):
                raise Unsupported('SHTns changed requested grid')
            theta = np.arccos(self.plan.cos_theta)
            from .source_validation import nodes
            if not np.allclose(theta,nodes(w),rtol=0,atol=1e-12):
                raise Unsupported('SHTns latitude nodes differ from requested geometry')
            self.setup_components_s['independent_coordinate_validation'] = time.perf_counter()-started
            started = time.perf_counter()
            if w.spin:
                degrees = np.array([l for l,m in indices(w)])
                scale = np.sqrt(degrees*(degrees+1))
                scale[0] = 1
                self.factor = np.stack((-1/scale,1/scale))
                self.encode = lambda a: np.ascontiguousarray(a*self.factor)
                self.decode = lambda a: np.asarray(a)/self.factor
            self.setup_components_s['coefficient_convention_preparation'] = time.perf_counter()-started
        elif backend in SPHARM_BACKENDS:
            started = time.perf_counter()
            self.plan = self.module.Spharmt(w.nlon,w.nlat,rsphere=1.0,gridtype='gaussian' if w.grid=='GL' else 'regular',legfunc='stored')
            self.setup_components_s['native_constructor_grid_precomputation'] = time.perf_counter()-started
            self.native_setup_s = self.setup_components_s['native_constructor_grid_precomputation']
            self.setup_policy = 'public Spharmt constructor includes scalar/vector stored Legendre precomputation; no finer public split'
            self.native_api = 'spharm.Spharmt'
            self.thread_control = ('Spherepack API has no thread-count argument/query; OMP_NUM_THREADS requested before import; native linkage recorded in build manifest')
            started = time.perf_counter()
            # Spherepack uses no CS phase and Y00=1/sqrt(2).
            self.factor = np.array([(-1)**m / np.sqrt(2*np.pi) for l,m in indices(w)],dtype=np.float32)
            self.encode = lambda a: np.ascontiguousarray(a[0]*self.factor)
            self.decode = lambda a: np.asarray(a)[None,:]/self.factor
            if w.spin:
                scale = np.array([np.sqrt(l*(l+1)) if l else 1 for l,m in indices(w)],dtype=np.float32)
                self.factor *= scale
                self.encode = lambda a: np.ascontiguousarray(a*self.factor)
                self.decode = lambda a: np.asarray(a)/self.factor
            if w.batch>1:
                if w.spin:
                    self.encode=lambda a: np.ascontiguousarray(np.moveaxis(a,0,-1)*self.factor[None,:,None])
                    self.decode=lambda a: np.moveaxis(np.asarray(a),-1,0)/self.factor[None,None,:]
                else:
                    self.encode=lambda a: np.ascontiguousarray(a[:,0,:].T*self.factor[:,None])
                    self.decode=lambda a: np.asarray(a).T[:,None,:]/self.factor[None,None,:]
        elif backend == 'pyspharm-cpp':
            started = time.perf_counter()
            self.plan = self.module.Spharmt(
                w.nlon, w.nlat, rsphere=1.0,
                gridtype='gaussian' if w.grid=='GL' else 'regular',
                legfunc='computed', dtype=np.dtype(w.precision),
            )
            self.setup_components_s['native_constructor'] = time.perf_counter()-started
            self.native_setup_s = self.setup_components_s['native_constructor']
            self.setup_policy = 'public Spharmt constructor selects grid and precision; computed Legendre functions are evaluated within each transform'
            self.native_api = 'spharm_cpp.Spharmt'
            self.thread_control = 'DUCC0_NUM_THREADS set before the first numerical operation; effective count is not exposed'
            started = time.perf_counter()
            real_dtype = np.float32 if w.precision == 'float32' else np.float64
            self.factor = np.array([(-1)**m / np.sqrt(2*np.pi) for l,m in indices(w)], dtype=real_dtype)
            self.encode = lambda a: np.ascontiguousarray(a[0]*self.factor)
            self.decode = lambda a: np.asarray(a)[None,:]/self.factor[None,:]
            if w.spin:
                scale = np.array([np.sqrt(l*(l+1)) if l else 1 for l,m in indices(w)], dtype=real_dtype)
                self.factor *= scale
                self.encode = lambda a: np.ascontiguousarray(a*self.factor[None,:])
                self.decode = lambda a: np.asarray(a)/self.factor[None,:]
            if w.batch>1:
                if w.spin:
                    self.encode=lambda a: np.ascontiguousarray(np.moveaxis(a,0,-1)*self.factor[None,:,None])
                    self.decode=lambda a: np.moveaxis(np.asarray(a),-1,0)/self.factor[None,None,:]
                else:
                    self.encode=lambda a: np.ascontiguousarray(a[:,0,:].T*self.factor[:,None])
                    self.decode=lambda a: np.asarray(a).T[:,None,:]/self.factor[None,None,:]
            self.setup_components_s['coefficient_convention_preparation'] = time.perf_counter()-started
        if w.spin and backend in ('ducc','rustsht'):
            self.encode = lambda a: np.ascontiguousarray(-a)
            self.decode = lambda a: -np.asarray(a)
        self.native_conventions = {
            'canonical':'m-major packed complex positive orders; orthonormal; Condon–Shortley phase; real-field positive-order factor two; spin E is negative normalized scalar gradient',
            'conversion':'identity for scalar; negate canonical E/B for native spin' if backend in ('ducc','rustsht') else
                         ('scalar identity; spin S=-E/sqrt(l(l+1)), T=B/sqrt(l(l+1))' if backend=='shtns' else
                          'scalar a_native=(-1)^m*a/sqrt(2*pi); spin additionally multiplies sqrt(l(l+1)) to divergence/vorticity, calls getuv(vorticity,divergence), maps (u,v) to (-v,u)'),
            'conversion_timing':'coefficients, input component conversion and output buffers prepared outside transform timing; canonical output conversion is excluded from native first-call and warm timing',
        }

    def synthesis(self,a,out=None):
        w = self.w
        if self.backend == 'ducc':
            if w.batch>1:return self.batch_plan.synthesis(a,out)
            return self.module.sht.synthesis_2d(alm=a,spin=w.spin,lmax=w.lmax,mmax=w.mmax,geometry=w.grid,ntheta=w.nlat,nphi=w.nlon,phi0=w.phi0,nthreads=self.threads,**({'map':out} if out is not None else {}))
        if self.backend == 'rustsht':
            if out is None: return self.plan.synthesis(a)
            self.plan.synthesis_into(a,out)
            return out
        if self.backend == 'shtns':
            if out is not None:
                if w.spin: self.plan.SHsphtor_to_spat(a[0],a[1],out[0],out[1])
                else: self.plan.SH_to_spat(a[0],out[0])
                return out
            return np.asarray(self.plan.synth(*a)) if w.spin else self.plan.synth(a[0])[None,:,:]
        if w.spin:
            u,v = self.plan.getuv(a[1],a[0])
            result=np.stack((-v,u))
            return np.moveaxis(result,-1,0) if w.batch>1 else result
        result=self.plan.spectogrd(a)
        return np.moveaxis(result,-1,0)[:,None,:,:] if w.batch>1 else result[None,:,:]

    def analysis(self,m,out=None):
        w = self.w
        if self.backend == 'ducc':
            if w.batch>1:
                if out is not None:raise Unsupported('native batched analysis output-buffer API unavailable')
                return self.batch_plan.analysis_for_validation(m)
            return self.module.sht.analysis_2d(map=m,spin=w.spin,lmax=w.lmax,mmax=w.mmax,geometry=w.grid,phi0=w.phi0,nthreads=self.threads,**({'alm':out} if out is not None else {}))
        if self.backend == 'rustsht':
            if out is None: return self.plan.analysis(m)
            self.plan.analysis_into(m,out)
            return out
        if self.backend == 'shtns':
            if out is not None:
                if w.spin: self.plan.spat_to_SHsphtor(m[0],m[1],out[0],out[1])
                else: self.plan.spat_to_SH(m[0],out[0])
                return out
            return np.asarray(self.plan.analys(*m)) if w.spin else self.plan.analys(m[0])[None,:]
        if w.spin:
            u,v=(np.moveaxis(m[:,1],0,-1),-np.moveaxis(m[:,0],0,-1)) if w.batch>1 else (m[1],-m[0])
            vrt,div = self.plan.getvrtdivspec(u,v,ntrunc=w.lmax)
            return np.stack((div,vrt))
        field=np.moveaxis(m[:,0],0,-1) if w.batch>1 else m[0]
        return self.plan.grdtospec(field,ntrunc=w.lmax)

    def canonical_synthesis_output(self,result):
        """Convert a native allocating synthesis result outside native-call timing."""
        w = self.w
        if self.backend=='ducc' and w.batch>1:
            return result.reshape(w.batch,w.components,w.nlat,w.nlon)
        if self.backend=='shtns':
            return np.asarray(result) if w.spin else result[None,:,:]
        if self.backend in (*SPHARM_BACKENDS, 'pyspharm-cpp'):
            if w.spin:
                u,v = result
                result = np.stack((-v,u))
                return np.moveaxis(result,-1,0) if w.batch>1 else result
            return np.moveaxis(result,-1,0)[:,None,:,:] if w.batch>1 else result[None,:,:]
        return np.asarray(result)

    def operation(self,a,m,out=None,*,operation=None):
        op = self.w.operation if operation is None else operation
        if op in ('analysis','synthesis') and self.backend=='ducc':
            w = self.w
            if w.batch>1:
                if op=='analysis':
                    # Only a labelled first-call diagnostic / numerical gate.
                    return lambda: self.batch_plan.analysis_for_validation(m)
                native_fn = self.module.sht.synthesis
                kwargs = dict(alm=a,spin=w.spin,lmax=w.lmax,mmax=w.mmax,nthreads=self.threads,**self.batch_plan.rings)
                if out is None:return lambda: native_fn(**kwargs)
                kwargs['map'] = out.reshape(w.batch,w.components,-1)
            else:
                kwargs = dict(spin=w.spin,lmax=w.lmax,mmax=w.mmax,geometry=w.grid,phi0=w.phi0,nthreads=self.threads)
                if op=='synthesis':
                    native_fn = self.module.sht.synthesis_2d
                    kwargs.update(alm=a,ntheta=w.nlat,nphi=w.nlon)
                    if out is not None:kwargs['map']=out
                else:
                    native_fn = self.module.sht.analysis_2d
                    kwargs['map'] = m
                    if out is not None:kwargs['alm']=out
                if out is None:return lambda: native_fn(**kwargs)
            def into():
                native_fn(**kwargs)
                return out
            return into
        if op in ('analysis','synthesis') and self.backend=='rustsht':
            data = m if op=='analysis' else a
            native_fn = getattr(self.plan,op if out is None else op+'_into')
            if out is None:return lambda: native_fn(data)
            def into():
                native_fn(data,out)
                return out
            return into
        if op in ('analysis','synthesis') and self.backend=='shtns' and out is not None:
            if op=='synthesis':
                native_fn = self.plan.SHsphtor_to_spat if self.w.spin else self.plan.SH_to_spat
                args = (a[0],a[1],out[0],out[1]) if self.w.spin else (a[0],out[0])
            else:
                native_fn = self.plan.spat_to_SHsphtor if self.w.spin else self.plan.spat_to_SH
                args = (m[0],m[1],out[0],out[1]) if self.w.spin else (m[0],out[0])
            def into():
                native_fn(*args)
                return out
            return into
        if self.backend in (*SPHARM_BACKENDS, 'pyspharm-cpp'):
            if op in ('gradient','deriv1'):
                scale=np.array([np.sqrt(l*(l+1)) if l else 1 for l,order in indices(self.w)],dtype=np.float32)
                coeff=np.ascontiguousarray(-a/scale if op=='gradient' else -a)
                return lambda: self.plan.getgrad(coeff)
            if self.w.spin:
                if op=='synthesis': return lambda: self.plan.getuv(a[1],a[0])
                u,v=(np.moveaxis(m[:,1],0,-1),-np.moveaxis(m[:,0],0,-1)) if self.w.batch>1 else (m[1],-m[0])
                u,v=np.ascontiguousarray(u),np.ascontiguousarray(v)
                return lambda: self.plan.getvrtdivspec(u,v,ntrunc=self.w.lmax)
            if op=='synthesis': return lambda: self.plan.spectogrd(a)
            field=np.ascontiguousarray(np.moveaxis(m[:,0],0,-1) if self.w.batch>1 else m[0])
            return lambda: self.plan.grdtospec(field,ntrunc=self.w.lmax)
        if self.backend == 'shtns' and out is None and op in ('analysis','synthesis'):
            if op=='synthesis': return lambda: self.plan.synth(*a)
            return lambda: self.plan.analys(*m)
        if op == 'analysis': return lambda: self.analysis(m,out)
        if op == 'synthesis': return lambda: self.synthesis(a,out)
        if self.backend == 'shtns':
            scale = np.array([np.sqrt(l*(l+1)) if l else 1 for l,m in indices(self.w)])
            coeff = np.ascontiguousarray(-a[0]/scale if op=='gradient' else -a[0])
            coeff[0] = 0
            return lambda: self.plan.synth_grad(coeff)
        if self.backend == 'rustsht':
            plan=self.plan
            if op=='gradient':
                w=self.w
                plan=self.module.ShtPlan(w.grid,w.nlat,w.nlon,w.lmax,w.mmax,spin=1,phi0=w.phi0)
            fn = getattr(plan,{'gradient':'synthesis_gradient','deriv1':'synthesis_deriv1'}.get(op,op))
            if op in ('gradient','deriv1'):
                coeff = np.ascontiguousarray(-a)
                if op=='gradient':
                    # A scalar constant has zero gradient; spin-1 coefficients
                    # cannot contain an l=0 mode. Preserve the caller's input.
                    coeff[...,0] = 0
                return lambda: fn(coeff)
            return lambda: fn(m if op == 'adjoint_synthesis' else a)
        w = self.w
        if op in ('gradient','deriv1'):
            coeff = np.ascontiguousarray(-a)
            return lambda: self.module.sht.synthesis_2d(alm=coeff,spin=1,lmax=w.lmax,mmax=w.mmax,geometry=w.grid,ntheta=w.nlat,nphi=w.nlon,phi0=w.phi0,nthreads=self.threads,mode='GRAD_ONLY' if op=='gradient' else 'DERIV1')
        kwargs = dict(spin=w.spin,lmax=w.lmax,mmax=w.mmax,geometry=w.grid,phi0=w.phi0,nthreads=self.threads)
        if op == 'adjoint_synthesis':
            return lambda: self.module.sht.adjoint_synthesis_2d(map=m,**kwargs)
        return lambda: self.module.sht.adjoint_analysis_2d(alm=a,ntheta=w.nlat,nphi=w.nlon,**kwargs)

    def canonical_operation_output(self,result):
        if self.backend in SPHARM_BACKENDS and self.w.operation in ('gradient','deriv1'):
            u,v=result
            return np.stack((-v,u))
        return np.asarray(result)
