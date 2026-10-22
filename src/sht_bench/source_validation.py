"""Independent normalized-Legendre reference and numerical validity gates."""
from __future__ import annotations

import numpy as np
from .source_adapters import fixtures


def nodes(w):
    if w.grid == 'GL':
        return np.arccos(np.polynomial.legendre.leggauss(w.nlat)[0][::-1])
    if w.grid == 'CC':
        return np.linspace(0,np.pi,w.nlat)
    return (np.arange(w.nlat)+0.5)*np.pi/w.nlat


def reference_points(w,a,theta,phi):
    """Evaluate real scalar or tangent-vector harmonics at paired coordinates.

    Uses a normalized associated-Legendre recurrence, independent of backends.
    E is minus the normalized scalar gradient; B is its 90-degree rotation.
    """
    if a.ndim==3:
        return np.stack([reference_points(w,part,theta,phi) for part in a])
    theta = np.asarray(theta,dtype=float)
    phi = np.asarray(phi,dtype=float)
    # Pole limits follow the continuous spherical-coordinate frame.
    t = theta
    x,s = np.cos(t),np.sin(t)
    out = np.zeros((w.components,len(t)))
    qmm = np.full_like(t,1/np.sqrt(4*np.pi))
    dmm = np.zeros_like(t)
    offset = 0
    for m in range(w.mmax+1):
        if m:
            c = -np.sqrt((2*m+1)/(2*m))
            qmm,dmm = c*s*qmm,c*(x*qmm+s*dmm)
        if not np.any(a[:,offset:offset+w.lmax-m+1]):
            offset += w.lmax-m+1
            continue
        qprev,dprev = np.zeros_like(t),np.zeros_like(t)
        q,d = qmm.copy(),dmm.copy()
        phase = np.exp(1j*m*phi)
        factor = 1 if m == 0 else 2
        for l in range(m,w.lmax+1):
            z = a[:,offset,None]*phase[None,:]
            if not w.spin:
                out[0] += factor*(z[0]*q).real
            elif l:
                norm = np.sqrt(l*(l+1))
                ratio = np.divide(q,s,out=np.zeros_like(q),where=np.abs(s)>1e-14)
                if m == 1:
                    ratio = np.where(np.abs(s)<=1e-14,d/x,ratio)
                dt,dp = -d/norm,-1j*m*ratio/norm
                out[0] += factor*(z[0]*dt-z[1]*dp).real
                out[1] += factor*(z[0]*dp+z[1]*dt).real
            offset += 1
            if l == w.lmax: break
            lp = l+1
            c1 = np.sqrt((4*lp*lp-1)/(lp*lp-m*m))
            c2 = np.sqrt(((lp-1)**2-m*m)/(4*(lp-1)**2-1)) if lp>1 else 0
            qnext = c1*(x*q-c2*qprev)
            dnext = c1*(-s*q+x*d-c2*dprev)
            qprev,q = q,qnext
            dprev,d = d,dnext
    return out


def errors(actual,expected):
    a,b = np.asarray(actual),np.asarray(expected)
    if a.shape != b.shape:
        return {'relative_l2': float('inf'),'max_absolute':float('inf'),'scaled_error':float('inf'),'shape_mismatch':[list(a.shape),list(b.shape)]}
    delta = np.abs(a-b).astype(np.float64)
    norm = np.linalg.norm(b.ravel())
    scale = max(float(np.max(np.abs(b))),np.finfo(float).tiny)
    return {'relative_l2':float(np.linalg.norm(delta.ravel())/max(norm,np.finfo(float).tiny)),
            'max_absolute':float(delta.max(initial=0)),
            'scaled_error':float(np.max(delta/(scale+np.abs(a)+np.abs(b)),initial=0))}


def acceptable(metrics,tolerance):
    return all(np.isfinite(metrics[k]) and metrics[k] <= tolerance for k in ('relative_l2','scaled_error'))


def validate(adapter):
    """Check independent synthesis points and full coefficient recovery before timing."""
    w = adapter.w
    tol = 5e-5 if w.precision == 'float32' else 2e-10
    # Include both poles, low and high latitudes, and several longitude phases.
    ii = np.unique(np.linspace(0,w.nlat-1,min(w.nlat,17),dtype=int))
    jj = np.unique(np.linspace(0,w.nlon-1,min(w.nlon,11),dtype=int))
    rows,cols = np.meshgrid(ii,jj,indexing='ij')
    rows,cols = rows.ravel(),cols.ravel()
    theta = nodes(w)[rows]
    phi = w.phi0+2*np.pi*cols/w.nlon
    report = {'status':'supported_and_validated','tolerance':tol,
              'reference':'independent normalized associated-Legendre recurrence at sampled grid nodes',
              'reference_points':len(rows),'fixtures':[]}
    for name,a in fixtures(w):
        native = adapter.encode(a)
        m = np.asarray(adapter.synthesis(native))
        recovered = adapter.decode(adapter.analysis(m))
        expected = reference_points(w,a,theta,phi)
        synth = errors(m[...,rows,cols],expected)
        full_synthesis = None
        common_recovery = None
        if name=='random-triangular':
            common = np.ascontiguousarray(reference_map(w,a),dtype=w.precision)
            full_synthesis = errors(m,common)
            common_recovery = errors(adapter.decode(adapter.analysis(common)),a)
        recovery = errors(recovered,a)
        repeated = np.asarray(adapter.synthesis(adapter.encode(recovered)))
        rt = errors(repeated,m)
        passed = all(acceptable(e,tol) for e in (synth,recovery,rt))
        dtype_passed = m.dtype==np.dtype(w.precision) and recovered.dtype==np.dtype(w.cdtype)
        passed = passed and dtype_passed and (full_synthesis is None or acceptable(full_synthesis,tol))
        passed = passed and (common_recovery is None or acceptable(common_recovery,tol))
        report['fixtures'].append({'name':name,'synthesis':synth,'coefficient_recovery':recovery,'map_roundtrip':rt,'passed':passed})
        report['fixtures'][-1]['dtype_passed'] = dtype_passed
        if full_synthesis is not None:
            report['fixtures'][-1]['full_grid_synthesis'] = full_synthesis
            report['fixtures'][-1]['common_reference_analysis'] = common_recovery
        if not passed:
            report['status'] = 'numerical_validation_failed'
    return report


def validate_operation(adapter,a,m,fn):
    """Validate derivative synthesis or a selected discrete adjoint identity."""
    from dataclasses import replace
    from .source_adapters import indices
    w = adapter.w
    tol = 5e-5 if w.precision == 'float32' else 2e-10
    op = w.operation
    if op in ('gradient','deriv1'):
        rows,cols = np.meshgrid(np.unique(np.linspace(0,w.nlat-1,min(w.nlat,17),dtype=int)),np.unique(np.linspace(0,w.nlon-1,min(w.nlon,11),dtype=int)),indexing='ij')
        rows,cols=rows.ravel(),cols.ravel()
        e = np.concatenate((a,np.zeros_like(a)),axis=0)
        if op=='deriv1':
            e *= np.array([np.sqrt(l*(l+1)) for l,order in indices(w)])
        expected = reference_points(replace(w,spin=1),e,nodes(w)[rows],w.phi0+2*np.pi*cols/w.nlon)
        actual = adapter.canonical_operation_output(fn())
        metric = errors(actual[:,rows,cols],expected)
        return {'passed':acceptable(metric,tol),'tolerance':tol,'independent_synthesis':metric}
    # Positive-order complex coefficients use the real-field inner product.
    weights = np.array([1 if order==0 else 2 for l,order in indices(w)])
    rng = np.random.default_rng(2801)
    x = rng.normal(size=m.shape).astype(w.precision)
    if op == 'adjoint_synthesis':
        # fn uses m, so an identity on the deterministic nonzero bandlimited map.
        y = adapter.decode(np.asarray(fn()))
        lhs = float(np.sum(adapter.synthesis(adapter.encode(a))*m))
        rhs = float(np.real(np.sum(weights*np.conj(a)*y)))
    else:
        adjoint_map = np.asarray(fn())
        lhs = float(np.real(np.sum(weights*np.conj(a)*adapter.decode(adapter.analysis(x)))))
        rhs = float(np.sum(adjoint_map*x))
    scaled = abs(lhs-rhs)/max(abs(lhs),abs(rhs),np.finfo(float).tiny)
    return {'passed':bool(np.isfinite(scaled) and scaled<=tol),'tolerance':tol,'adjoint_lhs':lhs,'adjoint_rhs':rhs,'scaled_residual':scaled}


def _reference_map(w,a):
    """Synthesize a complete grid by normalized recurrence and a NumPy inverse FFT."""
    theta = nodes(w)
    x,s = np.cos(theta),np.sin(theta)
    spectrum = np.zeros((w.components,w.nlat,w.nlon//2+1),dtype=np.complex128)
    qmm=np.full(w.nlat,1/np.sqrt(4*np.pi));dmm=np.zeros(w.nlat)
    offset=0
    for m in range(w.mmax+1):
        if m:
            c=-np.sqrt((2*m+1)/(2*m))
            qmm,dmm=c*s*qmm,c*(x*qmm+s*dmm)
        if not np.any(a[:,offset:offset+w.lmax-m+1]):
            offset+=w.lmax-m+1
            continue
        qprev,dprev=np.zeros(w.nlat),np.zeros(w.nlat)
        q,d=qmm.copy(),dmm.copy()
        for l in range(m,w.lmax+1):
            z=a[:,offset,None]
            if not w.spin:
                spectrum[0,:,m]+=z[0]*q
            elif l:
                ratio=np.divide(q,s,out=np.zeros_like(q),where=np.abs(s)>1e-14)
                if m==1:ratio=np.where(np.abs(s)<=1e-14,d/x,ratio)
                norm=np.sqrt(l*(l+1))
                dt,dp=-d/norm,-1j*m*ratio/norm
                spectrum[0,:,m]+=z[0]*dt-z[1]*dp
                spectrum[1,:,m]+=z[0]*dp+z[1]*dt
            offset+=1
            if l==w.lmax:break
            lp=l+1
            c1=np.sqrt((4*lp*lp-1)/(lp*lp-m*m))
            c2=np.sqrt(((lp-1)**2-m*m)/(4*(lp-1)**2-1)) if lp>1 else 0
            qnext=c1*(x*q-c2*qprev)
            dnext=c1*(-s*q+x*d-c2*dprev)
            qprev,q=q,qnext
            dprev,d=d,dnext
        spectrum[:,:,m]*=np.exp(1j*m*w.phi0)
    return np.fft.irfft(spectrum,n=w.nlon,axis=-1,norm='forward')


def reference_map(w,a):
    """Cache complete reference fields by geometry, coefficients and harness hash."""
    if a.ndim==3:
        return np.stack([reference_map(w,part) for part in a])
    import hashlib
    import json
    import os
    from pathlib import Path
    root = os.environ.get('SHT_BENCH_REFERENCE_CACHE')
    if not root: return _reference_map(w,a)
    description = {k:getattr(w,k) for k in ('grid','nlat','nlon','lmax','mmax','spin','precision','phi0')}
    key = hashlib.sha256(json.dumps(description,sort_keys=True).encode()+a.tobytes()).hexdigest()
    directory = Path(root)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f'{key}.npy'
    digest_path = directory/f'{key}.sha256'
    if path.exists() and digest_path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest_path.read_text().strip():
            raise ValueError('independent reference cache checksum mismatch')
        return np.load(path,mmap_mode='r',allow_pickle=False)
    result = _reference_map(w,a)
    np.save(path,result,allow_pickle=False)
    digest_path.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    return result
