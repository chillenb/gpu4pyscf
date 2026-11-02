# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Truncated and smooth truncated Coulomb (sTC) exchange kernels

exxdiv = 'vcut_sph'        : spherical truncation, PRB 77, 193110
exxdiv = 'vcut_ws'         : Wigner-Seitz truncation, PRB 87, 165122
exxdiv = 'smooth_vcut_sph' : smooth truncation (sphere), arXiv 2609.00203
exxdiv = 'smooth_vcut_ws'  : smooth truncation (Wigner-Seitz), arXiv 2609.00203

The smooth truncated kernel is

    v(G) = v_trunc(G) f(G) + 4pi/G^2 (1 - f(G)),   f(G) = exp(-G^2/(4 omega_stc^2))

with v(0) = v_trunc(0) + pi/omega_stc^2. omega_stc = omega_dot_Rc / Rc, where Rc
is the radius of the truncation sphere or the inradius of the Born-von Karman
supercell. For the long-range (erf(omega r)/r) Coulomb operator, v_trunc is the
truncated long-range kernel and the second term carries exp(-G^2/(4 omega^2)).

The kernels follow pyscf.pbc.tools.pbc.get_coulG. The Wigner-Seitz kernels
reuse the one-time precomputation of pyscf (precompute_exx, precompute_lr_exx),
cached on the DF object as _ws_exx and _ws_lr_exx.
'''

import os
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import asarray

TRUNC_EXXDIV = ('vcut_sph', 'vcut_ws')
STC_EXXDIV = ('smooth_vcut_sph', 'smooth_vcut_ws')

# Default omega_stc*Rc, same as the omega_dot_Rc attribute of the DF classes
OMEGA_DOT_RC = 4.

_XSF_INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

# Spherically truncated long-range Coulomb kernel
#   4pi/q \int_0^Rc erf(omega r) sin(q r) dr
# The complex erfcx is evaluated with the Faddeeva package of scipy/xsf.
_sph_lr_kernel = cp.ElementwiseKernel(
    'float64 absG2, float64 Rc, float64 omega', 'float64 out',
    '''
    const double a = omega * Rc;
    const double erfa = erf(a);
    if (absG2 == 0) {
        out = 2*M_PI * (erfa * (Rc*Rc - .5/(omega*omega))
                        + Rc * exp(-a*a) / (omega * sqrt(M_PI)));
    } else {
        const double q = sqrt(absG2);
        const double b = .5 * q / omega;
        double s, c;
        sincos(q * Rc, &s, &c);
        complex<double> w = Faddeeva::erfcx(complex<double>(a, b));
        double t = exp(-b*b) - exp(-a*a) * (c * w.real() + s * w.imag());
        out = 4*M_PI / absG2 * (t - erfa * c);
    }
    ''',
    'gpu4pyscf_vcut_sph_lr',
    preamble='#include <xsf/gpu4pyscf_compat.h>\n#include <xsf/faddeeva.h>',
    options=('-I' + _XSF_INCLUDE, '--std=c++17'))

def erfcx(z):
    '''Scaled complementary error function exp(z^2) erfc(z) for complex z'''
    return _erfcx_kernel(asarray(z, dtype=np.complex128))

_erfcx_kernel = cp.ElementwiseKernel(
    'complex128 z', 'complex128 out',
    'out = Faddeeva::erfcx(z)',
    'gpu4pyscf_erfcx_complex',
    preamble='#include <xsf/gpu4pyscf_compat.h>\n#include <xsf/faddeeva.h>',
    options=('-I' + _XSF_INCLUDE, '--std=c++17'))

def sph_truncated_lr(absG2, Rc, omega):
    '''Fourier transform of erf(omega r)/r truncated at the sphere of radius Rc'''
    absG2 = asarray(absG2, dtype=np.float64)
    coulG = _sph_lr_kernel(absG2, float(Rc), float(omega))
    # The closed form loses ~eps/(q*Rc)^2 relative accuracy as q -> 0. Use the
    # Taylor expansion of 4pi \int_0^Rc erf(omega r) r sin(q r)/(q r) dr instead.
    small = (absG2 > 0) & (absG2 < (SPH_SERIES_QRC/Rc)**2)
    if small.any():
        coef = _sph_lr_series_coef(Rc, omega)
        x2 = absG2[small]
        val = cp.full_like(x2, coef[-1])
        for c in coef[-2::-1]:
            val *= x2
            val += c
        coulG[small] = val
    return coulG

# Below q*Rc = SPH_SERIES_QRC, sph_truncated_lr uses the Taylor series
SPH_SERIES_QRC = .1

def _sph_lr_series_coef(Rc, omega, nterms=7):
    r'''Coefficients c_n of v(q) = sum_n c_n q^{2n}, where
    c_n = 4pi (-1)^n / (2n+1)! \int_0^Rc erf(omega r) r^{2n+1} dr'''
    from math import factorial
    from scipy.special import erf
    x, w = np.polynomial.legendre.leggauss(200)
    r = .5 * Rc * (x + 1)
    wr = .5 * Rc * w * erf(omega * r)
    return [4*np.pi * (-1)**n / factorial(2*n+1) * wr.dot(r**(2*n+1))
            for n in range(nterms)]

def sph_radius(cell, kpts):
    '''Radius of the sphere with the volume of the Born-von Karman supercell'''
    return (3*len(kpts)*cell.vol/(4*np.pi))**(1./3)

def ws_radius(cell, kpts):
    '''Inradius of the Wigner-Seitz cell of the Born-von Karman supercell'''
    from pyscf.pbc.lo.base import get_kmesh
    from pyscf.pbc.tools.pbc import get_ws_inradius
    kmesh = get_kmesh(cell, kpts)
    return get_ws_inradius(cell.lattice_vectors(), kmesh)

def get_omega_stc(cell, kpts, exxdiv, mf=None):
    '''The Gaussian smoothing exponent omega_stc = omega_dot_Rc / Rc'''
    if exxdiv == 'smooth_vcut_ws':
        Rc = ws_radius(cell, kpts)
    else:
        Rc = sph_radius(cell, kpts)
    omega_dot_Rc = getattr(mf, 'omega_dot_Rc', OMEGA_DOT_RC)
    return omega_dot_Rc / Rc

class STCKernel:
    '''Smooth truncated Coulomb kernel of a k-point mesh, used by the
    range-separated density fitting builder.

    Attributes:
        cell : the unit cell (lattice and precision for the precomputation)
        kpts : the regular k-point mesh of the Born-von Karman supercell
        exxdiv : smooth_vcut_sph or smooth_vcut_ws
        omega_stc : Gaussian smoothing exponent
        mf : object to cache the precomputed Wigner-Seitz kernels
    '''
    def __init__(self, cell, kpts, exxdiv, omega_stc=None, mf=None):
        assert exxdiv in STC_EXXDIV
        self.cell = cell
        self.kpts = np.asarray(kpts).reshape(-1, 3)
        self.exxdiv = exxdiv
        if omega_stc is None:
            omega_stc = get_omega_stc(cell, self.kpts, exxdiv, mf)
        self.omega_stc = float(omega_stc)
        self.mf = mf

    def get_coulG(self, kpt, Gv, mesh, withSR=True):
        '''Full-range smooth truncated kernel at k+G (not weighted)'''
        from gpu4pyscf.pbc.tools.pbc import get_coulG
        return get_coulG(self.cell, k=kpt, exx=self.exxdiv, mf=self.mf, mesh=mesh,
                         Gv=Gv, wrap_around=True, omega=0., kpts=self.kpts,
                         omega_stc=self.omega_stc, withSR=withSR)

def _get_ws_cache(mf, cell, kpts, omega, omega_stc):
    '''Wigner-Seitz kernel data precomputed by pyscf and cached on mf'''
    from pyscf.pbc.tools.pbc import precompute_exx, precompute_lr_exx
    if mf is None:
        raise RuntimeError('Wigner-Seitz truncated kernels require the DF object '
                           '(mf) to cache the precomputed kernel')
    if omega == 0:
        if not getattr(mf, '_ws_exx', None):
            mf._ws_exx = precompute_exx(cell, kpts)
        return mf._ws_exx

    omega = float(omega)
    if omega_stc is not None:
        omega_stc = float(omega_stc)
    cache = getattr(mf, '_ws_lr_exx', None)
    if (not cache or abs(cache['alpha'] - omega) > 1e-9 or
        (cache['omega_stc'] is None) != (omega_stc is None) or
        (omega_stc is not None and abs(cache['omega_stc'] - omega_stc) > 1e-9)):
        mf._ws_lr_exx = cache = precompute_lr_exx(
            cell, kpts, omega=omega, omega_stc=omega_stc)
    return cache

def _ws_vq_lookup(cache, kG):
    '''Index k+G into the precomputed Wigner-Seitz kernel vq. Points outside of
    the precomputed mesh are set to zero.'''
    from gpu4pyscf.pbc.tools.pbc import fftk
    kcell = cache['kcell']
    a = asarray(kcell.lattice_vectors())
    gxyz = kG.dot(a.T) / (2*np.pi)
    shift = (gxyz[0] + .5) % 1 - .5
    gxyz_int = cp.rint(gxyz - shift).astype(np.int64)
    if abs(gxyz - gxyz_int - shift).max() > 1e-6:
        raise RuntimeError('k+G vectors are incompatible with the FFT mesh')
    shift = shift.get()

    if abs(shift).max() < 1e-9:
        if '_vq_gpu' not in cache:
            cache['_vq_gpu'] = asarray(cache['vq'])
        vq = cache['_vq_gpu']
    else:
        key = tuple(np.round(shift, 12))
        gpu_cache = cache.setdefault('_vq_cache_gpu', {})
        if key not in gpu_cache:
            # See pyscf.pbc.tools.pbc.get_coulG for the treatment of the
            # degenerate r_mic on the WS boundary.
            delta = np.dot(shift, kcell.reciprocal_vectors())
            phase = cp.exp(-1j * asarray(cache['r_mic']).dot(asarray(delta)))
            vG = (kcell.vol / len(phase)) * fftk(
                asarray(cache['vR']), kcell.mesh, phase)
            gpu_cache[key] = vG.real.copy()
        vq = gpu_cache[key]

    mesh = cp.asarray(kcell.mesh, dtype=np.int64)
    lower = -(mesh // 2)
    upper = (mesh - 1) // 2
    in_mesh = ((gxyz_int >= lower) & (gxyz_int <= upper)).all(axis=1)
    gxyz = (gxyz_int + mesh) % mesh
    qidx = (gxyz[:,0]*mesh[1] + gxyz[:,1])*mesh[2] + gxyz[:,2]
    return cp.where(in_mesh, vq[qidx * in_mesh], 0.)

def _sr_complement(absG2, omega_stc):
    '''4pi/G^2 (1 - exp(-G^2/(4 omega_stc^2))), the Fourier transform of erfc(omega_stc r)/r'''
    with np.errstate(divide='ignore', invalid='ignore'):
        return -4*np.pi/absG2 * cp.expm1(-.25/omega_stc**2 * absG2)

def get_coulG(cell, kG, absG2, exxdiv, mf=None, kpts=None, omega=0,
              omega_stc=None, withSR=True):
    '''Truncated Coulomb kernels for exxdiv in vcut_sph, vcut_ws,
    smooth_vcut_sph, smooth_vcut_ws

    Args:
        kG, absG2 : k+G vectors and |k+G|^2 (cupy arrays)
        mf : the DF object. It provides omega_dot_Rc and caches the precomputed
            Wigner-Seitz kernels.
        kpts : the regular k-point mesh which defines the Born-von Karman
            supercell.
        omega : 0 for the full-range Coulomb, > 0 for erf(omega r)/r and < 0
            for erfc(|omega| r)/r.
        omega_stc : Gaussian smoothing exponent of the smooth truncation.
            Defaults to mf.omega_dot_Rc/Rc.
        withSR : Whether to include the term 4pi/G^2 (1 - f(G)) for the
            full-range smooth kernels. Range-separated density fitting excludes
            this term at G != 0 since it is evaluated in real space.
    '''
    assert cell.dimension == 3
    assert exxdiv in TRUNC_EXXDIV + STC_EXXDIV
    if kpts is None:
        kpts = np.zeros((1, 3))
    kpts = np.asarray(kpts).reshape(-1, 3)
    if omega < 0:
        return (get_coulG(cell, kG, absG2, exxdiv, mf, kpts, 0, omega_stc, withSR) -
                get_coulG(cell, kG, absG2, exxdiv, mf, kpts, -omega, omega_stc, withSR))

    G0 = absG2 == 0
    smooth = exxdiv in STC_EXXDIV
    if smooth and omega_stc is None:
        omega_stc = get_omega_stc(cell, kpts, exxdiv, mf)

    if exxdiv in ('vcut_sph', 'smooth_vcut_sph'):
        Rc = sph_radius(cell, kpts)
        if omega == 0:
            # 4pi/G^2 (1 - cos(G Rc)) = 8pi/G^2 sin(G Rc/2)^2
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG = 8*np.pi/absG2 * cp.sin(.5*Rc*cp.sqrt(absG2))**2
            coulG[G0] = 2*np.pi*Rc**2
        else:
            coulG = sph_truncated_lr(absG2, Rc, omega)
    else:
        cache = _get_ws_cache(mf, cell, kpts, omega, omega_stc if smooth else None)
        if omega == 0:
            alpha = cache['alpha']
            coulG = _sr_complement(absG2, alpha)
            coulG[G0] = np.pi / alpha**2
        else:
            coulG = cp.zeros_like(absG2)
        coulG += _ws_vq_lookup(cache, kG)

    if smooth:
        f = cp.exp(-.25/omega_stc**2 * absG2)
        v0 = coulG[G0]
        coulG *= f
        if omega == 0:
            if withSR:
                coulG += _sr_complement(absG2, omega_stc)
        else:
            # The long-range Coulomb outside of the smooth truncation
            with np.errstate(divide='ignore', invalid='ignore'):
                coulG -= 4*np.pi/absG2 * cp.exp(-.25/omega**2 * absG2) * cp.expm1(-.25/omega_stc**2 * absG2)
        coulG[G0] = v0 + np.pi / omega_stc**2
    return coulG
