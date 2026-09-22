# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

'''A deliberately simple KRKS driver with density mixing.

Unlike the regular SCF driver, this module does not use Fock damping or DIIS.
The only accelerator is :meth:`SimpleKRKS.mix_density`, which is kept as a
small public hook so that alternative density-mixing formulae are easy to
prototype.
'''

__all__ = ['SimpleKRKS']

import numpy as np
import cupy as cp

from pyscf.scf import chkfile

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray
from gpu4pyscf.pbc.dft import krks


def _get_plain_fock(mf, h1e, s1e, vhf, dm):
    '''Build the physical Fock matrix without Fock mixing or level shifting.'''
    return mf.get_fock(
        h1e=h1e,
        s1e=s1e,
        vhf=vhf,
        dm=dm,
        cycle=-1,
        diis=None,
        level_shift_factor=0.0,
        damp_factor=0.0,
    )


def _norm(value):
    return float(cp.linalg.norm(value).item())


def _validate_density(dm, reference):
    dm = asarray(dm)
    if dm.shape != reference.shape:
        raise ValueError(
            'mix_density returned shape %s; expected %s'
            % (dm.shape, reference.shape)
        )
    if not bool(cp.all(cp.isfinite(dm)).item()):
        raise FloatingPointError('mix_density returned a nonfinite density')
    return dm


def _kernel(
    mf,
    conv_tol=None,
    conv_tol_grad=None,
    dump_chk=True,
    dm0=None,
    callback=None,
    conv_check=True,
    **kwargs,
):
    '''Run fixed-point KRKS iterations with explicit density mixing.'''
    if conv_tol is None:
        conv_tol = mf.conv_tol
    if conv_tol_grad is None:
        conv_tol_grad = conv_tol**0.5

    log = logger.new_logger(mf, mf.verbose)
    log.info('Set gradient and density-residual thresholds to %g',
             conv_tol_grad)
    t0 = t1 = log.init_timer()
    cell = mf.cell

    if dm0 is None:
        dm0 = mf.get_init_guess(cell, mf.init_guess)
        t1 = log.timer_debug1('generating initial guess', *t1)
    dm = cp.asarray(dm0, order='C')

    h1e = cp.asarray(mf.get_hcore())
    s1e = cp.asarray(mf.get_ovlp())
    x_orth = mf.check_linear_dependency(s1e, log)
    t1 = log.timer_debug1('hcore and overlap', *t1)

    vhf = mf.get_veff(cell, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)
    log.timer('SCF initialization', *t0)

    mf.converged = False
    mf.cycles = 0
    mf.dm = dm
    mf.dm_output = None
    mf.density_residual = None
    mf.norm_density_residual = None
    mf.norm_density_step = None

    # Compute orbitals but do not update the density when iterations are
    # explicitly disabled.
    if mf.max_cycle <= 0:
        fock = _get_plain_fock(mf, h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        mf.dm_output = asarray(mf.make_rdm1(mo_coeff, mo_occ))
        return False, e_tot, mo_energy, mo_coeff, mo_occ

    dump_chk = bool(dump_chk and mf.chkfile)
    if dump_chk:
        chkfile.save_mol(cell, mf.chkfile)

    mo_energy = mo_coeff = mo_occ = None
    for cycle in range(mf.max_cycle):
        cycle_t0 = log.init_timer()
        dm_last = dm
        vhf_last = vhf
        last_e = e_tot

        # F[D_in] -> orbitals -> D_out
        fock_in = _get_plain_fock(mf, h1e, s1e, vhf_last, dm_last)
        mo_energy, mo_coeff = mf.eig(fock_in, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_output = asarray(mf.make_rdm1(mo_coeff, mo_occ))
        density_residual = dm_output - dm_last

        # D_next = M(D_in, D_out).  This is the only mixing operation in the
        # driver and the intended experimentation point.
        dm = _validate_density(
            mf.mix_density(dm_last, dm_output, cycle), dm_last)
        vhf = mf.get_veff(cell, dm, dm_last, vhf_last)

        # Evaluate convergence against the physical Fock matrix at D_next.
        fock = _get_plain_fock(mf, h1e, s1e, vhf, dm)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        norm_gorb = _norm(mf.get_grad(mo_coeff, mo_occ, fock))
        norm_density_residual = _norm(density_residual)
        norm_density_step = _norm(dm - dm_last)
        e_diff = abs(e_tot - last_e)

        mf.dm = dm
        mf.dm_output = dm_output
        mf.density_residual = density_residual
        mf.norm_density_residual = norm_density_residual
        mf.norm_density_step = norm_density_step

        log.timer('cycle=%d' % (cycle + 1), *cycle_t0)
        log.info(
            'cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  '
            '|Dout-Din|= %4.3g  |Dnext-Din|= %4.3g',
            cycle + 1,
            e_tot,
            e_tot - last_e,
            norm_gorb,
            norm_density_residual,
            norm_density_step,
        )

        if dump_chk:
            mf.dump_chk(locals())
        if callable(callback):
            callback(locals())

        if callable(mf.check_convergence):
            mf.converged = bool(mf.check_convergence(locals()))
        else:
            mf.converged = (
                e_diff < conv_tol
                and norm_gorb < conv_tol_grad
                and norm_density_residual < conv_tol_grad
            )
        if mf.converged:
            break
    else:
        log.warn('SCF failed to converge')

    mf.cycles = cycle + 1
    return mf.converged, e_tot, mo_energy, mo_coeff, mo_occ


def scf(mf, dm0=None, **kwargs):
    '''Run the density-mixed SCF calculation and return the total energy.'''
    cput0 = logger.init_timer(mf)
    mf.dump_flags()
    mf.build(mf.cell)

    if dm0 is None:
        if mf.dm is not None:
            dm0 = mf.dm
        elif mf.mo_coeff is not None and mf.mo_occ is not None:
            dm0 = mf.make_rdm1()

    result = _kernel(
        mf,
        conv_tol=mf.conv_tol,
        conv_tol_grad=mf.conv_tol_grad,
        dm0=dm0,
        callback=mf.callback,
        conv_check=mf.conv_check,
        **kwargs,
    )
    if mf.max_cycle > 0 or mf.mo_coeff is None:
        (
            mf.converged,
            mf.e_tot,
            mf.mo_energy,
            mf.mo_coeff,
            mf.mo_occ,
        ) = result
    else:
        mf.e_tot = result[1]

    logger.timer(mf, 'SCF', *cput0)
    mf._finalize()
    return mf.e_tot


class SimpleKRKS(krks.KRKS):
    '''KRKS with a transparent, density-mixed fixed-point iteration.

    ``mixing_factor`` is the fraction of the newly diagonalized density used
    by the default linear mixer:

    .. math::

        D_{n+1} = D_n + \\alpha (D_{\\mathrm{out}} - D_n).

    Override :meth:`mix_density` to try a different formula.  ``dm_old`` and
    ``dm_output`` are CuPy arrays with shape ``(nkpts, nao, nao)`` and
    ``cycle`` is zero based.
    '''

    diis = False
    mixing_factor = 0.5

    _keys = {
        'mixing_factor',
        'dm',
        'dm_output',
        'density_residual',
        'norm_density_residual',
        'norm_density_step',
    }

    def __init__(
        self,
        cell,
        kpts=None,
        xc='LDA,VWN',
        exxdiv='ewald',
        mixing_factor=0.5,
    ):
        super().__init__(cell, kpts=kpts, xc=xc, exxdiv=exxdiv)
        self.diis = False
        self.mixing_factor = float(mixing_factor)
        self.dm = None
        self.dm_output = None
        self.density_residual = None
        self.norm_density_residual = None
        self.norm_density_step = None

    def check_sanity(self):
        super().check_sanity()
        if (
            not np.isfinite(self.mixing_factor)
            or self.mixing_factor <= 0.0
            or self.mixing_factor > 1.0
        ):
            raise ValueError('mixing_factor must be in the interval (0, 1]')
        return self

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'density mixing factor = %g', self.mixing_factor)
        return self

    def mix_density(self, dm_old, dm_output, cycle=None):
        '''Return the density used to build the next Fock matrix.

        Subclasses can override this method to implement a nonlinear or
        history-dependent mixer.  The default is linear density mixing.
        '''
        if (
            not np.isfinite(self.mixing_factor)
            or self.mixing_factor <= 0.0
            or self.mixing_factor > 1.0
        ):
            raise ValueError('mixing_factor must be in the interval (0, 1]')
        return dm_old + self.mixing_factor * (dm_output - dm_old)

    def reset(self, cell=None):
        super().reset(cell)
        self.dm = None
        self.dm_output = None
        self.density_residual = None
        self.norm_density_residual = None
        self.norm_density_step = None
        return self

    kernel = scf = scf
