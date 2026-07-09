#!/usr/bin/env python
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

from types import SimpleNamespace

import cupy
import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints
from pyscf.scf.smearing import _fermi_smearing_occ, _smearing_optimize
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.scf import khf
from gpu4pyscf.pbc.scf.smearing import _partition_occ
from gpu4pyscf.scf.gcscf import (
    GCSCF_SIGMA,
    GCSCF_STEP,
    GCSCF_MIN_STEP,
    GCSCF_LINE_MAX_CYCLE,
    _as_scalar_mu,
    _eigh_tensor,
    _fermi_entropy_cpu,
    _fermi_occupation_derivative,
    _hermitian_part,
    _line_minimize,
    _make_diagonal,
    _matrix_rotation,
    _norm,
    _smearing_matrix_gradient,
    _vdot,
)


def gcscf(mf, sigma=None, mu0=None, fix_spin=False):
    '''Finite-temperature GPU k-point SCF via auxiliary-Hamiltonian minimization.

    Args:
        mf : a GPU KRHF, KRKS, KUHF, or KUKS object
            Periodic k-point mean-field object to decorate.
        sigma : float
            Electronic temperature in Hartree.
        mu0 : float or None
            Fixed chemical potential in Hartree.  If None, the electron number
            is fixed to ``cell.tot_electrons(nkpts)``.
        fix_spin : bool
            For KUHF/KUKS, whether to use separate alpha and beta chemical
            potentials to preserve ``mf.nelec`` when ``mu0`` is None.
    '''
    if isinstance(mf, _GCKSCF):
        if sigma is not None:
            mf.sigma = sigma
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    if not isinstance(mf, khf.KSCF):
        raise NotImplementedError('GPU GC-KSCF requires a k-point SCF object')
    if mf.istype('_CIAH_SOSCF'):
        raise NotImplementedError('GPU GC-KSCF with second order SCF is not '
                                  'supported')
    if isinstance(mf.kpts, KPoints):
        raise NotImplementedError('GPU GC-KSCF with symmetry-adapted KPoints '
                                  'is not supported yet')
    if mf.istype('KROHF') or mf.istype('KGHF'):
        raise NotImplementedError('GPU GC-KSCF currently supports KRHF/KRKS '
                                  'and KUHF/KUKS objects')
    if not (mf.istype('KRHF') or mf.istype('KUHF')):
        raise NotImplementedError('GPU GC-KSCF currently supports KRHF/KRKS '
                                  'and KUHF/KUKS objects')

    return lib.set_class(_GCKSCF(mf, sigma, mu0, fix_spin),
                         (_GCKSCF, mf.__class__))


def gcscf_(mf, *args, **kwargs):
    mf1 = gcscf(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


grand_canonical = gcscf
grand_canonical_ = gcscf_


def remove_gcscf(mf):
    '''Remove the GPU GC-KSCF decorator.'''
    return mf.undo_gcscf()


class _GCKSCF:
    '''Finite-temperature GPU k-point SCF via auxiliary-Hamiltonian minimization.'''

    __name_mixin__ = 'GC'

    _keys = {
        'sigma', 'mu0', 'fix_spin', 'auxh_step', 'auxh_min_step',
        'auxh_line_max_cycle', 'entropy', 'e_free', 'e_zero', 'e_grand',
        'mu', 'nelectron', 'haux', 'n_haux_eval', 'auxh_residual_norm',
    }

    def __init__(self, mf, sigma, mu0, fix_spin):
        self.__dict__.update(mf.__dict__)
        self.sigma = GCSCF_SIGMA if sigma is None else sigma
        self.mu0 = mu0
        self.fix_spin = fix_spin
        self.auxh_step = GCSCF_STEP
        self.auxh_min_step = GCSCF_MIN_STEP
        self.auxh_line_max_cycle = GCSCF_LINE_MAX_CYCLE
        self.entropy = None
        self.e_free = None
        self.e_zero = None
        self.e_grand = None
        self.mu = None
        self.nelectron = None
        self.haux = None
        self.n_haux_eval = 0
        self.auxh_residual_norm = None

    def undo_gcscf(self):
        '''Remove the GPU GC-KSCF mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _GCKSCF))
        del obj.sigma
        del obj.mu0
        del obj.fix_spin
        del obj.auxh_step
        del obj.auxh_min_step
        del obj.auxh_line_max_cycle
        del obj.entropy
        del obj.e_free
        del obj.e_zero
        del obj.e_grand
        del obj.mu
        del obj.nelectron
        del obj.haux
        del obj.n_haux_eval
        del obj.auxh_residual_norm
        return obj

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        super().dump_flags(verbose)
        log.info('******** GPU GC-KSCF flags ********')
        log.info('sigma = %s', self.sigma)
        if self.mu0 is None:
            log.info('electron number fixed to cell.tot_electrons(nkpts) = %s',
                     self.cell.tot_electrons(len(self.kpts)))
        else:
            log.info('mu0 = %s', self.mu0)
        log.info('fix_spin = %s', self.fix_spin)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('auxh_step = %g', self.auxh_step)
        log.info('auxh_min_step = %g', self.auxh_min_step)
        log.info('auxh_line_max_cycle = %d', self.auxh_line_max_cycle)
        return self

    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        if self.sigma is None:
            return super().get_occ(mo_energy_kpts, mo_coeff_kpts)
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GPU GC-KSCF')
        if isinstance(self.kpts, KPoints):
            raise NotImplementedError('GPU GC-KSCF get_occ with '
                                      'symmetry-adapted KPoints is not '
                                      'supported yet')

        mo_energy = cupy.asarray(mo_energy_kpts, dtype=float)
        self.mu, mo_occ, self.entropy = _occupations(
            self, mo_energy, len(self.kpts))
        logger.info(self, '    sigma = %g  Optimized mu = %s  '
                    'entropy = %.12g', self.sigma, self.mu, self.entropy)
        return mo_occ

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = super().energy_tot(dm, h1e, vhf)
        if self.sigma is not None and self.mo_occ is not None:
            nkpts = len(self.kpts)
            mo_occ = cupy.asarray(self.mo_occ)
            spin_degeneracy = 1.0 if self.istype('KUHF') else 2.0
            self.entropy = (_fermi_entropy_cpu(cupy.asnumpy(mo_occ),
                                               spin_degeneracy) / nkpts)
            self.nelectron = float(cupy.sum(mo_occ).get() / nkpts)
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            mu = np.asarray(self.mu0 if self.mu0 is not None else self.mu)
            self.e_grand = _grand_potential(self.e_free, mu, mo_occ, nkpts)
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  '
                        'Grand potential = %.15g',
                        e_tot, self.e_free, self.e_grand)
        return e_tot

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if self.sigma is None:
            raise ValueError('sigma must be specified for GPU GC-KSCF')
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GPU GC-KSCF')
        if isinstance(self.kpts, KPoints):
            raise NotImplementedError('GPU GC-KSCF with symmetry-adapted '
                                      'KPoints is not supported yet')

        conv_tol = kwargs.pop('conv_tol', self.conv_tol)
        conv_tol_grad = kwargs.pop('conv_tol_grad', self.conv_tol_grad)
        if conv_tol_grad is None:
            conv_tol_grad = conv_tol ** .5
            logger.info(self, 'Set gradient conv threshold to %g',
                        conv_tol_grad)
        max_cycle = kwargs.pop('max_cycle', self.max_cycle)
        callback = kwargs.pop('callback', self.callback)
        if kwargs:
            logger.warn(self, 'GPU GC-KSCF kernel ignored unsupported '
                        'kwargs %s', sorted(kwargs))

        self.build(self.cell)
        self.dump_flags()
        self.pre_kernel(locals())

        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)
        is_uhf = self.istype('KUHF')
        hcore_ao = cupy.asarray(self.get_hcore(cell, kpts))
        s_kpts = cupy.asarray(self.get_ovlp(cell, kpts))
        x_kpts = self.check_linear_dependency(s_kpts, verbose=self.verbose)
        invalid_energy = max(100.0, 1000.0 * self.sigma)
        hcore = _orthogonalize_kpts(
            hcore_ao, x_kpts, padding=invalid_energy)
        n_haux_eval = 0

        def haux_eval_counting(mo_energy, mo_coeff_orth):
            nonlocal n_haux_eval
            n_haux_eval += 1
            return _haux_eval(self, mo_energy, mo_coeff_orth, x_kpts,
                              hcore_ao, hcore)

        if isinstance(dm0, str):
            dm0 = self.from_chk(dm0)
        elif dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            dm0 = self.make_rdm1(self.mo_coeff, self.mo_occ)
        elif dm0 is None:
            dm0 = self.get_init_guess(cell, key=self.init_guess)
        dm0 = cupy.asarray(dm0)

        vhf_ao = self.get_veff(cell, dm0)
        if is_uhf:
            vhf = cupy.stack([
                _orthogonalize_kpts(vhf_ao[s], x_kpts) for s in range(2)
            ])
            haux = (hcore[None,:,:,:] + vhf).reshape(2 * nkpts,
                                                     hcore.shape[-1],
                                                     hcore.shape[-1])
        else:
            haux = hcore + _orthogonalize_kpts(vhf_ao, x_kpts)
        haux = _hermitian_part(haux)

        mo_energy, mo_coeff_orth = _eigh_tensor(haux)
        state = haux_eval_counting(mo_energy, mo_coeff_orth)
        residual_norm = _norm(state.haux_gradient)
        alpha_t = float(self.auxh_step)
        previous_gradient = None
        previous_direction = None
        previous_gknorm = None
        niter_done = 0
        converged = residual_norm < conv_tol_grad

        log = logger.new_logger(self)
        if self.mu0 is None:
            log.info('GPU GC-KSCF auxiliary-Hamiltonian minimization, '
                     'sigma = %.12g Ha, fixed nelectron/cell = %.12g, '
                     'nkpts = %d', self.sigma, cell.nelectron, nkpts)
        else:
            log.info('GPU GC-KSCF auxiliary-Hamiltonian minimization, '
                     'sigma = %.12g Ha, mu0 = %s Ha, nkpts = %d',
                     self.sigma, self.mu0, nkpts)

        for cycle in range(1, max_cycle + 1):
            if converged:
                break

            gradient = state.haux_gradient
            steepest_direction = _hermitian_part(
                state.hsub - _make_diagonal(state.mo_energy))
            gknorm = -_vdot(gradient, steepest_direction)
            beta = 0.0
            cg_reset = False
            if (previous_gradient is not None and previous_direction is not None
                    and previous_gknorm is not None and previous_gknorm > 0.0):
                dot_gprev_kg = -_vdot(previous_gradient, steepest_direction)
                beta = (gknorm - dot_gprev_kg) / previous_gknorm
                if beta < 0.0 or not np.isfinite(beta):
                    beta = 0.0
                    cg_reset = True

            if previous_direction is None:
                direction = steepest_direction
            else:
                direction = _hermitian_part(steepest_direction
                                            + beta * previous_direction)
            gdotd = _vdot(gradient, direction)
            if gdotd >= 0.0 and beta != 0.0:
                beta = 0.0
                cg_reset = True
                direction = steepest_direction
                gdotd = _vdot(gradient, direction)
            if gdotd >= 0.0:
                beta = 0.0
                cg_reset = True
                direction = _hermitian_part(-gradient)

            line_eval_start = n_haux_eval
            line_result = _line_minimize(
                haux_eval_counting, haux, direction, state,
                alpha_t=alpha_t, alpha_t_min=self.auxh_min_step,
                max_cycle=self.auxh_line_max_cycle)
            if not line_result.success and beta != 0.0:
                beta = 0.0
                cg_reset = True
                direction = steepest_direction
                line_result = _line_minimize(
                    haux_eval_counting, haux, direction, state,
                    alpha_t=alpha_t, alpha_t_min=self.auxh_min_step,
                    max_cycle=self.auxh_line_max_cycle)
            line_haux_eval = n_haux_eval - line_eval_start
            if not line_result.success:
                if residual_norm < conv_tol_grad:
                    converged = True
                    log.info('GPU GC-KSCF line minimization stopped at cycle '
                             '%d with converged residual: %s',
                             cycle, line_result.message)
                    break
                log.info('GPU GC-KSCF line minimization failed at cycle %d '
                         'after %d iterations: %s',
                         cycle, line_result.niter, line_result.message)
                break

            old_objective = state.objective
            haux = line_result.haux
            state = line_result.state
            residual_norm = line_result.residual_norm
            alpha_t = line_result.next_alpha_t
            if alpha_t < self.auxh_min_step:
                alpha_t = float(self.auxh_step)
            rot = line_result.rotation
            previous_gradient = _hermitian_part(
                _matrix_rotation(gradient, rot))
            previous_direction = _hermitian_part(
                _matrix_rotation(direction, rot))
            previous_gknorm = gknorm
            d_objective = state.objective - old_objective
            niter_done = cycle

            log.info('cycle=%3d E=%.15g A=%.15g Omega=%.15g N=%.9g '
                     'S=%.9g |R|=%.3e alpha=%.3g alphaT=%.3g beta=%.3g '
                     'reset=%d line_iter=%d evals=%d line_haux_eval=%d',
                     cycle, state.e_tot, state.e_free, state.e_grand,
                     state.nelectron, state.entropy, residual_norm,
                     line_result.alpha, alpha_t, beta, int(cg_reset),
                     line_result.niter, n_haux_eval, line_haux_eval)

            if callable(callback):
                callback(locals())

            if abs(d_objective) < conv_tol and residual_norm < conv_tol_grad:
                converged = True
                break

        self.converged = converged
        self.e_tot = float(state.e_tot)
        self.e_free = float(state.e_free)
        self.e_zero = float(state.e_tot - self.sigma * state.entropy * .5)
        self.e_grand = float(state.e_grand)
        mu = np.asarray(state.mu)
        self.mu = float(mu) if mu.ndim == 0 else np.array(mu, copy=True)
        self.entropy = float(state.entropy)
        self.nelectron = float(state.nelectron)
        self.mo_energy = cupy.array(state.mo_energy_kpts, copy=True)
        self.mo_coeff = cupy.array(state.mo_coeff, copy=True)
        self.mo_occ = cupy.array(state.mo_occ, copy=True)
        self.haux = cupy.array(
            _haux_orth_to_ao(haux, s_kpts, x_kpts, is_uhf), copy=True)
        self.n_haux_eval = n_haux_eval
        self.auxh_residual_norm = float(residual_norm)
        self.cycles = niter_done

        if self.chkfile:
            self.dump_chk({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                           'mo_coeff': self.mo_coeff, 'mo_occ': self.mo_occ})
        self.post_kernel(locals())
        logger.timer(self, 'GPU GC-KSCF', *cput0)
        self._finalize()
        return self.e_tot

    scf = kernel

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged GPU GC-KSCF energy = %.15g',
                        self.e_tot)
        else:
            logger.note(self, 'GPU GC-KSCF not converged.')
            logger.note(self, 'GPU GC-KSCF energy = %.15g', self.e_tot)
        logger.note(self, 'GPU GC-KSCF free energy = %.15g', self.e_free)
        logger.note(self, 'GPU GC-KSCF grand potential = %.15g',
                    self.e_grand)
        logger.note(self, 'GPU GC-KSCF electron number/cell = %.15g',
                    self.nelectron)
        logger.note(self, 'GPU GC-KSCF entropy = %.15g', self.entropy)
        logger.note(self, 'GPU GC-KSCF AuxH evaluations = %d',
                    self.n_haux_eval)
        logger.note(self, 'GPU GC-KSCF AuxH residual norm = %.3e',
                    self.auxh_residual_norm)
        return self

    def to_cpu(self):
        from pyscf.pbc.scf import gcscf as cpu_gcscf
        obj = cpu_gcscf.gcscf(self.undo_gcscf().to_cpu(), self.sigma,
                              self.mu0, self.fix_spin)
        obj.conv_tol_grad = self.conv_tol_grad
        obj.auxh_step = self.auxh_step
        obj.auxh_min_step = self.auxh_min_step
        obj.auxh_line_max_cycle = self.auxh_line_max_cycle
        return obj


def _haux_eval(mf, mo_energy, mo_coeff_orth, x_kpts, hcore_ao, hcore):
    mo_energy = cupy.asarray(mo_energy, dtype=float)
    mo_coeff_orth = cupy.asarray(mo_coeff_orth)
    nkpts = len(x_kpts)
    is_uhf = mf.istype('KUHF')
    mu, mo_occ, entropy = _occupations(mf, mo_energy, nkpts)

    if is_uhf:
        nmo = mo_energy.shape[-1]
        mo_energy_kpts = mo_energy.reshape(2, nkpts, nmo)
        mo_coeff_orth_kpts = mo_coeff_orth.reshape(2, nkpts, nmo, nmo)
        mo_coeff = cupy.stack([
            cupy.stack([
                x_kpts[k] @ mo_coeff_orth_kpts[s,k,:x_kpts[k].shape[1]]
                for k in range(nkpts)
            ])
            for s in range(2)
        ])
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mf.cell, dm)
        fock_kpts = hcore[None,:,:,:] + cupy.stack([
            _orthogonalize_kpts(vhf[s], x_kpts) for s in range(2)
        ])
        fock = fock_kpts.reshape(2 * nkpts, nmo, nmo)
        hsub = _matrix_rotation(fock, mo_coeff_orth)
        grad_filling = _hermitian_part(hsub) - _make_diagonal(mo_energy)
        if mf.mu0 is None:
            occ_blocks = mo_occ.reshape(2 * nkpts, nmo)
            occ_prime = _fermi_occupation_derivative(occ_blocks, mf.sigma, 1.0)
            diag_grad = cupy.diagonal(grad_filling, axis1=-2,
                                      axis2=-1).real
            if mf.fix_spin:
                occ_prime_s = occ_prime.reshape(2, nkpts, nmo)
                diag_grad_s = diag_grad.reshape(2, nkpts, nmo)
                dmu = cupy.zeros(2)
                for s in range(2):
                    denom = float(cupy.sum(occ_prime_s[s]).get())
                    if abs(denom) > np.finfo(float).tiny:
                        dmu[s] = (cupy.sum(occ_prime_s[s] * diag_grad_s[s])
                                  / denom)
                grad_filling = grad_filling - _make_diagonal(
                    cupy.repeat(dmu, nkpts * nmo).reshape(2 * nkpts, nmo))
            else:
                denom = float(cupy.sum(occ_prime).get())
                if abs(denom) > np.finfo(float).tiny:
                    dmu = float(cupy.sum(occ_prime * diag_grad).get()
                                / denom)
                    grad_filling = grad_filling - _make_diagonal(
                        cupy.full_like(mo_energy, dmu))
        haux_gradient = (_smearing_matrix_gradient(
            mo_energy, mo_occ.reshape(2 * nkpts, nmo), mf.sigma,
            grad_filling, 1.0) / nkpts)

    else:
        mo_energy_kpts = mo_energy
        mo_coeff = cupy.stack([
            x_kpts[k] @ mo_coeff_orth[k,:x_kpts[k].shape[1]]
            for k in range(nkpts)
        ])
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mf.cell, dm)
        fock = hcore + _orthogonalize_kpts(vhf, x_kpts)
        hsub = _matrix_rotation(fock, mo_coeff_orth)
        grad_filling = _hermitian_part(hsub) - _make_diagonal(mo_energy)
        if mf.mu0 is None:
            occ_prime = _fermi_occupation_derivative(mo_occ, mf.sigma, 2.0)
            diag_grad = cupy.diagonal(grad_filling, axis1=-2,
                                      axis2=-1).real
            denom = float(cupy.sum(occ_prime).get())
            if abs(denom) > np.finfo(float).tiny:
                dmu = float(cupy.sum(occ_prime * diag_grad).get() / denom)
                grad_filling = grad_filling - _make_diagonal(
                    cupy.full_like(mo_energy, dmu))
        haux_gradient = (_smearing_matrix_gradient(
            mo_energy, mo_occ, mf.sigma, grad_filling, 2.0) / nkpts)

    e_tot = float(mf.energy_elec(dm, hcore_ao, vhf)[0] + mf.energy_nuc())
    e_free = e_tot - mf.sigma * entropy
    nelectron = float(cupy.sum(mo_occ).get() / nkpts)
    e_grand = _grand_potential(e_free, mu, mo_occ, nkpts)

    return SimpleNamespace(
        mo_energy=mo_energy, mo_energy_kpts=mo_energy_kpts,
        mo_coeff=mo_coeff, mo_coeff_orth=mo_coeff_orth, mo_occ=mo_occ,
        dm=dm, vhf=vhf, fock=fock, hsub=hsub, entropy=entropy,
        nelectron=nelectron, mu=mu, e_tot=e_tot, e_free=e_free,
        e_grand=e_grand, objective=e_free if mf.mu0 is None else e_grand,
        haux_gradient=haux_gradient)


def _occupations(mf, mo_energy, nkpts):
    mo_energy_cpu = cupy.asnumpy(cupy.asarray(mo_energy))
    if mf.istype('KUHF'):
        nmo = mo_energy_cpu.shape[-1]
        mo_energy_kpts = mo_energy_cpu.reshape(2, nkpts, nmo)
        if mf.fix_spin:
            if mf.mu0 is None:
                mu = np.empty(2)
                mo_occ_cpu = np.empty_like(mo_energy_kpts)
                for s in range(2):
                    mu[s], occ = _smearing_optimize(
                        _fermi_smearing_occ, mo_energy_kpts[s].reshape(-1),
                        mf.nelec[s], mf.sigma)
                    mu[s] = _as_scalar_mu(mu[s])
                    mo_occ_cpu[s] = np.asarray(
                        _partition_occ(occ, mo_energy_kpts[s]))
            else:
                if np.isscalar(mf.mu0):
                    mu = np.array([float(mf.mu0), float(mf.mu0)])
                elif len(mf.mu0) == 2:
                    mu = np.asarray(mf.mu0, dtype=float)
                else:
                    raise TypeError(f'Unsupported mu0: {mf.mu0}')
                mo_occ_cpu = np.asarray([
                    np.asarray(_partition_occ(
                        _fermi_smearing_occ(mu[s],
                                            mo_energy_kpts[s].reshape(-1),
                                            mf.sigma),
                        mo_energy_kpts[s]))
                    for s in range(2)
                ])
        else:
            flat_energy = mo_energy_cpu.reshape(-1)
            if mf.mu0 is None:
                mu, flat_occ = _smearing_optimize(
                    _fermi_smearing_occ, flat_energy,
                    mf.cell.tot_electrons(nkpts), mf.sigma)
                mu = _as_scalar_mu(mu)
            else:
                if not np.isscalar(mf.mu0):
                    raise TypeError(f'Unsupported mu0: {mf.mu0}')
                mu = float(mf.mu0)
                flat_occ = _fermi_smearing_occ(mu, flat_energy, mf.sigma)
            mo_occ_cpu = flat_occ.reshape(2, nkpts, nmo)
        entropy = _fermi_entropy_cpu(mo_occ_cpu, 1.0) / nkpts
        return mu, cupy.asarray(mo_occ_cpu), entropy

    flat_energy = mo_energy_cpu.reshape(-1)
    if mf.mu0 is None:
        nocc = (mf.cell.tot_electrons(nkpts) + 1) // 2
        mu, flat_occ = _smearing_optimize(
            _fermi_smearing_occ, flat_energy, nocc, mf.sigma)
        mu = _as_scalar_mu(mu)
    else:
        if not np.isscalar(mf.mu0):
            raise TypeError(f'Unsupported mu0: {mf.mu0}')
        mu = float(mf.mu0)
        flat_occ = _fermi_smearing_occ(mu, flat_energy, mf.sigma)
    mo_occ_cpu = 2.0 * np.asarray(_partition_occ(flat_occ, mo_energy_cpu))
    entropy = _fermi_entropy_cpu(mo_occ_cpu, 2.0) / nkpts
    return mu, cupy.asarray(mo_occ_cpu), entropy


def _grand_potential(e_free, mu, mo_occ, nkpts):
    mu = np.asarray(mu)
    if mu.ndim == 0:
        return float(e_free - float(mu) * cupy.sum(mo_occ).get() / nkpts)
    nelec = cupy.asnumpy(cupy.sum(mo_occ, axis=(1, 2))) / nkpts
    return float(e_free - mu @ nelec)


def _orthogonalize_kpts(matrix_kpts, x_kpts, padding=0.0):
    nmo = max(x.shape[1] for x in x_kpts)
    out = cupy.zeros((len(x_kpts), nmo, nmo),
                     dtype=cupy.asarray(matrix_kpts).dtype)
    for k, x in enumerate(x_kpts):
        n = x.shape[1]
        out[k,:n,:n] = x.conj().T @ matrix_kpts[k] @ x
        if n < nmo and padding != 0.0:
            idx = cupy.arange(n, nmo)
            out[k,idx,idx] = padding
    return out


def _haux_orth_to_ao(haux, s_kpts, x_kpts, is_uhf):
    nkpts = len(x_kpts)
    if is_uhf:
        nmo = haux.shape[-1]
        haux = haux.reshape(2, nkpts, nmo, nmo)
        haux_ao = cupy.stack([
            cupy.stack([
                s_kpts[k] @ x_kpts[k] @ haux[s,k,:x_kpts[k].shape[1],
                                             :x_kpts[k].shape[1]]
                @ x_kpts[k].conj().T @ s_kpts[k]
                for k in range(nkpts)
            ])
            for s in range(2)
        ])
        return (haux_ao + haux_ao.conj().swapaxes(-1, -2)) * .5
    return _hermitian_part(cupy.stack([
        s_kpts[k] @ x_kpts[k] @ haux[k,:x_kpts[k].shape[1],
                                      :x_kpts[k].shape[1]]
        @ x_kpts[k].conj().T @ s_kpts[k]
        for k in range(nkpts)
    ]))


def from_cpu(method):
    from pyscf.pbc.scf import gcscf as cpu_gcscf
    assert isinstance(method, cpu_gcscf._GCKSCF)
    return gcscf(method.undo_gcscf().to_gpu(), method.sigma, method.mu0,
                 method.fix_spin)
