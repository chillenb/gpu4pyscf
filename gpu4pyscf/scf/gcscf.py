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
from pyscf import __config__, lib
from pyscf.scf.smearing import _fermi_smearing_occ, _smearing_optimize
from gpu4pyscf.lib import logger


GCSCF_SIGMA = getattr(__config__, 'scf_gcscf_sigma', None)
GCSCF_STEP = getattr(__config__, 'scf_gcscf_step', 1.0)
GCSCF_MIN_STEP = getattr(__config__, 'scf_gcscf_min_step', 1e-8)
GCSCF_LINE_MAX_CYCLE = getattr(__config__, 'scf_gcscf_line_max_cycle', 10)


def gcscf(mf, sigma=None, mu0=None, fix_spin=False):
    '''Finite-temperature GPU SCF via auxiliary-Hamiltonian minimization.

    Args:
        mf : a GPU RHF, RKS, UHF, or UKS object
            Molecular mean-field object to decorate.
        sigma : float
            Electronic temperature in Hartree.
        mu0 : float or None
            Fixed chemical potential in Hartree.  If None, the electron number
            is fixed to ``mol.nelectron``.
        fix_spin : bool
            For UHF/UKS, whether to use separate alpha and beta chemical
            potentials to preserve ``mf.nelec`` when ``mu0`` is None.
    '''
    if isinstance(mf, _GCSCF):
        if sigma is not None:
            mf.sigma = sigma
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    if not mf.__module__.startswith('gpu4pyscf'):
        raise NotImplementedError('GPU GC-SCF requires a gpu4pyscf SCF object')
    if mf.istype('_CIAH_SOSCF'):
        raise NotImplementedError('GPU GC-SCF with second order SCF is not '
                                  'supported')
    if mf.istype('ROHF') or mf.istype('GHF'):
        raise NotImplementedError('GPU GC-SCF currently supports RHF/RKS and '
                                  'UHF/UKS objects')
    if not (mf.istype('RHF') or mf.istype('UHF')):
        raise NotImplementedError('GPU GC-SCF currently supports RHF/RKS and '
                                  'UHF/UKS objects')

    return lib.set_class(_GCSCF(mf, sigma, mu0, fix_spin),
                         (_GCSCF, mf.__class__))


def gcscf_(mf, *args, **kwargs):
    mf1 = gcscf(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf


grand_canonical = gcscf
grand_canonical_ = gcscf_


def remove_gcscf(mf):
    '''Remove the GPU GC-SCF decorator.'''
    return mf.undo_gcscf()


class _GCSCF:
    '''Finite-temperature GPU SCF via auxiliary-Hamiltonian minimization.'''

    __name_mixin__ = 'GC-SCF'

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
        self.conv_tol_grad = None
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
        '''Remove the GPU GC-SCF mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _GCSCF))
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
        log.info('******** GPU GC-SCF flags ********')
        log.info('sigma = %s', self.sigma)
        if self.mu0 is None:
            log.info('electron number fixed to mol.nelectron = %s',
                     self.mol.nelectron)
        else:
            log.info('mu0 = %s', self.mu0)
        log.info('fix_spin = %s', self.fix_spin)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('auxh_step = %g', self.auxh_step)
        log.info('auxh_min_step = %g', self.auxh_min_step)
        log.info('auxh_line_max_cycle = %d', self.auxh_line_max_cycle)
        return self

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if self.sigma is None:
            return super().get_occ(mo_energy, mo_coeff)
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GPU GC-SCF')

        mo_energy_cpu = cupy.asnumpy(cupy.asarray(mo_energy))
        if mo_energy_cpu.ndim == 1:
            if self.mu0 is None:
                self.mu, mo_occ = _smearing_optimize(
                    _fermi_smearing_occ, mo_energy_cpu,
                    self.mol.nelectron / 2.0, self.sigma)
                self.mu = _as_scalar_mu(self.mu)
            else:
                self.mu = float(self.mu0)
                mo_occ = _fermi_smearing_occ(self.mu, mo_energy_cpu,
                                             self.sigma)
            mo_occ = 2.0 * mo_occ
            self.entropy = _fermi_entropy_cpu(mo_occ, 2.0)
        elif self.fix_spin:
            if self.mu0 is None:
                self.mu = np.empty(2)
                mo_occ = np.empty_like(mo_energy_cpu)
                for s in range(2):
                    self.mu[s], mo_occ[s] = _smearing_optimize(
                        _fermi_smearing_occ, mo_energy_cpu[s], self.nelec[s],
                        self.sigma)
                    self.mu[s] = _as_scalar_mu(self.mu[s])
            else:
                if np.isscalar(self.mu0):
                    self.mu = np.array([float(self.mu0), float(self.mu0)])
                elif len(self.mu0) == 2:
                    self.mu = np.asarray(self.mu0, dtype=float)
                else:
                    raise TypeError(f'Unsupported mu0: {self.mu0}')
                mo_occ = np.asarray([
                    _fermi_smearing_occ(self.mu[s], mo_energy_cpu[s],
                                        self.sigma)
                    for s in range(2)
                ])
            self.entropy = _fermi_entropy_cpu(mo_occ, 1.0)
        else:
            mo_energy_flat = mo_energy_cpu.reshape(-1)
            if self.mu0 is None:
                self.mu, mo_occ = _smearing_optimize(
                    _fermi_smearing_occ, mo_energy_flat, self.mol.nelectron,
                    self.sigma)
                self.mu = _as_scalar_mu(self.mu)
            else:
                if not np.isscalar(self.mu0):
                    raise TypeError(f'Unsupported mu0: {self.mu0}')
                self.mu = float(self.mu0)
                mo_occ = _fermi_smearing_occ(self.mu, mo_energy_flat,
                                             self.sigma)
            mo_occ = mo_occ.reshape(mo_energy_cpu.shape)
            self.entropy = _fermi_entropy_cpu(mo_occ, 1.0)
        logger.info(self, '    sigma = %g  Optimized mu = %s  '
                    'entropy = %.12g', self.sigma, self.mu, self.entropy)
        return cupy.asarray(mo_occ)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = super().energy_tot(dm, h1e, vhf)
        if self.sigma is not None and self.mo_occ is not None:
            mo_occ = cupy.asarray(self.mo_occ)
            spin_degeneracy = 2.0 if mo_occ.ndim == 1 else 1.0
            self.entropy = _fermi_entropy_cpu(cupy.asnumpy(mo_occ),
                                              spin_degeneracy)
            self.nelectron = float(cupy.sum(mo_occ).get())
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            mu = np.asarray(self.mu0 if self.mu0 is not None else self.mu)
            if mu.ndim == 0:
                self.e_grand = self.e_free - float(mu) * self.nelectron
            else:
                self.e_grand = (
                    self.e_free
                    - mu @ cupy.asnumpy(cupy.sum(mo_occ, axis=1))
                )
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  '
                        'Grand potential = %.15g',
                        e_tot, self.e_free, self.e_grand)
        return e_tot

    def kernel(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if self.sigma is None:
            raise ValueError('sigma must be specified for GPU GC-SCF')
        if self.sigma <= 0:
            raise ValueError('sigma must be positive for GPU GC-SCF')

        conv_tol = kwargs.pop('conv_tol', self.conv_tol)
        conv_tol_grad = kwargs.pop('conv_tol_grad', self.conv_tol_grad)
        if conv_tol_grad is None:
            conv_tol_grad = conv_tol ** .5
            logger.info(self, 'Set gradient conv threshold to %g',
                        conv_tol_grad)
        max_cycle = kwargs.pop('max_cycle', self.max_cycle)
        callback = kwargs.pop('callback', self.callback)
        if kwargs:
            logger.warn(self, 'GPU GC-SCF kernel ignored unsupported kwargs %s',
                        sorted(kwargs))

        self.build(self.mol)
        self.dump_flags()

        hcore_ao = cupy.asarray(self.get_hcore(self.mol))
        S = cupy.asarray(self.get_ovlp(self.mol))
        x = cupy.asarray(self.check_linear_dependency(S, verbose=self.verbose))
        hcore = _hermitian_part(x.conj().T @ hcore_ao @ x)
        n_haux_eval = 0

        def haux_eval_counting(mo_energy, mo_coeff_orth):
            nonlocal n_haux_eval
            n_haux_eval += 1
            return _haux_eval(self, mo_energy, mo_coeff_orth, x,
                              hcore_ao, hcore)

        if isinstance(dm0, str):
            dm0 = self.from_chk(dm0)
        elif dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            dm0 = self.make_rdm1(self.mo_coeff, self.mo_occ)
        elif dm0 is None:
            dm0 = self.get_init_guess(self.mol, key=self.init_guess)
        dm0 = cupy.asarray(dm0)

        vhf_ao = self.get_veff(self.mol, dm0)
        if self.istype('UHF'):
            vhf = cupy.stack([x.conj().T @ v @ x for v in vhf_ao])
        else:
            vhf = x.conj().T @ vhf_ao @ x
        haux = _hermitian_part(hcore + vhf)
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
            log.info('GPU GC-SCF auxiliary-Hamiltonian minimization, '
                     'sigma = %.12g Ha, fixed nelectron = %.12g',
                     self.sigma, self.mol.nelectron)
        else:
            log.info('GPU GC-SCF auxiliary-Hamiltonian minimization, '
                     'sigma = %.12g Ha, mu0 = %s Ha',
                     self.sigma, self.mu0)

        for cycle in range(1, max_cycle + 1):
            if converged:
                break

            gradient = state.haux_gradient
            steepest_direction = _hermitian_part(
                state.hsub - _make_diagonal(state.mo_energy)
            )
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
                    log.info('GPU GC-SCF line minimization stopped at cycle '
                             '%d with converged residual: %s',
                             cycle, line_result.message)
                    break
                log.info('GPU GC-SCF line minimization failed at cycle %d '
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
                _matrix_rotation(gradient, rot)
            )
            previous_direction = _hermitian_part(
                _matrix_rotation(direction, rot)
            )
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
        self.mo_energy = cupy.array(state.mo_energy, copy=True)
        self.mo_coeff = cupy.array(state.mo_coeff, copy=True)
        self.mo_occ = cupy.array(state.mo_occ, copy=True)
        if haux.ndim == 2:
            self.haux = cupy.array(
                _hermitian_part(S @ x @ haux @ x.conj().T @ S), copy=True)
        else:
            self.haux = cupy.array(_hermitian_part(cupy.stack([
                S @ x @ h @ x.conj().T @ S for h in haux
            ])), copy=True)
        self.n_haux_eval = n_haux_eval
        self.auxh_residual_norm = float(residual_norm)
        self.cycles = niter_done

        if self.chkfile:
            self.dump_chk({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                           'mo_coeff': self.mo_coeff, 'mo_occ': self.mo_occ})
        logger.timer(self, 'GPU GC-SCF', *cput0)
        self._finalize()
        return self.e_tot

    scf = kernel

    def _finalize(self):
        if self.converged:
            logger.note(self, 'converged GPU GC-SCF energy = %.15g',
                        self.e_tot)
        else:
            logger.note(self, 'GPU GC-SCF not converged.')
            logger.note(self, 'GPU GC-SCF energy = %.15g', self.e_tot)
        logger.note(self, 'GPU GC-SCF free energy = %.15g', self.e_free)
        logger.note(self, 'GPU GC-SCF grand potential = %.15g', self.e_grand)
        logger.note(self, 'GPU GC-SCF electron number = %.15g',
                    self.nelectron)
        logger.note(self, 'GPU GC-SCF entropy = %.15g', self.entropy)
        logger.note(self, 'GPU GC-SCF AuxH evaluations = %d',
                    self.n_haux_eval)
        logger.note(self, 'GPU GC-SCF AuxH residual norm = %.3e',
                    self.auxh_residual_norm)
        return self

    def to_cpu(self):
        from pyscf.scf import gcscf as cpu_gcscf
        obj = cpu_gcscf.gcscf(self.undo_gcscf().to_cpu(), self.sigma,
                              self.mu0, self.fix_spin)
        obj.conv_tol_grad = self.conv_tol_grad
        obj.auxh_step = self.auxh_step
        obj.auxh_min_step = self.auxh_min_step
        obj.auxh_line_max_cycle = self.auxh_line_max_cycle
        return obj


def _haux_eval(mf, mo_energy, mo_coeff_orth, x, hcore_ao, hcore):
    mo_energy = cupy.asarray(mo_energy, dtype=float)
    mo_coeff_orth = cupy.asarray(mo_coeff_orth)
    mo_energy_cpu = cupy.asnumpy(mo_energy)

    if mo_energy.ndim == 1:
        mo_coeff = x @ mo_coeff_orth
        if mf.mu0 is None:
            mu, mo_occ_cpu = _smearing_optimize(
                _fermi_smearing_occ, mo_energy_cpu, mf.mol.nelectron / 2.0,
                mf.sigma)
            mu = _as_scalar_mu(mu)
        else:
            mu = float(mf.mu0)
            mo_occ_cpu = _fermi_smearing_occ(mu, mo_energy_cpu, mf.sigma)
        mo_occ_cpu = 2.0 * mo_occ_cpu
        mo_occ = cupy.asarray(mo_occ_cpu)

        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mf.mol, dm)
        fock = hcore + x.conj().T @ vhf @ x
        entropy = _fermi_entropy_cpu(mo_occ_cpu, 2.0)
        e_tot = float(mf.energy_elec(dm=dm, h1e=hcore_ao, vhf=vhf)[0]
                      + mf.energy_nuc())
        e_free = e_tot - mf.sigma * entropy
        nelectron = float(np.sum(mo_occ_cpu))
        e_grand = float(e_free - mu * nelectron)
        hsub = mo_coeff_orth.conj().T @ fock @ mo_coeff_orth
        grad_filling = _hermitian_part(hsub) - cupy.diag(mo_energy)
        if mf.mu0 is None:
            occ_prime = _fermi_occupation_derivative(mo_occ, mf.sigma, 2.0)
            denom = float(cupy.sum(occ_prime).get())
            if abs(denom) > np.finfo(float).tiny:
                diag_grad = cupy.diag(grad_filling).real
                dmu = float(cupy.sum(occ_prime * diag_grad).get() / denom)
                grad_filling = (
                    grad_filling - cupy.eye(mo_energy.size) * dmu
                )
        haux_gradient = _smearing_matrix_gradient(
            mo_energy, mo_occ, mf.sigma, grad_filling, 2.0)

    else:
        mo_coeff = cupy.stack([x @ c for c in mo_coeff_orth])
        if mf.fix_spin:
            if mf.mu0 is None:
                mu = np.empty(2)
                mo_occ_cpu = np.empty_like(mo_energy_cpu)
                for s in range(2):
                    mu[s], mo_occ_cpu[s] = _smearing_optimize(
                        _fermi_smearing_occ, mo_energy_cpu[s], mf.nelec[s],
                        mf.sigma)
                    mu[s] = _as_scalar_mu(mu[s])
            else:
                if np.isscalar(mf.mu0):
                    mu = np.array([float(mf.mu0), float(mf.mu0)])
                elif len(mf.mu0) == 2:
                    mu = np.asarray(mf.mu0, dtype=float)
                else:
                    raise TypeError(f'Unsupported mu0: {mf.mu0}')
                mo_occ_cpu = np.asarray([
                    _fermi_smearing_occ(mu[s], mo_energy_cpu[s], mf.sigma)
                    for s in range(2)
                ])
        else:
            mo_energy_flat = mo_energy_cpu.reshape(-1)
            if mf.mu0 is None:
                mu, mo_occ_cpu = _smearing_optimize(
                    _fermi_smearing_occ, mo_energy_flat, mf.mol.nelectron,
                    mf.sigma)
                mu = _as_scalar_mu(mu)
            else:
                if not np.isscalar(mf.mu0):
                    raise TypeError(f'Unsupported mu0: {mf.mu0}')
                mu = float(mf.mu0)
                mo_occ_cpu = _fermi_smearing_occ(mu, mo_energy_flat,
                                                 mf.sigma)
            mo_occ_cpu = mo_occ_cpu.reshape(mo_energy_cpu.shape)
        mo_occ = cupy.asarray(mo_occ_cpu)

        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mf.mol, dm)
        fock = hcore + cupy.stack([x.conj().T @ v @ x for v in vhf])
        entropy = _fermi_entropy_cpu(mo_occ_cpu, 1.0)
        e_tot = float(mf.energy_elec(dm=dm, h1e=hcore_ao, vhf=vhf)[0]
                      + mf.energy_nuc())
        e_free = e_tot - mf.sigma * entropy
        nelectron = float(np.sum(mo_occ_cpu))
        mu_array = np.asarray(mu)
        if mu_array.ndim == 0:
            e_grand = float(e_free - float(mu_array) * nelectron)
        else:
            e_grand = float(e_free - mu_array @ np.sum(mo_occ_cpu, axis=1))
        hsub = _matrix_rotation(fock, mo_coeff_orth)
        grad_filling = _hermitian_part(hsub) - _make_diagonal(mo_energy)
        if mf.mu0 is None:
            occ_prime = _fermi_occupation_derivative(mo_occ, mf.sigma, 1.0)
            diag_grad = cupy.diagonal(grad_filling, axis1=-2,
                                      axis2=-1).real
            if mf.fix_spin:
                dmu = cupy.zeros(2)
                for s in range(2):
                    denom = float(cupy.sum(occ_prime[s]).get())
                    if abs(denom) > np.finfo(float).tiny:
                        dmu[s] = cupy.sum(
                            occ_prime[s] * diag_grad[s]) / denom
                grad_filling = grad_filling - _make_diagonal(
                    cupy.repeat(dmu[:,None], mo_energy.shape[1], axis=1))
            else:
                denom = float(cupy.sum(occ_prime).get())
                if abs(denom) > np.finfo(float).tiny:
                    dmu = float(cupy.sum(occ_prime * diag_grad).get()
                                / denom)
                    grad_filling = grad_filling - _make_diagonal(
                        cupy.full_like(mo_energy, dmu))
        haux_gradient = _smearing_matrix_gradient(
            mo_energy, mo_occ, mf.sigma, grad_filling, 1.0)

    return SimpleNamespace(
        mo_energy=mo_energy, mo_coeff=mo_coeff, mo_coeff_orth=mo_coeff_orth,
        mo_occ=mo_occ, dm=dm, vhf=vhf, fock=fock, hsub=hsub,
        entropy=entropy, nelectron=nelectron, mu=mu, e_tot=e_tot,
        e_free=e_free, e_grand=e_grand,
        objective=e_free if mf.mu0 is None else e_grand,
        haux_gradient=haux_gradient)


def _eigh_tensor(A):
    if A.ndim == 2:
        return cupy.linalg.eigh(A)
    mo_energy = []
    mo_coeff = []
    for block in A:
        e, c = cupy.linalg.eigh(block)
        mo_energy.append(e)
        mo_coeff.append(c)
    return cupy.stack(mo_energy), cupy.stack(mo_coeff)


def _make_diagonal(vv):
    if vv.ndim == 1:
        return cupy.diag(vv)
    return cupy.stack([cupy.diag(v) for v in vv])


def _identity_rotation(mo_energy):
    nmo = mo_energy.shape[-1]
    if mo_energy.ndim == 1:
        return cupy.eye(nmo)
    else:
        nspin = mo_energy.shape[0]
        return cupy.stack([cupy.eye(nmo) for _ in range(nspin)])


def _matrix_rotation(mat, rot):
    if mat.ndim == 2 and rot.ndim == 2:
        return rot.conj().T @ mat @ rot
    elif mat.ndim == 3 and rot.ndim == 3:
        return cupy.stack([r.conj().T @ m @ r for m, r in zip(mat, rot)])
    else:
        raise ValueError("Input matrix must be 2D or 3D")


def _haux_from_mo(mo_energy, mo_coeff_orth):
    mo_energy = cupy.asarray(mo_energy, dtype=float)
    mo_coeff_orth = cupy.asarray(mo_coeff_orth)
    if mo_energy.ndim == 1:
        return (mo_coeff_orth * mo_energy) @ mo_coeff_orth.conj().T
    else:
        return _hermitian_part(
            (mo_coeff_orth * mo_energy[:, None, :])
            @ mo_coeff_orth.conj().transpose(0, 2, 1))


def _line_minimize(haux_eval, haux, direction, state, alpha_t,
                   alpha_t_min, max_cycle):
    if alpha_t <= 0:
        raise ValueError('auxh_step must be positive')
    if alpha_t_min <= 0:
        raise ValueError('auxh_min_step must be positive')
    if max_cycle <= 0:
        raise ValueError('auxh_line_max_cycle must be positive')

    alpha_reduce_factor = 0.1
    alpha_increase_factor = 3.0
    objective_slop = 1e-12
    objective0 = float(state.objective)
    gdotd = _vdot(state.haux_gradient, direction)
    identity = _identity_rotation(state.mo_energy)
    haux_eigenbasis = _make_diagonal(state.mo_energy)
    mo_coeff0 = state.mo_coeff_orth
    line_candidates = [
        (0.0, state, _norm(state.haux_gradient), haux, identity)
    ]

    def line_result(candidate, niter, next_alpha_t, success, message):
        alpha, candidate_state, residual, candidate_haux, rotation = candidate
        return SimpleNamespace(
            haux=candidate_haux, state=candidate_state, alpha=alpha,
            residual_norm=residual, niter=niter, next_alpha_t=next_alpha_t,
            rotation=rotation, success=success, message=message)

    if gdotd >= 0.0:
        return line_result(line_candidates[0], 0, alpha_t, False,
                           'search direction is not downhill')

    def make_trialstate(alpha):
        alpha = float(max(alpha, 0.0))
        trial_haux_eigenbasis = _hermitian_part(haux_eigenbasis
                                                + alpha * direction)
        mo_energy, rotation = _eigh_tensor(trial_haux_eigenbasis)
        mo_coeff_orth = mo_coeff0 @ rotation
        trial_state = haux_eval(mo_energy, mo_coeff_orth)
        trial_haux = _haux_from_mo(mo_energy, mo_coeff_orth)
        result = (
            alpha, trial_state, _norm(trial_state.haux_gradient),
            trial_haux, rotation
        )
        line_candidates.append(result)
        return result

    niter = 0
    alpha_trial = float(alpha_t)
    alpha = alpha_trial
    for _ in range(max_cycle):
        if alpha_trial < alpha_t_min:
            return line_result(line_candidates[0], niter, alpha_trial, False,
                               'test step fell below auxh_min_step')

        niter += 1
        test_candidate = make_trialstate(alpha_trial)
        test_state = test_candidate[1]
        test_residual = test_candidate[2]
        test_objective = float(test_state.objective)
        if not np.isfinite(test_objective):
            alpha_trial *= alpha_reduce_factor
            continue

        denominator = alpha_trial * gdotd + objective0 - test_objective
        denominator_tol = np.finfo(float).eps * max(
            1.0, abs(alpha_trial * gdotd), abs(objective0),
            abs(test_objective))
        if abs(denominator) <= denominator_tol:
            if test_objective <= objective0 or (
                    test_objective <= objective0 + objective_slop
                    and test_residual < line_candidates[0][2]):
                return line_result(
                    test_candidate, niter,
                    alpha_trial * alpha_increase_factor, True,
                    'accepted test step with flat quadratic curvature')
            alpha_trial *= alpha_reduce_factor
            continue

        alpha = 0.5 * alpha_trial * alpha_trial * gdotd / denominator
        if not np.isfinite(alpha):
            alpha_trial *= alpha_reduce_factor
            continue
        if alpha < 0.0:
            return line_result(
                test_candidate, niter, alpha_trial * alpha_increase_factor,
                True, 'accepted downhill test step with wrong quadratic '
                'curvature')
        if alpha / alpha_trial > alpha_increase_factor:
            alpha_trial *= alpha_increase_factor
            continue
        if alpha > 0.0 and alpha_trial / alpha < alpha_reduce_factor:
            alpha_trial *= alpha_reduce_factor
            continue
        break
    else:
        return _best_line_min_candidate(line_candidates, niter,
                                        'test step adjustment failed')

    for _ in range(max_cycle):
        niter += 1
        trial_state = make_trialstate(alpha)[1]
        trial_objective = float(trial_state.objective)
        if not np.isfinite(trial_objective):
            alpha *= alpha_reduce_factor
            continue
        line_candidate = _best_line_min_candidate(
            line_candidates, niter, 'accepted best line candidate',
            objective_slop=objective_slop)
        if line_candidate.success:
            return line_candidate
        if trial_objective > objective0:
            alpha *= alpha_reduce_factor
            continue

    return _best_line_min_candidate(line_candidates, niter,
                                    'predicted step failed to reduce objective',
                                    objective_slop=objective_slop)


def _best_line_min_candidate(line_candidates, niter, failure_message,
                             objective_slop=0.0):
    finite_candidates = [candidate for candidate in line_candidates
                         if np.isfinite(candidate[1].objective)]
    objective_threshold = min(candidate[1].objective
                              for candidate in finite_candidates)
    objective_threshold += objective_slop
    best_alpha, best_state, best_residual, best_haux, best_rotation = min(
        (candidate for candidate in finite_candidates
         if candidate[1].objective <= objective_threshold),
        key=lambda item: (item[2], item[1].objective))
    return SimpleNamespace(
        haux=best_haux, state=best_state, alpha=best_alpha,
        residual_norm=best_residual, niter=niter, next_alpha_t=best_alpha,
        rotation=best_rotation, success=best_alpha != 0.0,
        message=('%s; accepted best lower-energy trial' % failure_message
                 if best_alpha != 0.0 else failure_message))


def _smearing_matrix_gradient(eta, mo_occ, sigma, grad_filling,
                              spin_degeneracy):
    eta = cupy.asarray(eta, dtype=float)
    mo_occ = cupy.asarray(mo_occ, dtype=float)
    grad_filling = cupy.asarray(grad_filling)
    if eta.ndim == 2:
        return cupy.stack([
            _smearing_matrix_gradient(eta[s], mo_occ[s], sigma,
                                      grad_filling[s], spin_degeneracy)
            for s in range(eta.shape[0])
        ])

    occ_prime = _fermi_occupation_derivative(mo_occ, sigma, spin_degeneracy)
    energy_diff = eta[:,None] - eta[None,:]
    occ_diff = mo_occ[:,None] - mo_occ[None,:]
    factors = occ_diff / energy_diff

    near_degenerate = cupy.isclose(energy_diff, 0.0, atol=1e-12, rtol=1e-12)
    derivative_average = .5 * (occ_prime[:, None] + occ_prime[None, :])
    factors = cupy.where(cupy.abs(energy_diff) > 1e-12, factors, 0.0)
    factors[near_degenerate] = derivative_average[near_degenerate]

    return _hermitian_part(_hermitian_part(grad_filling) * factors)


def _fermi_occupation_derivative(mo_occ, sigma, spin_degeneracy):
    f = cupy.asarray(mo_occ, dtype=float) / spin_degeneracy
    return -spin_degeneracy * f * (1.0 - f) / sigma


def _fermi_entropy_cpu(mo_occ, spin_degeneracy):
    f = np.asarray(mo_occ, dtype=float) / spin_degeneracy
    f = f[(f > 0.0) & (f < 1.0)]
    if f.size == 0:
        return 0.0
    entropy_per_spin = -(f * np.log(f) + (1.0 - f) * np.log(1.0 - f)).sum()
    return float(spin_degeneracy * entropy_per_spin)


def _hermitian_part(matrix):
    matrix = cupy.asarray(matrix)
    if matrix.ndim == 2:
        return (matrix + matrix.conj().T) * .5
    elif matrix.ndim == 3:
        return (matrix + matrix.conj().transpose(0, 2, 1)) * .5
    else:
        raise ValueError("Input matrix must be 2D or 3D")


def _as_scalar_mu(mu):
    return float(mu if not hasattr(mu, '__iter__') else mu[0])


def _vdot(left, right):
    return float(cupy.vdot(left, right).real.get())


def _norm(matrix):
    return float(cupy.linalg.norm(matrix).get())


def from_cpu(method):
    from pyscf.scf import gcscf as cpu_gcscf
    assert isinstance(method, cpu_gcscf._GCSCF)
    return gcscf(method.undo_gcscf().to_gpu(), method.sigma, method.mu0,
                 method.fix_spin)
