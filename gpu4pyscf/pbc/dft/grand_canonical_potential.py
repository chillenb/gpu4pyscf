"""Experimental potential-space backend for :class:`GrandCanonicalKRKS`.

The established auxiliary-Hamiltonian DIIS and NLCG entry points do not call
this module.  It mixes only the complete density-dependent scalar local KS
potential exposed by the LPBE multigrid NumInt contract.
"""

from types import SimpleNamespace

import cupy as cp
import numpy as np

from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import potential_mixing
from gpu4pyscf.pbc.tools import elliptic
from gpu4pyscf.pbc.tools import pbc as pbc_tools


class PotentialCycle:
    __slots__ = (
        'electronic', 'v_in_g', 'v_out_g', 'residual_g',
        'grid_residual_rms', 'cavity_r', 'lpbe_pot_guess',
        'mixing_diagnostics', 'delta_v0',
    )

    def __init__(self, electronic, v_in_g, v_out_g, residual_g,
                 grid_residual_rms, cavity_r, lpbe_pot_guess,
                 mixing_diagnostics=None, delta_v0=0.0):
        self.electronic = electronic
        self.v_in_g = v_in_g
        self.v_out_g = v_out_g
        self.residual_g = residual_g
        self.grid_residual_rms = float(grid_residual_rms)
        self.cavity_r = cavity_r
        self.lpbe_pot_guess = lpbe_pot_guess
        self.mixing_diagnostics = mixing_diagnostics
        self.delta_v0 = float(delta_v0)


def _copy_optional(value):
    return None if value is None else cp.asarray(value).copy()


def _numint(solver):
    ni = getattr(solver.mf, '_numint', None)
    required = ('local_potential_to_ao', 'mesh', 'pot_guess')
    if ni is None or any(not hasattr(ni, name) for name in required):
        raise TypeError(
            'potential_scf requires LPBEMultiGridNumInt with the LPBE '
            'grid contract')
    return ni


def _context(solver, cavity_r):
    ni = _numint(solver)
    mesh = tuple(int(x) for x in ni.mesh)
    Gv = cp.asarray(pbc_tools.get_Gv(solver.cell, mesh))
    G2 = elliptic.reciprocal_laplacian_symbol(Gv, mesh)
    return SimpleNamespace(
        cell=solver.cell, mesh=mesh, Gv=Gv, G2=G2,
        cavity_r=cp.asarray(cavity_r).copy())


def _validate_vlocal(vlocal_g, mesh, name='local potential'):
    vlocal_g = cp.asarray(vlocal_g, dtype=cp.complex128).reshape(-1)
    ngrids = int(np.prod(mesh))
    if vlocal_g.size != ngrids:
        raise ValueError(
            '%s has %d values; mesh requires %d' %
            (name, vlocal_g.size, ngrids))
    if not bool(cp.all(cp.isfinite(vlocal_g)).item()):
        raise FloatingPointError('%s contains nonfinite values' % name)
    return vlocal_g.copy()


def _initial_dm(solver, dm0):
    if dm0 is None:
        kwargs = {'kpts': solver.mf.kpts}
        if hasattr(solver.mf, 'init_guess'):
            kwargs['key'] = solver.mf.init_guess
        try:
            dm0 = solver.mf.get_init_guess(solver.cell, **kwargs)
        except TypeError:
            kwargs.pop('key', None)
            try:
                dm0 = solver.mf.get_init_guess(solver.cell, **kwargs)
            except TypeError:
                dm0 = solver.mf.get_init_guess(solver.cell)
    dm = cp.asarray(dm0)
    if dm.ndim == 2 and solver.nkpts == 1:
        dm = dm[None]
    if dm.shape != (solver.nkpts, solver.nao, solver.nao):
        raise ValueError('initial density has the wrong shape')
    return cp.stack(solver._hermi([dm[k] for k in range(solver.nkpts)]))


def _initial_potential(solver, dm0):
    """Evaluate ``dm0`` once and retain its tagged output potential."""
    ni = _numint(solver)
    dm = _initial_dm(solver, dm0)
    solver.nfev += 1
    veff = solver.mf.get_veff(
        solver.cell, dm, dm_last=None, vhf_last=None, hermi=1,
        kpts=solver.mf.kpts, kpts_band=None)
    grid = getattr(veff, 'lpbe_grid', None)
    if grid is None:
        raise TypeError('LPBE get_veff did not return lpbe_grid metadata')
    vlocal_g = _validate_vlocal(grid.vlocal_g, ni.mesh)
    # This seed evaluation is accepted as a warm-start source.  The first
    # fixed-N evaluation below remains transactional relative to it.
    ni.pot_guess = _copy_optional(grid.lpbe_pot_guess)
    return vlocal_g


def evaluate_fixed_n_potential(solver, v_in_g, nelec):
    """Evaluate one fixed-N KS map without committing NumInt warm state."""
    ni = _numint(solver)
    mesh = tuple(int(x) for x in ni.mesh)
    v_in_g = _validate_vlocal(v_in_g, mesh, 'input local potential')
    accepted_lpbe_guess = _copy_optional(ni.pot_guess)

    local_ao = ni.local_potential_to_ao(
        v_in_g, kpts=solver.mf.kpts, hermi=1)
    local_ao = cp.asarray(local_ao)
    if local_ao.shape != (solver.nkpts, solver.nao, solver.nao):
        raise ValueError('local-potential AO conversion has the wrong shape')
    h_ao = cp.stack([
        hcore + local_ao[k]
        for k, hcore in enumerate(solver.hcore_ao)
    ])
    h = solver._sanitize_h(solver._to_orth(h_ao))

    try:
        electronic = solver.calculate_cycle(
            h, nelec=nelec, align_fixed_n_gauge=False)
        grid = electronic.grid_data
        if grid is None:
            raise TypeError(
                'LPBE get_veff did not return lpbe_grid metadata')
        v_out_g = _validate_vlocal(
            grid.vlocal_g, mesh, 'output local potential')
        candidate_lpbe_guess = _copy_optional(grid.lpbe_pot_guess)
        cavity_r = cp.asarray(grid.cavity_r, dtype=cp.float64).copy()
    finally:
        # A rejected outer trial must not influence any later physical LPBE
        # solve.  ``commit_potential_cycle`` is the only commit point.
        ni.pot_guess = accepted_lpbe_guess

    delta_g0 = v_out_g[0] - v_in_g[0]
    if abs(float(delta_g0.imag.item())) > 1e-10:
        raise ValueError('constant potential residual has an imaginary part')
    delta_v0 = float(delta_g0.real.item()) / float(solver.cell.vol)

    # At fixed N, a constant input shift changes no density.  Align the input
    # field to the physical LPBE output gauge exactly, then shift every
    # derived one-particle quantity by the corresponding constant AO field.
    v_in_g[0] = v_out_g[0]
    electronic.h = [
        value + delta_v0 * eye
        for value, eye in zip(electronic.h, solver.identity)
    ]
    electronic.eig = [value + delta_v0 for value in electronic.eig]
    electronic.mu += delta_v0
    electronic.residual = [
        value - delta_v0 * eye
        for value, eye in zip(electronic.residual, solver.identity)
    ]
    electronic.residual_rms = solver._rms(electronic.residual)
    electronic.grand_potential = (
        electronic.free_energy - electronic.mu * electronic.nelec)

    residual_g = v_out_g - v_in_g
    residual_g[0] = 0.0
    context = _context(solver, cavity_r)
    residual_r = potential_mixing.reciprocal_to_real(
        residual_g, context)
    grid_residual_rms = potential_mixing.grid_rms(residual_r)
    return PotentialCycle(
        electronic, v_in_g, v_out_g, residual_g,
        grid_residual_rms, cavity_r, candidate_lpbe_guess,
        delta_v0=delta_v0)


def commit_potential_cycle(solver, cycle):
    """Commit warm state belonging to an accepted potential cycle."""
    _numint(solver).pot_guess = _copy_optional(cycle.lpbe_pot_guess)


def _make_preconditioner(kind, q0_sq, a_out, b_metal,
                         preconditioner_tol, preconditioner_maxiter):
    if hasattr(kind, 'apply'):
        return kind
    kind = str(kind).strip().lower()
    if kind == 'identity':
        return potential_mixing.IdentityPreconditioner()
    if kind == 'kerker':
        return potential_mixing.KerkerPreconditioner(q0_sq)
    if kind == 'elliptic':
        fallback = potential_mixing.KerkerPreconditioner(q0_sq)
        return potential_mixing.EllipticPreconditioner(
            a_out=a_out, b_metal=b_metal,
            tol=preconditioner_tol,
            maxiter=preconditioner_maxiter, fallback=fallback)
    raise ValueError(
        "preconditioner must be 'identity', 'kerker', 'elliptic', "
        'or an object with apply()')


def _score(cycle, grid_tolerance, matrix_tolerance):
    return max(
        cycle.grid_residual_rms / grid_tolerance,
        cycle.electronic.residual_rms / matrix_tolerance)


def potential_scf(self, dm0=None, v0_g=None, preconditioner='identity',
                  alpha=0.2, anderson_space=0, potential_conv_tol=None,
                  q0_sq=1.0, a_out=1.0, b_metal=0.1,
                  preconditioner_tol=1e-8, preconditioner_maxiter=200,
                  max_step_rms=0.5, max_step_abs=2.0,
                  residual_growth_factor=2.0, max_backtracks=6):
    """Run the opt-in fixed-N LPBE potential-space SCF backend.

    ``R_G = V_out,G - V_in,G`` is used throughout.  Rejected contractions
    restore the accepted physical LPBE warm start and never enter Anderson
    history.
    """
    self.build()
    if self.nelec is None:
        raise NotImplementedError(
            'potential_scf fixed-mu integration requires the fixed-N '
            'potential gate first')
    if potential_conv_tol is None:
        potential_conv_tol = self.conv_tol
    potential_conv_tol = float(potential_conv_tol)
    if not np.isfinite(potential_conv_tol) or potential_conv_tol <= 0.0:
        raise ValueError('potential_conv_tol must be finite and positive')
    if (not np.isfinite(residual_growth_factor)
            or residual_growth_factor <= 1.0):
        raise ValueError('residual_growth_factor must exceed one')
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')

    self.dump_flags()
    self.converged = False
    self.cycles = 0
    self.outer_cycles = 0
    self.nfev = 0
    self.refinements = 0
    self.message = ''
    ni = _numint(self)
    if v0_g is None:
        v0_g = _initial_potential(self, dm0)
    else:
        v0_g = _validate_vlocal(v0_g, ni.mesh, 'initial local potential')

    preconditioner_object = _make_preconditioner(
        preconditioner, q0_sq, a_out, b_metal,
        preconditioner_tol, preconditioner_maxiter)
    mixer = potential_mixing.AndersonMixer(
        alpha=alpha, history=anderson_space,
        max_step_rms=max_step_rms, max_step_abs=max_step_abs)

    current = evaluate_fixed_n_potential(self, v0_g, self.nelec)
    commit_potential_cycle(self, current)
    context = _context(self, current.cavity_r)
    mixer.accept(current.v_in_g, current.residual_g, context)
    self.cycles = 1
    best = current
    best_score = _score(current, potential_conv_tol, self.conv_tol)
    converged = False

    logger.info(
        self,
        'Potential cycle %d  N = %.12g  mu = %.12g  A = %.12g  '
        'grid residual = %.6g  matrix residual = %.6g  delta V0 = %.6g',
        self.cycles, self.nelec, current.electronic.mu,
        current.electronic.free_energy, current.grid_residual_rms,
        current.electronic.residual_rms, current.delta_v0)

    while self.cycles < self.max_cycle:
        if (current.grid_residual_rms <= potential_conv_tol and
                current.electronic.residual_rms <= self.conv_tol):
            converged = True
            self.message = 'converged fixed-N potential residuals'
            break

        context = _context(self, current.cavity_r)
        proposal = mixer.propose(
            current.v_in_g, current.residual_g,
            preconditioner_object, context)
        step_g = proposal.step_g
        accepted = None
        rejected = 0
        for contraction in range(max_backtracks + 1):
            scale = 0.5 ** contraction
            trial_v_g = current.v_in_g + scale * step_g
            trial = evaluate_fixed_n_potential(
                self, trial_v_g, self.nelec)
            trial.mixing_diagnostics = proposal.diagnostics
            growth_limit = (
                residual_growth_factor
                * max(current.grid_residual_rms,
                      np.finfo(float).tiny))
            if trial.grid_residual_rms <= growth_limit:
                accepted = trial
                break
            rejected += 1
            logger.info(
                self,
                'Rejected potential trial contraction %.6g  grid '
                'residual %.6g > %.6g',
                scale, trial.grid_residual_rms, growth_limit)

        if accepted is None:
            mixer.reset()
            self.message = (
                'potential step rejected after %d backtracks' %
                max_backtracks)
            break
        if rejected:
            mixer.reset()

        current = accepted
        commit_potential_cycle(self, current)
        context = _context(self, current.cavity_r)
        mixer.accept(current.v_in_g, current.residual_g, context)
        self.cycles += 1
        score = _score(current, potential_conv_tol, self.conv_tol)
        if score < best_score:
            best = current
            best_score = score
        logger.info(
            self,
            'Potential cycle %d  N = %.12g  mu = %.12g  A = %.12g  '
            'grid residual = %.6g  matrix residual = %.6g  '
            'step RMS = %.6g  backtracks = %d',
            self.cycles, self.nelec, current.electronic.mu,
            current.electronic.free_energy, current.grid_residual_rms,
            current.electronic.residual_rms,
            proposal.diagnostics.step_rms, rejected)
        if callable(self.callback):
            self.callback({
                'solver': self,
                'cycle_data': current.electronic,
                'potential_cycle': current,
                'best_cycle_data': best.electronic,
                'best_potential_cycle': best,
                'cycle': self.cycles,
                'electron_number': self.nelec,
                'rejected_trials': rejected,
            })
    else:
        self.message = 'maximum fixed-N potential cycles reached'

    if (current.grid_residual_rms <= potential_conv_tol and
            current.electronic.residual_rms <= self.conv_tol):
        converged = True
        self.message = 'converged fixed-N potential residuals'
        best = current
    elif not self.message:
        self.message = 'maximum fixed-N potential cycles reached'

    commit_potential_cycle(self, best)
    self._potential_cycle = best
    logger.info(
        self, '%s; total Fock/LPBE evaluations = %d',
        self.message, self.nfev)
    energy = self._finalize(best.electronic, converged)
    self.potential_residual_rms = best.grid_residual_rms
    self.scf_summary.update({
        'potential_residual_rms': best.grid_residual_rms,
        'potential_fock_evaluations': self.nfev,
    })
    self.mf.scf_summary.update({
        'potential_residual_rms': best.grid_residual_rms,
        'potential_fock_evaluations': self.nfev,
    })
    return energy


potential_kernel = potential_scf
