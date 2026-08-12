"""Restartable PBE/LPBE calculation for 1/4-ML CO/Cu(111)."""

import os

import cupy as cp
import numpy as np
import pyscf
from pyscf.pbc import gto

from gpu4pyscf.pbc.dft import GrandCanonicalKRKS
from gpu4pyscf.pbc.solvent.lpbe_as_numint import multigrid_lpbe


POSCAR = 'Cu_CO_224_ads.vasp'
SIGMA = float(os.environ.get('GC_SIGMA', '5e-3'))  # Ha
if not np.isfinite(SIGMA) or SIGMA <= 0.0:
    raise ValueError('GC_SIGMA must be finite and positive')
TARGET_MU = None
if 'GC_MU' in os.environ:
    TARGET_MU = float(os.environ['GC_MU'])
    if not np.isfinite(TARGET_MU):
        raise ValueError('GC_MU must be finite')
KPT_MESH = [3, 3, 1]
IONIC_STRENGTH = 1.0  # mol/L
REL_PERMITTIVITY = 78.4
LPBE_TOL = 1e-12

CHECKPOINT = os.environ.get(
    'GC_CHECKPOINT', 'co_cu_lpbe_sigma5e-3.npz')
RESTART = os.environ.get('GC_RESTART')
METHOD = os.environ.get('GC_METHOD', 'nlcg').strip().lower()
POTENTIAL_METHODS = {
    'potential-simple': ('identity', 0),
    'potential-kerker': ('kerker', 0),
    'potential-anderson': ('identity', 6),
    'potential-kerker-anderson': ('kerker', 6),
    'potential-elliptic': ('elliptic', 6),
}
if METHOD not in ('nlcg', 'diis', 'potential', *POTENTIAL_METHODS):
    raise ValueError(
        'GC_METHOD must be nlcg, diis, potential, or a documented '
        'potential-* mode')


def save_checkpoint(state, path, potential_cycle=None):
    payload = {'count': np.asarray(len(state.h), dtype=np.int64)}
    payload.update({
        'h%d' % index: cp.asnumpy(value)
        for index, value in enumerate(state.h)
    })
    if potential_cycle is not None:
        payload.update({
            'v_in_g': cp.asnumpy(potential_cycle.v_in_g),
            'lpbe_pot_guess': cp.asnumpy(
                potential_cycle.lpbe_pot_guess),
            'electron_number': np.asarray(
                potential_cycle.electronic.nelec),
        })
    temporary = path + '.tmp.npz'
    np.savez(temporary, **payload)
    os.replace(temporary, path)


a, atom = gto.fromfile(POSCAR)
cell = pyscf.M(
    a=a,
    atom=atom,
    basis={
        'Cu': 'DZVP-MOLOPT-PBE-GTH',
        'C': 'DZVP-MOLOPT-SR-GTH',
        'O': 'DZVP-MOLOPT-SR-GTH',
    },
    pseudo='GTH-PBE',
    verbose=4,
    ke_cutoff=200.0,
    nelec_frac=True,
)
cell.build()
kpts = cell.make_kpts(KPT_MESH)

mf = cell.KRKS(xc='pbe', kpts=kpts).to_gpu()
mf = multigrid_lpbe(
    mf,
    tol=LPBE_TOL,
    ionic_strength=IONIC_STRENGTH,
    rel_permittivity=REL_PERMITTIVITY,
)

if TARGET_MU is None:
    solver = GrandCanonicalKRKS(
        mf, nelec=cell.tot_electrons(1), sigma=SIGMA)
else:
    solver = GrandCanonicalKRKS(mf, mu=TARGET_MU, sigma=SIGMA)
solver.max_cycle = int(os.environ.get('GC_MAX_CYCLE', '100'))
if 'GC_DIIS_SPACE' in os.environ:
    solver.diis_space = int(os.environ['GC_DIIS_SPACE'])
if 'GC_DAMP' in os.environ:
    solver.damp = float(os.environ['GC_DAMP'])
if 'GC_DIIS_EXPANSION' in os.environ:
    solver.diis_expansion = float(os.environ['GC_DIIS_EXPANSION'])
if 'GC_NLCG_PULAY_STAGNATION_STEPS' in os.environ:
    from gpu4pyscf.pbc.dft import grand_canonical_cg as gc_cg
    pulay_stagnation_steps = int(
        os.environ['GC_NLCG_PULAY_STAGNATION_STEPS'])
    if pulay_stagnation_steps < 0:
        raise ValueError('GC_NLCG_PULAY_STAGNATION_STEPS must be nonnegative')
    gc_cg.NLCG_PULAY_STAGNATION_STEPS = pulay_stagnation_steps
    print('NLCG Pulay stagnation steps = %d' % pulay_stagnation_steps)
if 'GC_NLCG_OCCUPATION_PR' in os.environ:
    from gpu4pyscf.pbc.dft import grand_canonical_cg as gc_cg
    occupation_pr = os.environ['GC_NLCG_OCCUPATION_PR'].strip().lower()
    if occupation_pr not in ('0', '1', 'false', 'true', 'no', 'yes'):
        raise ValueError(
            'GC_NLCG_OCCUPATION_PR must be a boolean value')
    gc_cg.NLCG_OCCUPATION_PR_ENABLED = occupation_pr in ('1', 'true', 'yes')
    print('NLCG occupation-preconditioned PR+ = %s' %
          gc_cg.NLCG_OCCUPATION_PR_ENABLED)
if 'GC_NLCG_INITIAL_STEP' in os.environ:
    solver.nlcg_initial_step = float(os.environ['GC_NLCG_INITIAL_STEP'])
if 'GC_NLCG_MAX_LINE_EVALUATIONS' in os.environ:
    solver.nlcg_max_line_search_evaluations = int(
        os.environ['GC_NLCG_MAX_LINE_EVALUATIONS'])

h = None
v0_g = None
restart_nelec = None
if RESTART:
    with np.load(RESTART, allow_pickle=False) as data:
        count = int(data['count'])
        h = [cp.asarray(data['h%d' % index]) for index in range(count)]
        if 'v_in_g' in data:
            v0_g = cp.asarray(data['v_in_g'])
            restart_nelec = float(data['electron_number'])
            mf._numint.pot_guess = cp.asarray(data['lpbe_pot_guess'])
    print('Loaded %d Hamiltonian blocks from %s' % (count, RESTART))
    if v0_g is not None:
        print('Loaded potential-space restart at N = %.12g' % restart_nelec)


def checkpoint_callback(environment):
    state = environment['cycle_data']
    best_state = environment.get('best_cycle_data', state)
    potential_cycle = environment.get('potential_cycle')
    best_potential_cycle = environment.get(
        'best_potential_cycle', potential_cycle)
    save_checkpoint(
        state, CHECKPOINT + '.current.npz', potential_cycle)
    save_checkpoint(best_state, CHECKPOINT, best_potential_cycle)
    grid_residual = (
        np.nan if potential_cycle is None
        else potential_cycle.grid_residual_rms)
    print(
        'CHECKPOINT cycle=%d current_residual=%.12g '
        'current_grid_residual=%.12g best_residual=%.12g path=%s' % (
            environment['cycle'], state.residual_rms,
            grid_residual, best_state.residual_rms, CHECKPOINT),
        flush=True,
    )


solver.callback = checkpoint_callback
if METHOD == 'nlcg':
    solver.nlcg(h=h)
elif METHOD == 'diis':
    solver.kernel(h=h)
else:
    default_preconditioner, default_space = POTENTIAL_METHODS.get(
        METHOD, ('identity', 0))
    potential_preconditioner = os.environ.get(
        'GC_POTENTIAL_PRECONDITIONER', default_preconditioner).strip().lower()
    potential_anderson_space = int(os.environ.get(
        'GC_POTENTIAL_ANDERSON_SPACE', str(default_space)))
    print(
        'Potential mixing preconditioner=%s Anderson space=%d' %
        (potential_preconditioner, potential_anderson_space),
        flush=True)
    solver.potential_scf(
        v0_g=v0_g,
        initial_nelec=restart_nelec,
        preconditioner=potential_preconditioner,
        alpha=float(os.environ.get('GC_POTENTIAL_ALPHA', '0.2')),
        anderson_space=potential_anderson_space,
        potential_conv_tol=float(os.environ.get(
            'GC_POTENTIAL_CONV_TOL', str(solver.conv_tol))),
        q0_sq=float(os.environ.get('GC_POTENTIAL_Q0_SQ', '1.0')),
        a_out=float(os.environ.get('GC_POTENTIAL_A_OUT', '1.0')),
        b_metal=float(os.environ.get('GC_POTENTIAL_B_METAL', '0.1')),
        preconditioner_tol=float(os.environ.get(
            'GC_POTENTIAL_INNER_TOL', '1e-8')),
        preconditioner_maxiter=int(os.environ.get(
            'GC_POTENTIAL_INNER_MAXITER', '200')),
        max_step_rms=float(os.environ.get(
            'GC_POTENTIAL_MAX_STEP_RMS', '0.5')),
        max_step_abs=float(os.environ.get(
            'GC_POTENTIAL_MAX_STEP_ABS', '2.0')),
        residual_growth_factor=float(os.environ.get(
            'GC_POTENTIAL_RESIDUAL_GROWTH', '2.0')),
        max_backtracks=int(os.environ.get(
            'GC_POTENTIAL_MAX_BACKTRACKS', '6')),
        cavity_change_threshold=float(os.environ.get(
            'GC_POTENTIAL_CAVITY_RESET', '0.1')),
    )

print(
    'RESULT converged=%s cycles=%d nfev=%d free_energy=%.12g '
    'grand_potential=%.12g residual=%.12g grid_residual=%.12g '
    'nelec=%.12g mu=%.12g '
    'message=%s' % (
        solver.converged, solver.cycles, solver.nfev,
        solver.free_energy, solver.grand_potential, solver.residual_rms,
        (np.nan if solver.potential_residual_rms is None
         else solver.potential_residual_rms),
        solver.electron_number, solver.mu, solver.message),
    flush=True,
)
