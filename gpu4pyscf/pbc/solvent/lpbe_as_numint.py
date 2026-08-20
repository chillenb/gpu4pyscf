import ctypes


import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.fft as fft
from cupyx.scipy.sparse.linalg import LinearOperator, cg

from cupy.cuda.nvtx import RangePush, RangePop

import pyscf.pbc.gto as gto
from pyscf import lib
from pyscf.data import nist
from pyscf.pbc.dft.multigrid import multigrid

from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_gamma_point
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc.tools import pbc as pbc_tools
from gpu4pyscf.lib.cupy_helper import batched_vec_norm2, contract, tag_array

import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
import gpu4pyscf.pbc.dft.multigrid_v2 as multigrid_v2
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt
from gpu4pyscf.pbc.dft.multigrid_v2 import fft_in_place, ifft_in_place, evaluate_density_on_g_mesh, convert_xc_on_g_mesh_to_fock


class LPBEGridResult:
    """Persistent reciprocal/real-space data from one LPBE NumInt call.

    The arrays are snapshots rather than views of NumInt scratch storage.  In
    particular, ``vlocal_g`` uses the convention

        f_G = (cell.vol / ngrids) * fft(f_R)

    and is the complete density-dependent local KS potential (XC + Hartree +
    LPBE correction) immediately before its conversion to AO matrices.
    """

    __slots__ = (
        'vlocal_g', 'rho_g', 'cavity_r', 'eps_r', 'lpbe_mass_r',
        'lpbe_pot_guess',
    )

    def __init__(self, vlocal_g, rho_g, cavity_r, eps_r, lpbe_mass_r,
                 lpbe_pot_guess):
        self.vlocal_g = vlocal_g
        self.rho_g = rho_g
        self.cavity_r = cavity_r
        self.eps_r = eps_r
        self.lpbe_mass_r = lpbe_mass_r
        self.lpbe_pot_guess = lpbe_pot_guess


def shape_function(rhoR, sigma_k, nc_k, eps=1e-10):
    Z = cp.log(cp.maximum(rhoR.real, eps) / nc_k) * (1.0 / (np.sqrt(2) * sigma_k))
    S = 0.5 * cupyx.scipy.special.erfc(Z)
    Sprime = -(1/(np.sqrt(2*np.pi)*sigma_k)) * cp.exp(-Z**2) / cp.maximum(rhoR.real, eps)
    return S, Sprime

def vasp_dens_to_pyscf_dens(val):
    return val * (nist.BOHR ** 3)

def vasp_tau_to_pyscf_tau(val):
    """
    eV / A^2 -> Ha / Bohr^2
    """
    return val * (nist.BOHR ** 2) / nist.HARTREE2EV

def molar_to_au(conc):
    return conc * (nist.BOHR_SI**3 * 1000.0 * nist.AVOGADRO)

def molar_to_si(conc):
    return conc * nist.AVOGADRO * 1000.0

def debye_length_SI(ionic_strength, temperature, eps_r=1.0):
    """
    Debye length in meters. Ionic strength should be in particles per cubic meter.
    """
    eps0 = nist.E_CHARGE**2 / ( 2 * nist.ALPHA * nist.PLANCK * nist.LIGHT_SPEED_SI)
    return np.sqrt(eps0 * eps_r * nist.BOLTZMANN * temperature /
                   (2 * ionic_strength * nist.E_CHARGE**2))

def debye_length_au(ionic_strength, temperature, eps_r=1.0):
    """
    Debye length in Bohr. Ionic strength should be in particles per cubic Bohr.
    """
    eps0_au = 1 / (4*np.pi)
    boltzmann_ha = nist.BOLTZMANN / nist.HARTREE2J
    return np.sqrt(eps0_au * eps_r * boltzmann_ha * temperature /
                   (2 * ionic_strength))


def gradient_recip(F, Gv, out=None):
    """Compute the gradient of a function in reciprocal space.

    Parameters
    ----------
    F : ndarray
        The function values in reciprocal space.
    Gv : ndarray
        The reciprocal lattice vectors.

    Returns
    -------
    ndarray
        The gradient of the function in reciprocal space.
    """
    if out is None:
        grad_F = cp.empty((3,) + F.shape, dtype=np.complex128)
    else:
        grad_F = out
    for i in range(3):
        grad_F[i, ...] = 1j * Gv[..., i] * F
    return grad_F

def divergence_recip(Fv, Gv, out=None):
    """Compute the divergence of a vector function in reciprocal space.

    Parameters
    ----------
    Fv : ndarray
        The vector function values in reciprocal space.
    Gv : ndarray
        The reciprocal lattice vectors.

    Returns
    -------
    ndarray
        The divergence of the vector function in reciprocal space.
    """
    if out is None:
        div_F = cp.zeros(Fv.shape[1:], dtype=np.complex128)
    else:
        div_F = out
        div_F.fill(0.0)
    for i in range(3):
        div_F += 1j * Gv[..., i] * Fv[i, ...]
    return div_F


def pseudocore_density(cell, mesh):
    assert cell.dimension == 3
    Gv, (basex, basey, basez) = get_Gv_weights(cell, mesh)[:2]
    b = cell.reciprocal_vectors()
    coords = cell.atom_coords()
    rb = cp.asarray(coords.dot(b.T))
    SIx = cp.exp(-1j*rb[:,0,None] * basex)
    SIy = cp.exp(-1j*rb[:,1,None] * basey)
    SIz = cp.exp(-1j*rb[:,2,None] * basez)
    # G2 = contract('px,px->p', Gv, Gv)
    G2 = batched_vec_norm2(Gv)
    charges = cell.atom_charges()

    rhocoreG = cp.zeros(len(G2), dtype=np.complex128)

    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb not in cell._pseudo:
            continue

        if charges[ia] == 0:
            continue

        pp = cell._pseudo[symb]
        rloc = pp[1]

        # pure Gaussian density
        # with width rloc/2.5 and magnitude 1.0
        rcore = rloc / 2.5
        pcharge = 1.0


        SI = (SIx[ia,:,None,None] * SIy[ia,:,None] * SIz[ia]).ravel()
        G2_red = G2 * rcore**2
        SI *= cp.exp(-0.5*G2_red)
        rhocoreG += pcharge * SI
    return rhocoreG



def lpbe_inner(ni, rhoG, coul_kernelG, Gv, options=None, pot_guess=None):

    RangePush("lpbe_inner")

    if options is None:
        options = {}
    tol = options.get('tol', 1e-8)
    cav_smear = options.get('cav_smear', 0.6)
    eps        = options.get('eps', 1e-10)
    rel_permittivity = options.get('rel_permittivity', 78.4)

    cav_dens_cutoff = options.get('cav_dens_cutoff', vasp_dens_to_pyscf_dens(0.0025))
    cav_tension = options.get('cav_tension', vasp_tau_to_pyscf_tau(5.25e-4))

    has_electrolyte = options.get('has_electrolyte', True)
    temp_kelvin = options.get('temperature', 298.15)
    ionic_strength = options.get('ionic_strength', 1.0)
    debye_length = debye_length_au(molar_to_au(ionic_strength), temp_kelvin, eps_r=rel_permittivity)

    Gabs2 = cp.einsum('gi,gi->g', Gv, Gv)


    if has_electrolyte:
        ebkappa2 = rel_permittivity / (debye_length ** 2)
    else:
        ebkappa2 = 0.0

    mesh = ni.mesh
    ngrids = np.prod(mesh)
    cell = ni.cell
    vol = cell.vol
    weight = vol / ngrids
    log = logger.new_logger(cell)


    vpplocG = ni.get_vpplocG()
    pseudo_nucdensityG = Gabs2 * vpplocG * (-1.0 / (4*np.pi))
    charges = cell.atom_charges()
    tot_nuc_charge = np.sum(charges)
    pseudo_nucdensityG[0] = tot_nuc_charge
    pseudo_nucdensityR = pbc_tools.ifft(pseudo_nucdensityG.reshape(-1), mesh).real.reshape(*mesh) / weight

    rhoR = pbc_tools.ifft(rhoG.reshape(-1), mesh).real.reshape(*mesh) / weight
    # Charge sign convention is that electrons are positive.
    solute_chargeR = rhoR - pseudo_nucdensityR

    nelec_by_integration = cp.sum(rhoR) * vol / ngrids
    nuc_charge_by_integration = cp.sum(pseudo_nucdensityR) * vol / ngrids
    qsol = nelec_by_integration - nuc_charge_by_integration

    pseudocore_densityG = pseudocore_density(cell, mesh)
    pseudocore_densityR = pbc_tools.ifft(pseudocore_densityG.reshape(-1), mesh).real.reshape(*mesh) / weight

    RangePush("shape_function")
    S, Sprime = shape_function(rhoR + pseudocore_densityR, cav_smear, cav_dens_cutoff)
    RangePop()

    eps_r_field = 1. + (rel_permittivity - 1.) * S


    log.debug(f"Nelec by integration: {nelec_by_integration}")
    log.debug(f"Nuclear charge by integration of pseudo_nucdensityR: {nuc_charge_by_integration}")
    log.debug(f"Total solute charge by integration: {qsol}")

    Svol = cp.sum(S) * vol / ngrids
    Svol_ang = Svol * (nist.BOHR ** 3)
    cell_vol_ang = cell.vol * (nist.BOHR ** 3)
    log.debug(f"Svol: {Svol_ang} Ang^3")
    log.debug(f"Cell vol: {cell_vol_ang} Ang^3")
    log.debug(f"ebkappa2: {ebkappa2:.3e} 1/Bohr^2, debye length: {debye_length:.3f} Bohr")
    log.debug(f"ebkappa2: {ebkappa2 / (nist.BOHR ** 2):.3e} 1/Angstrom^2, debye length: {debye_length * nist.BOHR:.3f} Angstrom")

    # solve the equation
    # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
    # by preconditioned conjugate gradient.
    # Preconditioner = poisson.
    def make_aop(Skappa2):
        def Aop(phiG):
            # No scaling by weight of the intermediates is necessary thanks to linearity
            grad_phiG = gradient_recip(phiG, Gv).reshape(3, *mesh)
            grad_phiR = pbc_tools.ifft(grad_phiG.reshape(3, -1), mesh).reshape(3, *mesh)
            eps_grad_phiR = eps_r_field * grad_phiR
            eps_grad_phiG = pbc_tools.fft(eps_grad_phiR.reshape(3, -1), mesh)
            div_eps_grad_phiG = divergence_recip(eps_grad_phiG.reshape(3, -1), Gv)
            phi_R = pbc_tools.ifft(phiG, mesh).reshape(*mesh)
            debye_term_real = Skappa2 * phi_R.reshape(*mesh)
            debye_term_G = pbc_tools.fft(debye_term_real.reshape(-1), mesh)
            return -(div_eps_grad_phiG - debye_term_G).reshape(-1)
        return Aop

    mean_S = cp.mean(S.reshape(-1))
    mean_eps_r = 1. + (rel_permittivity - 1.) * mean_S
    mean_ebkappa2 = mean_S * ebkappa2

    if ebkappa2 == 0:
        yukawa_kernel = coul_kernelG
    else:
        yukawa_kernel = 4.0 * np.pi / (mean_eps_r * Gabs2 + mean_ebkappa2)

    # sqrt_yukawa_kernel = cp.sqrt(yukawa_kernel)
    # one_over_epsr = 1.0 / eps_r_field.reshape(-1)

    def Mprecond(phiG):
        precond_phiG = yukawa_kernel * phiG
        return precond_phiG.reshape(-1)


    t0 = log.init_timer()

    A = LinearOperator((ngrids, ngrids), matvec=make_aop(S*ebkappa2))
    M = LinearOperator((ngrids, ngrids), matvec=Mprecond)
    rhs = pbc_tools.fft(4*np.pi*solute_chargeR.reshape(-1), mesh) * weight

    niter = 0
    def callback(x):
        nonlocal niter
        niter += 1

    RangePush("lpbe_cg_solve")
    # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
    solution_phi_G, info = cg(A, rhs, M=M, x0=pot_guess, tol=tol, maxiter=400, callback=callback)

    RangePop()

    if info != 0:
        log.warn(f"Conjugate gradient did not converge: info={info}")

    log.debug(f"Number of CG iterations: {niter}")

    t1 = log.timer("LPBE CG solve", *t0)

    t2 = log.init_timer()


    RangePush("lpbe_postprocess")
    solution_phi_R = pbc_tools.ifft(solution_phi_G.reshape(-1), mesh).real.reshape(*mesh) / weight

    # rho_ion_R = solution_phi_R * S * (ebkappa2 / (4*np.pi))


    # compute solvation potential.
    solute_chargeG = pbc_tools.fft(solute_chargeR.reshape(-1), mesh).reshape(-1) * weight
    vac_coulomb_potentialG = coul_kernelG * solute_chargeG

    vac_coulomb_potentialG = vac_coulomb_potentialG.reshape(-1)
    
    # The next line ensures that the G=0 component of the vacuum Coulomb potential
    # is consistent with vpplocG.
    # vpplocG has a non-zero G=0 component.

    vac_coulomb_potentialG[0] += vpplocG.reshape(-1)[0]

    vac_coulomb_potentialR = pbc_tools.ifft(vac_coulomb_potentialG.reshape(-1), mesh).real.reshape(*mesh) / weight


    solvation_potentialR = solution_phi_R - vac_coulomb_potentialR

    solvation_potentialG = pbc_tools.fft(solvation_potentialR.reshape(-1), mesh).reshape(-1) * weight

    grad_solution_phiR = pbc_tools.ifft(gradient_recip(solution_phi_G, Gv), mesh).real / weight

    S_grad_solution_phiR = S * grad_solution_phiR.reshape(3, *mesh)
    div_S_grad_solution_phiG = divergence_recip( pbc_tools.fft(S_grad_solution_phiR.reshape(3, -1) * weight, mesh), Gv)
    div_S_grad_solution_phiR = pbc_tools.ifft(div_S_grad_solution_phiG.reshape(-1), mesh).real / weight
    diel_bound_charge_density_R = div_S_grad_solution_phiR * ( (rel_permittivity - 1.) / (4*np.pi) )
    del S_grad_solution_phiR, div_S_grad_solution_phiG, div_S_grad_solution_phiR

    # Ionic and dielectric components of solvation potential.
    lambdalq_ion = - 1.0/(8*np.pi) * ebkappa2 * solution_phi_R.reshape(-1)**2
    lambdalq_diel = -1.0/(8*np.pi) * (rel_permittivity - 1.) * cp.einsum('ng, ng ->g', grad_solution_phiR, grad_solution_phiR)

    vion_r = Sprime.reshape(-1) * lambdalq_ion
    vdiel_r = Sprime.reshape(-1) * lambdalq_diel

    # These terms should not be added to the free energy.
    Eion = cp.einsum('g, g ->', lambdalq_ion, S.reshape(-1)) * vol / ngrids
    Ediel = cp.einsum('g, g ->', lambdalq_diel, S.reshape(-1)) * vol / ngrids

    # The coulomb correction energy is just the difference between the coulomb energy
    # in solution and in vacuum.
    E_coul_corr = 0.5 * cp.sum( (solution_phi_R - vac_coulomb_potentialR) * solute_chargeR ) * weight

    # vpplocG[0]/vol is the constant local-pseudopotential alignment in
    # V_vac.  In 1/2 <rho_sol, phi - V_vac> this fixed one-body term is
    # counted only by half, whereas the vacuum DFT energy contains its full
    # electron--nuclear contribution.  Subtract the missing half here so
    # dE_coul_corr/drho is the full phi - V_vac response above.  The base
    # vacuum Fock potential plus that response is then phi, whose zero in
    # the empty region supplies the physical vacuum reference for mu.
    vacuum_alignment = vpplocG.reshape(-1)[0].real / vol
    E_coul_corr -= 0.5 * vacuum_alignment * qsol

    # Vacuum alignment in z-direction. This is important when there is no electrolyte.
    rhoG_smoothed = rhoG * cp.exp(-100.0 * (Gabs2) * 0.5)
    rhoR_smoothed = pbc_tools.ifft(rhoG_smoothed.reshape(-1), mesh).real.reshape(*mesh) * weight
    rhoR_z = rhoR_smoothed.mean(axis=(0, 1))
    dens_min_idx = cp.argmin(rhoR_z)
    vacpot_at_zmin = vac_coulomb_potentialR.reshape(mesh).real.mean(axis=(0, 1))[dens_min_idx]
    solpot_at_zmin = solution_phi_R.reshape(mesh).real.mean(axis=(0, 1))[dens_min_idx]

    del rhoG_smoothed, rhoR_smoothed, rhoR_z




    # V_\mathrm{cav} = \tau \partial_{\rho} S(r) \left( \frac{\nabla^{2}\rho}{|\nabla\rho|} - 
    #                  \frac{1}{|\nabla\rho|^{3}}(\nabla\rho)^{\mathrm{t}}\mathbf{H}_{\rho}(\nabla\rho) \right)
    # tau is self.cav_tension.

    # Cavitation potential.

    grad_rho_r = pbc_tools.ifft(gradient_recip(rhoG, Gv), mesh).real.reshape(3, *mesh) / weight

    lap_rho_r = pbc_tools.ifft((-Gabs2 * rhoG).reshape(-1), mesh).real.reshape(*mesh) / weight

    # grad_hess_grad_r = (nabla rho)^t H(rho) (nabla rho)
    grad_hess_grad_r = cp.zeros(mesh, dtype=cp.float64)

    for i in range(3):
        for j in range(3):
            hij_g = -(Gv[:, i] * Gv[:, j]) * rhoG
            hij_r = pbc_tools.ifft(hij_g.reshape(-1), mesh).real.reshape(*mesh) / weight
            grad_hess_grad_r += grad_rho_r[i] * hij_r * grad_rho_r[j]

    grad_abs_r = cp.sqrt(cp.einsum('i...,i...->...', grad_rho_r, grad_rho_r))
    grad_abs_safe_r = cp.maximum(grad_abs_r, eps)

    vcav_r = cav_tension * Sprime * (
        lap_rho_r / grad_abs_safe_r
        - grad_hess_grad_r / (grad_abs_safe_r ** 3)
    )
    vcav_r = vcav_r.reshape(-1)

    del lap_rho_r, grad_hess_grad_r, grad_abs_safe_r, grad_rho_r, hij_g, hij_r

    vcorr_r = vion_r + vdiel_r + vcav_r

    vcorr_g = solvation_potentialG + pbc_tools.fft(vcorr_r.reshape(-1), mesh).reshape(-1) * weight

    surf_area = cp.sum( (-Sprime * grad_abs_r).reshape(-1) ) * vol / ngrids
    Ecav = cav_tension * surf_area

    log.debug(f"Ecav: {Ecav:.3e} Hartree ({Ecav*nist.HARTREE2EV:.3e} eV)")
    log.debug(f"Coulomb correction energy: {E_coul_corr:.3e} Hartree ({E_coul_corr*nist.HARTREE2EV:.3e} eV)")
    log.debug(f"Vacuum potential in empty space: {vacpot_at_zmin:.3e} Hartree ({vacpot_at_zmin*nist.HARTREE2EV:.3e} eV)")
    log.debug(f"Phi in empty space: {solpot_at_zmin:.3e} Hartree ({solpot_at_zmin*nist.HARTREE2EV:.3e} eV)")
    log.debug(f"Surface area: {surf_area:.3f} Bohr^2")
    log.debug(f"Eion: {Eion:.3e} Hartree ({Eion*nist.HARTREE2EV:.3e} eV)")
    log.debug(f"Ediel: {Ediel:.3e} Hartree ({Ediel*nist.HARTREE2EV:.3e} eV)")

    results = {
        'vcorr_g': vcorr_g,
        'Eion': Eion,
        'Ediel': Ediel,
        'Ecav': Ecav,
        'E_coul_corr': E_coul_corr,
        'pot_guess': solution_phi_G,
        'cavity_r': S,
        'eps_r': eps_r_field,
        'mass_r': S * ebkappa2,
    }

    RangePop()
    RangePop()

    return results



def nr_rks_lpbe(ni, cell, grids, xc_code, dm_kpts, relativity=0, hermi=1,
           kpts=None, kpts_band=None, with_j=True, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.
        with_j : bool
            Whether to add the Coulomb matrix into the XC matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    RangePush("nr_rks_lpbe")
    cell = ni.cell
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    xc_type = ni._xc_type(xc_code)
    if ni.sorted_gaussian_pairs is None:
        ni.build(xc_type)

    if not with_j:
        raise ValueError("Why are you calling this function if you don't want electrostatics?")

    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape(-1, 3)
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    dms = None
    assert nset == 1

    mesh = ni.mesh
    ngrids = np.prod(mesh)

    RangePush("evaluate_density_on_g_mesh")
    density = evaluate_density_on_g_mesh(ni, dm_kpts, kpts, xc_type)
    rho_sf = density[0, 0]
    # ``ifft_in_place`` below reuses ``density`` storage, so retain the
    # reciprocal density now for the persistent grid contract.
    rho_g = rho_sf.copy()
    RangePop()

    Gv = pbc_tools.get_Gv(cell, mesh)
    coulomb_kernel_on_g_mesh = pbc_tools.get_coulG(cell, Gv=Gv)
    coulomb_on_g_mesh = rho_sf * coulomb_kernel_on_g_mesh
    coulomb_energy = complex(rho_sf.conj().dot(coulomb_on_g_mesh).get())
    coulomb_energy = (0.5 / cell.vol) * coulomb_energy
    log.debug("Multigrid Coulomb energy %s", coulomb_energy)
    t0 = log.timer("coulomb", *t0)
    weight = cell.vol / ngrids


    # LPBE
    ni.get_vpplocG()  # Invalidate the potential guess if the mesh changed.
    lpbe_res = lpbe_inner(
        ni, rho_sf, coulomb_kernel_on_g_mesh, Gv,
        options=ni.options, pot_guess=ni.pot_guess)
    ni.pot_guess = lpbe_res['pot_guess']
    vcorr_g = lpbe_res['vcorr_g']
    Ecorr = float(cp.real(
        lpbe_res['E_coul_corr'] + lpbe_res['Ecav']).get())
    coulomb_energy += Ecorr

    density = ifft_in_place(density.reshape(-1, *mesh)).real.reshape(-1, ngrids)
    n_electrons = float(density[0].sum().real.get())
    density /= weight

    RangePush("eval_xc_eff")
    # eval_xc_eff supports float64 only
    density = cp.asarray(density, dtype=np.float64, order='C')
    xc_for_energy, xc_for_fock = ni.eval_xc_eff(
        xc_code, density, deriv=1, xctype=xc_type, spin=0
    )[:2]
    RangePop()

    rho_sf = density[0].real
    xc_energy_sum = float(rho_sf.dot(xc_for_energy.ravel()).get()) * weight

    # To reduce the memory usage, we reuse the xc_for_fock name.
    # Now xc_for_fock represents xc on G space
    xc_for_fock *= weight
    xc_for_fock = fft_in_place(xc_for_fock.reshape(-1, *mesh)).reshape(-1, ngrids)

    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    if xc_type == "LDA" or xc_type == 'HF':
        pass
    elif xc_type == "GGA":
        xc_for_fock = (
            xc_for_fock[0] - contract("gp, pg -> p", xc_for_fock[1:4], Gv) * 1j
        )
        xc_for_fock = xc_for_fock.reshape((-1, ngrids))
    elif xc_type == "MGGA":
        xc_for_fock[0] -= contract("gp, pg -> p", xc_for_fock[1:4], Gv) * 1j
        xc_for_fock = cp.concatenate([
            xc_for_fock[0].reshape((-1, ngrids)),
            xc_for_fock[4].reshape((-1, ngrids)),
        ], axis = 0)
    else:
        raise ValueError(f"Incorrect xc_type = {xc_type}")

    # The LPBE potential should be considered as a correction to "J"
    if with_j:
        xc_for_fock[0] += coulomb_on_g_mesh + vcorr_g

    # Reciprocal GGA differentiation can leave imaginary values on Nyquist
    # planes.  The existing AO conversion projects through ``ifft(...).real``;
    # expose that same physical real scalar field, in normalized G space, so
    # potential mixing starts from an exactly Hermitian representation.
    vlocal_g = pbc_tools.fft(
        pbc_tools.ifft(xc_for_fock[0], mesh).real.reshape(-1), mesh)
    grid_result = LPBEGridResult(
        vlocal_g=vlocal_g.reshape(-1).copy(),
        rho_g=rho_g,
        cavity_r=lpbe_res['cavity_r'].copy(),
        eps_r=lpbe_res['eps_r'].copy(),
        lpbe_mass_r=lpbe_res['mass_r'].copy(),
        lpbe_pot_guess=lpbe_res['pot_guess'].copy(),
    )

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    RangePush("convert_xc_on_g_mesh_to_fock")
    veff = convert_xc_on_g_mesh_to_fock(ni, xc_for_fock, hermi, kpts_band, with_tau = (xc_type == "MGGA"))
    RangePop()
    veff = _format_jks(veff, dm_kpts, input_band, kpts)
    veff = tag_array(
        veff, ecoul=coulomb_energy, exc=xc_energy_sum,
        lpbe_grid=grid_result)
    t0 = log.timer("xc", *t0)
    RangePop()
    return n_electrons, xc_energy_sum, veff


class LPBEMultiGridNumInt(MultiGridNumInt):
    def __init__(self, cell, **options):
        super().__init__(cell)
        self.options = options
        self.vpplocG = None
        self.pot_guess = None
        self._lpbe_mesh = None

    def reset(self, cell=None):
        super().reset(cell)
        if cell is not None:
            self.mesh = cell.mesh
        self.vpplocG = None
        self.pot_guess = None
        self._lpbe_mesh = None
        return self

    def get_vpplocG(self):
        mesh = tuple(self.mesh)
        if self.vpplocG is None or self._lpbe_mesh != mesh:
            vpplocG = multigrid_v1.eval_vpplocG(self.cell, self.mesh)
            self.vpplocG = vpplocG
            self.pot_guess = None
            self._lpbe_mesh = mesh
        return self.vpplocG

    def local_potential_to_ao(self, vlocal_g, kpts=None, hermi=1):
        """Convert one scalar reciprocal-space local potential to AO blocks.

        Parameters follow :func:`convert_xc_on_g_mesh_to_fock`.  The returned
        array always has shape ``(nkpts, nao, nao)``; unlike ``nr_rks`` it is
        independent of the input density-matrix rank.
        """
        vlocal_g = cp.asarray(vlocal_g)
        ngrids = int(np.prod(self.mesh))
        if vlocal_g.size != ngrids:
            raise ValueError(
                'local potential has %d values; mesh requires %d' %
                (vlocal_g.size, ngrids))
        if not bool(cp.all(cp.isfinite(vlocal_g)).item()):
            raise FloatingPointError('local potential contains nonfinite values')
        if self.sorted_gaussian_pairs is None:
            self.build('LDA')
        if kpts is not None:
            kpts = np.asarray(kpts).reshape(-1, 3)
        matrices = convert_xc_on_g_mesh_to_fock(
            self, vlocal_g.reshape(-1), hermi=hermi, kpts=kpts,
            with_tau=False)
        return matrices[0]

    nr_rks = nr_rks_lpbe
    nr_uks = NotImplemented


def multigrid_lpbe(mf, mesh=None, **kwargs):
    mf2 = mf.copy()
    mf2._numint = LPBEMultiGridNumInt(mf2.cell, **kwargs)
    if mesh is not None:
        mf2._numint.mesh = mesh
    return mf2
