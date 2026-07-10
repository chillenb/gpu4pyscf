import ctypes


import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.fft as fft


import pyscf.pbc.gto as gto
from pyscf import lib
from pyscf.data import nist
from pyscf.pbc.dft.multigrid import multigrid

from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_gamma_point
from gpu4pyscf.dft import numint
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc.tools import pbc as pbc_tools
from gpu4pyscf.lib.cupy_helper import contract, tag_array

import gpu4pyscf.pbc.dft.multigrid as multigrid_v1
import gpu4pyscf.pbc.dft.multigrid_v2 as multigrid_v2
from gpu4pyscf.pbc.dft.multigrid_v2 import MultiGridNumInt
from gpu4pyscf.pbc.dft.multigrid_v2 import fft_in_place, ifft_in_place


def shape_function(rhoR, sigma_k, nc_k, eps=1e-10):
    Z = cp.log(cp.maximum(rhoR.real, eps) / nc_k) * (1.0 / (np.sqrt(2) * sigma_k))
    S = 0.5 * cupyx.scipy.special.erfc(Z)
    Sprime = (1/(np.sqrt(2*np.pi)*sigma_k)) * cp.exp(-Z**2) / cp.maximum(rhoR.real, eps)
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


class PeriodicLPBE(lib.StreamObject):
    def __init__(self, cell, kpts, mesh=None, **kwargs):
        self.cell = cell
        self.kpts = kpts
        self.verbose = cell.verbose

        if mesh is None:
            mesh = cell.mesh
        self.mesh = mesh
        self.ni = MultiGridNumInt(self.cell)
        self.ni.mesh = mesh

        self.Gv = None
        self.Gabs2 = None
        self.coul_kernel = None

        self.pot_guess = None
        self.ncalls = 0
        self.nskip = 0
        self.plot_results = False
        self.chkfile = None
        self.plot_filestem = "lpbe_results"

        self.extra_screening_list = None

        self.is_built = False
        self.tol = 1e-8
        self.frozen = False
        self.debug_checks = kwargs.get('debug_checks', False)

        self.eps        = kwargs.get('eps', 1e-10)
        self.cav_smear  = kwargs.get('cav_smear', 0.6)
        self.cav_dens_cutoff = kwargs.get('cav_dens_cutoff', vasp_dens_to_pyscf_dens(0.0025))
        self.cav_tension = kwargs.get('cav_tension', vasp_tau_to_pyscf_tau(5.25e-4))
        self.rel_permittivity = kwargs.get('rel_permittivity', 78.4)

        self.has_electrolyte = True
        self.debye_length = debye_length_au(molar_to_au(kwargs.get('ionic_strength', 1.0)), 298.15, eps_r=self.rel_permittivity)

        self.vpplocG = None


    def dump_flags(self, verbose=None):
        logger.info(self, "******** Periodic LPBE flags ********")
        logger.info(self, f"  kpts: {self.kpts}")
        logger.info(self, f"  tol: {self.tol}")
        logger.info(self, f"  cav_smear: {self.cav_smear}")
        logger.info(self, f"  cav_dens_cutoff: {self.cav_dens_cutoff}")
        logger.info(self, f"  cav_tension: {self.cav_tension}")
        logger.info(self, f"  rel_permittivity: {self.rel_permittivity}")
        logger.info(self, f"  debye_length: {self.debye_length}")
        return self
    
    def build(self):
        if not self.is_built:
            self.is_built = True
            self.ni.build()
            logger.info(self, f"LPBE: using mesh {self.ni.mesh}")
            self.Gv = pbc_tools._get_Gv(self.cell, self.mesh)
            self.coul_kernel = pbc_tools.get_coulG(self.cell, Gv=self.Gv)
            self.Gabs2 = cp.einsum('gi,gi->g', self.Gv, self.Gv)


    def get_vpplocG(self):
        if self.vpplocG is None:
            vpplocG = multigrid_v1.eval_vpplocG(self.cell, self.mesh)
            self.vpplocG = vpplocG
        return self.vpplocG

    def get_pseudo_nucdensity(self):
        """
        Nuclear charge density in reciprocal space.
        """
        vpplocG = self.get_vpplocG()
        nucdensity = self.Gabs2 * vpplocG * (-1.0 / (4*np.pi))
        charges = self.cell.atom_charges()
        tot_nuc_charge = np.sum(charges)
        nucdensity[0] = tot_nuc_charge
        return nucdensity

    def get_rhoG(self, dm_kpts):
        dms = _format_dms(dm_kpts, self.kpts)
        return multigrid_v2.evaluate_density_on_g_mesh(self.ni, dms, self.kpts)

    def kernel_detail(self, dm_kpts, tol=None):
        self.build()
        log = logger.new_logger(self, verbose=self.verbose)
        if tol is None:
            tol = self.tol
        tol = max(tol, self.tol)

        mesh = self.ni.mesh
        ngrids = np.prod(mesh)
        cell = self.cell
        vol = cell.vol
        kpts = self.kpts
        dms = _format_dms(dm_kpts, kpts)

        weight = vol / ngrids

        # The usual convention in PySCF for normalization
        # is that rhoG[0] = nelec.
        # This means rho(r) = 1/Omega sum_G rho[G] exp(i G · r).
        #                   = ifft(rhoG) / weight.
        # Conversely, rho[G] = weight * fft(rhoR).
        rhoG = self.get_rhoG(dms)
        rhoR = pbc_tools.ifft(rhoG.reshape(-1), mesh).real.reshape(*self.mesh) / weight

        pseudo_nucdensityG = self.get_pseudo_nucdensity()
        pseudo_nucdensityR = pbc_tools.ifft(pseudo_nucdensityG.reshape(-1), mesh).real.reshape(*self.mesh) / weight
        # This should integrate to the total nuclear charge.

        # Charge sign convention is that electrons are positive.
        solute_chargeR = rhoR - pseudo_nucdensityR

        nelec_by_integration = cp.sum(rhoR) * vol / ngrids
        nuc_charge_by_integration = cp.sum(pseudo_nucdensityR) * vol / ngrids
        qsol = nelec_by_integration - nuc_charge_by_integration

        if self.debug_checks:
            log.info(f"Nelec by integration: {nelec_by_integration}")
            log.info(f"Nelec from recip space: {rhoG[0].real.get()*weight}")
            log.info(f"Nuclear charge by integration of pseudo_nucdensityR: {nuc_charge_by_integration}")
            log.info(f"Total solute charge by integration: {qsol}")

        S, Sprime = shape_function(rhoR + pseudo_nucdensityR, self.cav_smear, self.cav_dens_cutoff)

        eps_r_field = 1. + (self.rel_permittivity - 1.) * S

        if self.has_electrolyte:
            ebkappa2 = self.rel_permittivity / (self.debye_length ** 2)
        else:
            ebkappa2 = 0.0
        # Used to damp charge sloshing.
        if self.extra_screening_list is not None and self.ncalls < len(self.extra_screening_list):
            extrakappa2 = self.extra_screening_list[self.ncalls]
        else:
            extrakappa2 = 0.0

        if self.debug_checks:
            Svol = cp.sum(S) * vol / ngrids
            Svol_ang = Svol * (nist.BOHR ** 3)
            cell_vol_ang = cell.vol * (nist.BOHR ** 3)
            log.info(f"Svol: {Svol_ang} Ang^3")
            log.info(f"Cell vol: {cell_vol_ang} Ang^3")
            log.info(f"ebkappa2: {ebkappa2:.3e} 1/Bohr^2, debye length: {self.debye_length:.3f} Bohr")
            log.info(f"ebkappa2: {ebkappa2 / (nist.BOHR ** 2):.3e} 1/Angstrom^2, debye length: {self.debye_length * nist.BOHR:.3f} Angstrom")

        # solve the equation
        # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
        # by preconditioned conjugate gradient.
        # Preconditioner = poisson.
        def make_aop(Skappa2):
            def Aop(phiG):
                # No scaling by weight of the intermediates is necessary thanks to linearity
                grad_phiG = gradient_recip(phiG, self.Gv).reshape(3, *mesh)
                grad_phiR = pbc_tools.ifft(grad_phiG.reshape(3, -1), mesh).reshape(3, *mesh)
                eps_grad_phiR = eps_r_field * grad_phiR
                eps_grad_phiG = pbc_tools.fft(eps_grad_phiR.reshape(3, -1), mesh)
                div_eps_grad_phiG = divergence_recip(eps_grad_phiG.reshape(3, -1), self.Gv)
                phi_R = pbc_tools.ifft(phiG, mesh).reshape(*mesh)
                debye_term_real = Skappa2 * phi_R.reshape(*mesh)
                debye_term_G = pbc_tools.fft(debye_term_real.reshape(-1), mesh)
                return -(div_eps_grad_phiG - debye_term_G).reshape(-1)
            return Aop

        mean_S = cp.mean(S.reshape(-1))
        mean_eps_r = (self.rel_permittivity - 1.) * mean_S
        mean_ebkappa2 = mean_S * ebkappa2

        if ebkappa2 == 0:
            yukawa_kernel = self.coul_kernel # laplacian operator
        else:
            yukawa_kernel = 4.0 * np.pi / (mean_eps_r * self.Gabs2 + mean_ebkappa2)

        sqrt_yukawa_kernel = cp.sqrt(yukawa_kernel)
        one_over_epsr = 1.0 / eps_r_field.reshape(-1)

        def Mprecond(phiG):
            precond_phiG = yukawa_kernel * phiG
            return precond_phiG.reshape(-1)

        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        t0 = log.init_timer()

        A = LinearOperator((ngrids, ngrids), matvec=make_aop(extrakappa2 + S*ebkappa2))
        M = LinearOperator((ngrids, ngrids), matvec=Mprecond)
        rhs = pbc_tools.fft(4*np.pi*solute_chargeR.reshape(-1), mesh) * weight

        niter = 0
        def callback(x):
            nonlocal niter
            niter += 1

        # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
        solution_phi_G, info = cg(A, rhs, M=M, x0=self.pot_guess, tol=self.tol, maxiter=400, callback=callback)

        if info != 0:
            logger.warn(self, f"Conjugate gradient did not converge: info={info}")

        log.info(f"Number of CG iterations: {niter}")

        t1 = log.timer("LPBE CG solve", *t0)

        t2 = log.init_timer()

        solution_phi_R = pbc_tools.ifft(solution_phi_G.reshape(-1), mesh).real.reshape(*mesh) / weight

        self.pot_guess = solution_phi_G

        rho_ion_R = solution_phi_R * S * (ebkappa2 / (4*np.pi))


        # compute solvation potential.
        solute_chargeG = pbc_tools.fft(solute_chargeR.reshape(-1), mesh).reshape(-1) * weight
        vac_coulomb_potentialG = self.coul_kernel * solute_chargeG

        vac_coulomb_potentialG = vac_coulomb_potentialG.reshape(-1)
        
        # The next line ensures that the G=0 component of the vacuum Coulomb potential
        # is consistent with vpplocG.
        # vpplocG has a non-zero G=0 component.

        vac_coulomb_potentialG[0] += self.vpplocG.reshape(-1)[0]

        vac_coulomb_potentialR = pbc_tools.ifft(vac_coulomb_potentialG.reshape(-1), mesh).real.reshape(*mesh) / weight


        solvation_potentialR = solution_phi_R - vac_coulomb_potentialR

        solvation_potentialG = pbc_tools.fft(solvation_potentialR.reshape(-1), mesh).reshape(-1) * weight

        grad_solution_phiR = pbc_tools.ifft(gradient_recip(solution_phi_G, self.Gv), mesh).real / weight

        S_grad_solution_phiR = S * grad_solution_phiR.reshape(3, *mesh)
        div_S_grad_solution_phiG = divergence_recip( pbc_tools.fft(S_grad_solution_phiR.reshape(3, -1) * weight, mesh), self.Gv)
        div_S_grad_solution_phiR = pbc_tools.ifft(div_S_grad_solution_phiG.reshape(-1), mesh).real / weight
        diel_bound_charge_density_R = div_S_grad_solution_phiR * ( (self.rel_permittivity - 1.) / (4*np.pi) )
        del S_grad_solution_phiR, div_S_grad_solution_phiG, div_S_grad_solution_phiR

        # Ionic and dielectric components of solvation potential.
        lambdalq_ion = - 1.0/(8*np.pi) * ebkappa2 * solution_phi_R.reshape(-1)**2
        lambdalq_diel = -1.0/(8*np.pi) * (self.rel_permittivity - 1.) * cp.einsum('ng, ng ->g', grad_solution_phiR, grad_solution_phiR)

        # shape_function returns the positive magnitude -dS/drho.  The
        # dielectric and ionic response terms require dS/drho itself.
        vion_r = -Sprime.reshape(-1) * lambdalq_ion
        vdiel_r = -Sprime.reshape(-1) * lambdalq_diel

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
        vacuum_alignment = self.vpplocG.reshape(-1)[0].real / vol
        E_coul_corr -= 0.5 * vacuum_alignment * qsol

        # Vacuum alignment in z-direction. This is important when there is no electrolyte.
        rhoG_smoothed = rhoG * cp.exp(-100.0 * (self.Gabs2) * 0.5)
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

        grad_rho_r = pbc_tools.ifft(gradient_recip(rhoG, self.Gv), mesh).real.reshape(3, *mesh) / weight

        lap_rho_r = pbc_tools.ifft((-self.Gabs2 * rhoG).reshape(-1), mesh).real.reshape(*mesh) / weight

        # grad_hess_grad_r = (nabla rho)^t H(rho) (nabla rho)
        grad_hess_grad_r = cp.zeros(mesh, dtype=cp.float64)

        for i in range(3):
            for j in range(3):
                hij_g = -(self.Gv[:, i] * self.Gv[:, j]) * rhoG
                hij_r = pbc_tools.ifft(hij_g.reshape(-1), mesh).real.reshape(*mesh) / weight
                grad_hess_grad_r += grad_rho_r[i] * hij_r * grad_rho_r[j]

        grad_abs_r = cp.sqrt(cp.einsum('i...,i...->...', grad_rho_r, grad_rho_r))
        grad_abs_safe_r = cp.maximum(grad_abs_r, self.eps)

        vcav_r = -self.cav_tension * Sprime * (
            lap_rho_r / grad_abs_safe_r
            - grad_hess_grad_r / (grad_abs_safe_r ** 3)
        )
        vcav_r = vcav_r.reshape(-1)

        del lap_rho_r, grad_hess_grad_r, grad_abs_safe_r, grad_rho_r, hij_g, hij_r

        vcorr_r = vion_r + vdiel_r + vcav_r

        vcorr_g = solvation_potentialG + pbc_tools.fft(vcorr_r.reshape(-1), mesh).reshape(-1) * weight
    
        t5 = log.init_timer()
        vcorr_mat = multigrid_v2.convert_xc_on_g_mesh_to_fock(self.ni, vcorr_g, hermi=1, kpts=self.kpts)
        if getattr(self, 'debug_components', False):
            component_g = {
                'electrostatic': solvation_potentialG,
                'ionic': pbc_tools.fft(vion_r.reshape(-1), mesh).reshape(-1) * weight,
                'dielectric': pbc_tools.fft(vdiel_r.reshape(-1), mesh).reshape(-1) * weight,
                'cavitation': pbc_tools.fft(vcav_r.reshape(-1), mesh).reshape(-1) * weight,
            }
            self.debug_v_components = {
                key: multigrid_v2.convert_xc_on_g_mesh_to_fock(
                    self.ni, value, hermi=1, kpts=self.kpts)
                for key, value in component_g.items()
            }
        log.timer("convert_xc_on_g_mesh_to_fock", *t5)


        surf_area = cp.sum( (Sprime * grad_abs_r).reshape(-1) ) * vol / ngrids
        Ecav = self.cav_tension * surf_area

        if self.debug_checks:
            log.info(f"Ecav: {Ecav:.3e} Hartree ({Ecav*nist.HARTREE2EV:.3e} eV)")
            log.info(f"Coulomb correction energy: {E_coul_corr:.3e} Hartree ({E_coul_corr*nist.HARTREE2EV:.3e} eV)")
            log.info(f"Vacuum potential in empty space: {vacpot_at_zmin:.3e} Hartree ({vacpot_at_zmin*nist.HARTREE2EV:.3e} eV)")
            log.info(f"Phi in empty space: {solpot_at_zmin:.3e} Hartree ({solpot_at_zmin*nist.HARTREE2EV:.3e} eV)")
            log.info(f"Surface area: {surf_area:.3f} Bohr^2")
            log.info(f"Eion: {Eion:.3e} Hartree ({Eion*nist.HARTREE2EV:.3e} eV)")
            log.info(f"Ediel: {Ediel:.3e} Hartree ({Ediel*nist.HARTREE2EV:.3e} eV)")

        results = {
            'vcorr_mat': vcorr_mat,
            'Eion': Eion,
            'Ediel': Ediel,
            'Ecav': Ecav,
            'E_coul_corr': E_coul_corr,
        }



        if self.plot_results or (self.chkfile is not None):
            mesh = self.mesh
            z = np.arange(mesh[2]) * self.cell.lattice_vectors(unit='A')[2, 2] / mesh[2]
            solution_phi_z = solution_phi_R.reshape(mesh).mean(axis=(0, 1))
            solvation_potential_z = solvation_potentialR.reshape(mesh).mean(axis=(0, 1))
            vac_coulomb_potential_z = vac_coulomb_potentialR.reshape(mesh).mean(axis=(0, 1))
            vcorr_z = vcorr_r.reshape(mesh).mean(axis=(0, 1))
            rho_z = rhoR.reshape(mesh).mean(axis=(0, 1))
            pseudo_nucdensity_z = pseudo_nucdensityR.reshape(mesh).mean(axis=(0, 1))
            vion_z = vion_r.reshape(mesh).mean(axis=(0, 1))
            vdiel_z = vdiel_r.reshape(mesh).mean(axis=(0, 1))
            vcav_z = vcav_r.reshape(mesh).mean(axis=(0, 1))
            rhoion_z = rho_ion_R.reshape(mesh).mean(axis=(0, 1))
            rhodiel_z = diel_bound_charge_density_R.reshape(mesh).mean(axis=(0, 1))
            S_z = S.reshape(mesh).mean(axis=(0, 1))
            Sprime_z = Sprime.reshape(mesh).mean(axis=(0, 1))
            Sphi_z = (solution_phi_R * S).reshape(mesh).mean(axis=(0, 1))

            solution_phi_z = solution_phi_z.get()
            solvation_potential_z = solvation_potential_z.get()
            vac_coulomb_potential_z = vac_coulomb_potential_z.get()
            vcorr_z = vcorr_z.get()
            rho_z = rho_z.get()
            pseudo_nucdensity_z = pseudo_nucdensity_z.get()
            vion_z = vion_z.get()
            vdiel_z = vdiel_z.get()
            vcav_z = vcav_z.get()
            rhoion_z = rhoion_z.get()
            rhodiel_z = rhodiel_z.get()
            S_z = S_z.get()
            Sprime_z = Sprime_z.get()
            Sphi_z = Sphi_z.get()

            if self.chkfile is not None:
                np.savez(self.chkfile,
                    z=z,
                    solution_phi_z=solution_phi_z,
                    solvation_potential_z=solvation_potential_z,
                    vac_coulomb_potential_z=vac_coulomb_potential_z,
                    vcorr_z=vcorr_z.real,
                    vion_z=vion_z.real,
                    vdiel_z=vdiel_z,
                    vcav_z=vcav_z,
                    rhoion_z=rhoion_z,
                    rhodiel_z=rhodiel_z,
                    S_z=S_z,
                    Sprime_z=Sprime_z,
                    rho_z=rho_z,
                    sphi_z=Sphi_z,
                    pseudo_nucdensity_z=pseudo_nucdensity_z,
                )

            if self.plot_results:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(z, solvation_potential_z, label='Difference due to solvation')
                ax.plot(z, solution_phi_z, label='Coulomb potential in solution')
                ax.plot(z, vac_coulomb_potential_z, label='Vacuum Coulomb potential')
                ax.plot(z, vcorr_z, label='VCorr', color='red')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('Hartree')
                ax.legend()
                fig.savefig(f"{self.plot_filestem}-pot-{self.ncalls}.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(z, vcorr_z, label='VCorr', color='red')
                ax.plot(z, vion_z, label='Vion', color='green')
                ax.plot(z, vdiel_z, label='Vdiel', color='orange')
                ax.plot(z, vcav_z, label='Vcav', color='blue')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('Hartree')
                ax.legend()
                fig.savefig(f"{self.plot_filestem}-vcorr-{self.ncalls}.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(z, rhoion_z, label='Ion density')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('density (e/Bohr^3)')
                ax.legend()
                ax.set_title('xy-averaged densities along z')
                fig.savefig(f"{self.plot_filestem}-rhoion-{self.ncalls}.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(z, Sphi_z, label='S * phi')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('Hartree')
                ax.legend()
                ax.set_title('xy-averaged densities along z')
                fig.savefig(f"{self.plot_filestem}-sphi-{self.ncalls}.png")
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(z, S_z, label='Cavity function', color='black')
                ax2 = ax.twinx()
                ax2.plot(z, Sprime_z, label='Cavity derivative', color='red')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('Cavity function')
                ax2.set_ylabel('Cavity derivative')
                ax.legend()
                ax2.legend(loc='upper right')
                fig.savefig(f"{self.plot_filestem}-cav-{self.ncalls}.png", dpi=600)
                plt.close()

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(z, rho_z, label='Charge density')
                ax.plot(z, pseudo_nucdensity_z, label='Pseudo nuclear charge density', color='red')
                ax.set_xlabel('Z (Angstrom)')
                ax.set_ylabel('density (e/Bohr^3)')
                ax.legend()
                ax.set_title('xy-averaged densities along z')
                fig.savefig(f"{self.plot_filestem}-chgdens-{self.ncalls}.png", dpi=600)
                plt.close()

        t3 = log.timer("LPBE workup", *t2)
        return results

    def kernel(self, dm_kpts, tol=None):
        results = self.kernel_detail(dm_kpts, tol=tol)
        Ecorr = results['E_coul_corr'] + results['Ecav']
        logger.info(self, "LPBE solvation free energy: %f", Ecorr)
        self.ncalls += 1
        if self.ncalls < self.nskip:
            return 0.0, 0.0
        else:
            return Ecorr, results['vcorr_mat'][0]
