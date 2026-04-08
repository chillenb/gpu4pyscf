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

        self.debye_length = debye_length_au(molar_to_au(kwargs.get('ionic_strength', 1.0)), 298.15)

        self.vpplocG = None


    def dump_flags(self, verbose=None):
        logger.info(self, "******** Periodic LPBE flags ********")
        logger.info(self, f"  kpts: {self.kpts}")
        logger.info(self, f"  tol: {self.tol}")
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
        vpplocG = multigrid_v1.eval_vpplocG(self.cell, self.mesh)
        return vpplocG

    def get_vpplocR(self):
        vpplocG = self.get_vpplocG()
        mesh = self.mesh
        ngrids = np.prod(mesh)
        weight = self.cell.vol / ngrids
        vpplocR = ifft_in_place(vpplocG.reshape(-1, *self.mesh))
        vpplocR /= weight
        return vpplocR

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

    def get_pseudo_nucdensityR(self):
        pseudo_nucdensityG = self.get_pseudo_nucdensity()
        mesh = self.mesh
        ngrids = np.prod(mesh)
        weight = self.cell.vol / ngrids
        pseudo_nucdensityR = ifft_in_place(pseudo_nucdensityG.reshape(-1, *self.mesh))
        pseudo_nucdensityR /= weight
        return pseudo_nucdensityR

    def get_rhoG(self, dm_kpts):
        dms = _format_dms(dm_kpts, self.kpts)
        return multigrid_v2.evaluate_density_on_g_mesh(self.ni, dms, self.kpts)

    def get_rhoR(self, dm_kpts):
        dms = _format_dms(dm_kpts, self.kpts)
        nset = dms.shape[0]
        density = self.get_rhoG(dm_kpts)
        mesh = self.mesh
        density = density.reshape(-1, *mesh)
        ngrids = np.prod(mesh)
        weight = self.cell.vol / ngrids
        density = ifft_in_place(density).real.reshape(-1, *self.mesh)
        density /= weight
        return density

    def potg_to_potmat(self, potG):
        weight = self.cell.vol / np.prod(self.mesh)
        return multigrid_v2.convert_xc_on_g_mesh_to_fock(self.ni, potG*weight, hermi=1, kpts=self.kpts)


    def kernel_detail(self, dm_kpts, tol=None):
        self.build()
        log = logger.new_logger(self, verbose=self.verbose)
        if tol is None:
            tol = self.tol
        tol = max(tol, self.tol)

        if self.vpplocG is None:
            self.vpplocG = self.get_vpplocG()
        mesh = self.ni.mesh
        ngrids = np.prod(mesh)
        cell = self.cell
        vol = cell.vol
        kpts = self.kpts
        dms = _format_dms(dm_kpts, kpts)

        rhoR = self.get_rhoR(dms)
        pseudo_nucdensityR = self.get_pseudo_nucdensityR().real

        rhoR = rhoR.reshape(*mesh)
        pseudo_nucdensityR = pseudo_nucdensityR.reshape(*mesh)

        solute_chargeR = -rhoR + pseudo_nucdensityR

        if self.debug_checks:
            nelec_by_integration = cp.sum(rhoR) * vol / ngrids
            log.info(f"Nelec by integration: {nelec_by_integration}")

            nuc_charge_by_integration = cp.sum(pseudo_nucdensityR) * vol / ngrids
            log.info(f"Nuclear charge by integration of pseudo_nucdensityR: {nuc_charge_by_integration}")


        S, Sprime = shape_function(rhoR + pseudo_nucdensityR, self.cav_smear, self.cav_dens_cutoff)

        eps_r_field = 1. + (self.rel_permittivity - 1.) * S
        kappa2 = 1.0 / (self.debye_length ** 2)

        if self.debug_checks:
            Svol = cp.sum(S) * vol / ngrids
            Svol_ang = Svol * (nist.BOHR ** 3)
            cell_vol_ang = cell.vol * (nist.BOHR ** 3)
            log.info(f"Svol: {Svol_ang} Ang^3")
            log.info(f"Cell vol: {cell_vol_ang} Ang^3")
            log.info(f"kappa2: {kappa2:.3e} 1/Bohr^2, debye length: {1/np.sqrt(kappa2):.3f} Bohr")


        if self.extra_screening_list is not None and self.ncalls < len(self.extra_screening_list):
            extrakappa2 = self.extra_screening_list[self.ncalls]
        else:
            extrakappa2 = 0.0


        # solve the equation
        # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
        # by preconditioned conjugate gradient.
        # Preconditioner = poisson.



        def make_aop(Skappa2):
            def Aop(phiG):
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

        if kappa2 == 0:
            yukawa_kernel = self.coul_kernel # laplacian operator
        else:
            yukawa_kernel = 4.0 * np.pi / (self.Gabs2 + kappa2/(self.rel_permittivity))

        def Mprecond(phiG):
            precond_phiG = yukawa_kernel * phiG
            return precond_phiG.reshape(-1)

        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        t0 = log.init_timer()

        A = LinearOperator((ngrids, ngrids), matvec=make_aop(extrakappa2 + S*kappa2))
        M = LinearOperator((ngrids, ngrids), matvec=Mprecond)
        rhs = -4*np.pi*solute_chargeR.reshape(-1)
        rhs = pbc_tools.fft(rhs.reshape(-1), mesh).reshape(-1)
        solution_phi_G, info = cg(A, rhs, M=M, x0=self.pot_guess, tol=1e-8, maxiter=200)
        if info != 0:
            logger.warn(self, f"Conjugate gradient did not converge: info={info}")

        t1 = log.timer("LPBE CG solve", *t0)
        # def makecallback():
        #     iter = 0
        #     def callback(xk):
        #         nonlocal iter
        #         iter += 1
        #         res = Aop(xk) - rhs
        #         res_norm = cp.linalg.norm(res)
        #         logger.info(self, f"CG iteration {iter}, residual norm {res_norm:.3e}")
        #     return callback


        solution_phi_R = pbc_tools.ifft(solution_phi_G.reshape(-1), mesh).reshape(*mesh)

        self.pot_guess = solution_phi_G

        rho_ion_R = solution_phi_R * S * kappa2 / (4*np.pi)


        # compute solvation potential.
        solute_chargeG = pbc_tools.fft(solute_chargeR.reshape(-1), mesh).reshape(-1)
        vac_coulomb_potentialG = -1.0 * self.coul_kernel * solute_chargeG



        vac_coulomb_potentialR = pbc_tools.ifft(vac_coulomb_potentialG.reshape(-1), mesh).reshape(*mesh)


        solvation_potentialR = solution_phi_R - vac_coulomb_potentialR

        solvation_potentialG = pbc_tools.fft(solvation_potentialR.reshape(-1), mesh).reshape(-1)

        grad_solution_phiG = gradient_recip(solution_phi_G, self.Gv)
        grad_solution_phiR = pbc_tools.ifft(grad_solution_phiG, mesh).real
    
        vcorr_r = -1.0/(8*np.pi) * Sprime.reshape(-1) * cp.einsum('ng, ng ->g', grad_solution_phiR, grad_solution_phiR) \
             - 1.0/(8*np.pi) * kappa2 * Sprime.reshape(-1) * solution_phi_R.reshape(-1)**2
        vcorr_g = solvation_potentialG + pbc_tools.fft(vcorr_r.reshape(-1), mesh).reshape(-1)
    
        vcorr_mat = self.potg_to_potmat(vcorr_g)


        results = {}
        results['S'] = S
        results['solution_phi_G'] = solution_phi_G.reshape(*mesh)
        results['solution_phi_R'] = solution_phi_R
        results['rho_ion_R'] = rho_ion_R

        rhog = pbc_tools.fft(rhoR.reshape(-1), mesh).reshape(-1)
        vhar = self.coul_kernel * rhog
        vj = self.potg_to_potmat(vhar)
        results['vj'] = vj
        results['solvation_potentialR'] = solvation_potentialR
        results['solvation_potentialG'] = solvation_potentialG
        results['vac_coulomb_potentialR'] = vac_coulomb_potentialR
        results['vcorr_r'] = vcorr_r
        results['vcorr_mat'] = vcorr_mat



        if self.plot_results:
            import matplotlib.pyplot as plt
            mesh = self.mesh
            fig, ax = plt.subplots(figsize=(12, 8))
            # plot XY-averaged potentials along Z.
            z = np.arange(mesh[2]) * self.cell.lattice_vectors(unit='A')[2, 2] / mesh[2]
            solution_phi_z = solution_phi_R.reshape(mesh).mean(axis=(0, 1))
            solvation_potential_z = solvation_potentialR.reshape(mesh).mean(axis=(0, 1))
            vac_coulomb_potential_z = vac_coulomb_potentialR.reshape(mesh).mean(axis=(0, 1))
            vcorr_z = vcorr_r.reshape(mesh).mean(axis=(0, 1))
            ax2 = ax.twinx()
            ax.plot(z, solvation_potential_z.real.get(), label='Difference due to solvation')
            ax.plot(z, solution_phi_z.real.get(), label='Coulomb potential in solution')
            ax.plot(z, vac_coulomb_potential_z.real.get(), label='Vacuum Coulomb potential')
            ax2.plot(z, vcorr_z.real.get(), label='VCorr', color='red')
            ax.set_xlabel('Z (Angstrom)')
            ax.set_ylabel('Hartree')
            ax2.set_ylabel('VCorr')
            ax.legend()
            ax2.legend(loc='upper right')
            fig.savefig(f"{self.plot_filestem}-pot-{self.ncalls}.png")
            plt.close()


            fig, ax = plt.subplots(figsize=(12, 8))
            # plot XY-averaged densities along Z.
            rhoion_z = rho_ion_R.reshape(mesh).mean(axis=(0, 1))
            ax.plot(z, rhoion_z.real.get(), label='Ion density')
            ax.set_xlabel('Z (Angstrom)')
            ax.set_ylabel('density (e/Bohr^3)')
            ax.legend()
            ax.set_title('xy-averaged densities along z')
            fig.savefig(f"{self.plot_filestem}-rhoion-{self.ncalls}.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 4))
            # plot XY-averaged densities along Z.
            S_z = S.reshape(mesh).mean(axis=(0, 1))
            Sprime_z = Sprime.reshape(mesh).mean(axis=(0, 1))
            ax.plot(z, S_z.real.get(), label='Cavity function', color='black')
            ax2 = ax.twinx()
            ax2.plot(z, Sprime_z.real.get(), label='Cavity derivative', color='red')
            ax.set_xlabel('Z (Angstrom)')
            ax.set_ylabel('Cavity function')
            ax2.set_ylabel('Cavity derivative')
            ax.legend()
            ax2.legend(loc='upper right')
            fig.savefig(f"{self.plot_filestem}-cav-{self.ncalls}.png", dpi=600)
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 8))
            #ax2 = ax.twinx()
            # plot XY-averaged densities along Z.
            rho_z = rhoR.reshape(mesh).mean(axis=(0, 1))
            pseudo_nucdensity_z = pseudo_nucdensityR.reshape(mesh).mean(axis=(0, 1))
            ax.plot(z, rho_z.real.get(), label='Charge density')
            ax.plot(z, pseudo_nucdensity_z.real.get(), label='Pseudo nuclear charge density', color='red')
            ax.set_xlabel('Z (Angstrom)')
            ax.set_ylabel('density (e/Bohr^3)')
            ax.legend()
            ax.set_title('xy-averaged densities along z')
            fig.savefig(f"{self.plot_filestem}-chgdens-{self.ncalls}.png", dpi=600)
            plt.close()

        return results

    def kernel(self, dm_kpts, tol=None):
        results = self.kernel_detail(dm_kpts, tol=tol)
        self.ncalls += 1
        if self.ncalls < self.nskip:
            return 0.0, 0.0
        else:
            return 0.0, results['vcorr_mat'][0]