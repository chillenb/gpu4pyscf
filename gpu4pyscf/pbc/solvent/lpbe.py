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


def gradient_recip(F, Gv):
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
    grad_F = cp.empty((3,) + F.shape, dtype=np.complex128)
    for i in range(3):
        grad_F[i, ...] = 1j * Gv[..., i] * F
    return grad_F

def divergence_recip(Fv, Gv):
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
    div_F = cp.zeros(Fv.shape[1:], dtype=np.complex128)
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

        self.is_built = False
        self.tol = 5e-5
        self.frozen = False
        self.debug_checks = kwargs.get('debug_checks', False)

        self.eps        = kwargs.get('eps', 1e-10)
        self.cav_smear  = kwargs.get('cav_smear', 0.6)
        self.cav_dens_cutoff = kwargs.get('cav_dens_cutoff', vasp_dens_to_pyscf_dens(0.0025))
        self.cav_tension = kwargs.get('cav_tension', vasp_tau_to_pyscf_tau(5.25e-4))
        self.rel_permittivity = kwargs.get('rel_permittivity', 78.4)

        default_debye_length = debye_length_au(molar_to_au(0.1), 298.15)

        self.debye_length = kwargs.get('debye_length', default_debye_length)

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
            logger.info(self, f"Nelec by integration: {nelec_by_integration}")

            nuc_charge_by_integration = cp.sum(pseudo_nucdensityR) * vol / ngrids
            logger.info(self, f"Nuclear charge by integration of pseudo_nucdensityR: {nuc_charge_by_integration}")


        S, Sprime = shape_function(rhoR + pseudo_nucdensityR, self.cav_smear, self.cav_dens_cutoff)

        eps_r_field = 1. + (self.rel_permittivity - 1.) * S
        kappa2 = 1.0 / (self.debye_length ** 2)

        if self.debug_checks:
            Svol = cp.sum(S) * vol / ngrids
            Svol_ang = Svol * (nist.BOHR ** 3)
            cell_vol_ang = cell.vol * (nist.BOHR ** 3)
            logger.info(self, f"Svol: {Svol_ang} Ang^3")
            logger.info(self, f"Cell vol: {cell_vol_ang} Ang^3")
            logger.info(self, f"kappa2: {kappa2:.3e} 1/Bohr^2, debye length: {1/np.sqrt(kappa2):.3f} Bohr")




        # solve the equation
        # Div( eps_r * Grad(phi) ) - S phi / (debye_length^2) = -4*pi*solute_chargeR.
        # by preconditioned conjugate gradient.
        # Preconditioner = poisson.
        def Aop(phiG):

            grad_phiG = gradient_recip(phiG, self.Gv).reshape(3, *mesh)
            grad_phiR = pbc_tools.ifft(grad_phiG.reshape(3, -1), mesh).reshape(3, *mesh) / ngrids
            eps_grad_phiR = eps_r_field * grad_phiR
            eps_grad_phiG = pbc_tools.fft(eps_grad_phiR.reshape(3, -1), mesh)
            div_eps_grad_phiG = divergence_recip(eps_grad_phiG.reshape(3, -1), self.Gv)

            phi_R = pbc_tools.ifft(phiG, mesh).reshape(*mesh) / ngrids

            debye_term_real = S * phi_R.reshape(*mesh) * kappa2
            debye_term_G = pbc_tools.fft(debye_term_real.reshape(-1), mesh)

            return -(div_eps_grad_phiG - debye_term_G).reshape(-1)

        if kappa2 == 0:
            yukawa_kernel = self.coul_kernel / (4.0*np.pi) # laplacian operator
        else:
            yukawa_kernel = 1.0 / (self.Gabs2 + kappa2/(4*np.pi))

        def Mprecond(phiG):
            precond_phiG = yukawa_kernel * phiG
            return precond_phiG.reshape(-1)
        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        A = LinearOperator((ngrids, ngrids), matvec=Aop)
        M = LinearOperator((ngrids, ngrids), matvec=Mprecond)
        rhs = -4*np.pi*solute_chargeR.reshape(-1)
        rhs = pbc_tools.fft(rhs.reshape(-1), mesh).reshape(-1)

        def makecallback():
            iter = 0
            def callback(xk):
                nonlocal iter
                iter += 1
                res = Aop(xk) - rhs
                res_norm = cp.linalg.norm(res)
                logger.info(self, f"CG iteration {iter}, residual norm {res_norm:.3e}")
            return callback

        solution_phi_G, info = cg(A, rhs, M=M, tol=1e-8, maxiter=200, callback=makecallback())
        if info != 0:
            logger.warn(self, f"Conjugate gradient did not converge: info={info}")

        solution_phi_R = pbc_tools.ifft(solution_phi_G.reshape(-1), mesh).reshape(*mesh) / ngrids

        rho_ion_R = solution_phi_R * S * kappa2 / (4*np.pi)


        # compute solvation potential.
        solute_chargeG = pbc_tools.fft(solute_chargeR.reshape(-1), mesh).reshape(-1)
        vac_coulomb_potentialG = -1.0 * self.coul_kernel * solute_chargeG
        vac_coulomb_potentialR = pbc_tools.ifft(vac_coulomb_potentialG.reshape(-1), mesh).reshape(*mesh)
        solvation_potentialR = solution_phi_R - vac_coulomb_potentialR
    
        solvation_potentialG = pbc_tools.fft(solvation_potentialR.reshape(-1), mesh).reshape(-1)

        grad_solution_phiG = gradient_recip(solution_phi_G, self.Gv)
        grad_solution_phiR = pbc_tools.ifft(grad_solution_phiG, mesh).real / ngrids
    
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


        return results

    def kernel(self, dm_kpts, tol=None):
        results = self.kernel_detail(dm_kpts, tol=tol)
        return 0.0, results['vcorr_mat'][0]