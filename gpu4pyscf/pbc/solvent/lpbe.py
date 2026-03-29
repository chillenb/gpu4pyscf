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

        self.is_built = False
        self.tol = 5e-5
        self.frozen = False
        self.debug_checks = kwargs.get('debug_checks', False)

        self.eps        = kwargs.get('eps', 1e-10)
        self.cav_smear  = kwargs.get('cav_smear', 0.6)
        self.cav_dens_cutoff = kwargs.get('cav_dens_cutoff', vasp_dens_to_pyscf_dens(0.0025))
        self.cav_tension = kwargs.get('cav_tension', vasp_tau_to_pyscf_tau(5.25e-4))
        self.rel_permittivity = kwargs.get('rel_permittivity', 78.4)
        self.debye_length = kwargs.get('debye_length', 10.0)

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


    def get_vpplocG(self):
        vpplocG = multigrid_v1.eval_vpplocG(self.cell, self.mesh)
        return vpplocG

    def get_rhoG(self, dm_kpts):
        return multigrid_v2.evaluate_density_on_g_mesh(self.ni, dm_kpts, self.kpts)
    
    def get_rhoR(self, dm_kpts):
        nset = dm_kpts.shape[0]
        density = self.get_rhoG(dm_kpts)
        mesh = self.mesh
        density = density.reshape(-1, *mesh)
        ngrids = np.prod(mesh)
        weight = self.cell.vol / ngrids
        density = ifft_in_place(density).real.reshape(nset, -1, ngrids)
        density /= weight
        return density

    def kernel(self, dm_kpts, tol=None):
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

        if self.debug_checks:
            nelec_by_integration = cp.sum(rhoR) * vol / ngrids
            logger.info(self, f"Nelec by integration: {nelec_by_integration}")

        S, Sprime = shape_function(rhoR, self.cav_smear, self.cav_dens_cutoff)

        if self.debug_checks:
            Svol = cp.sum(S) * vol / ngrids
            Svol_ang = Svol * (nist.BOHR ** 3)
            cell_vol_ang = cell.vol * (nist.BOHR ** 3)
            logger.info(self, f"Svol: {Svol_ang} Ang^3")
            logger.info(self, f"Cell vol: {cell_vol_ang} Ang^3")