import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.fft as fft
from cupyx.scipy.sparse.linalg import LinearOperator, cg

from cupy.cuda.nvtx import RangePush, RangePop

from pyscf.data import nist

from gpu4pyscf.pbc.gto.cell import get_Gv_weights
from gpu4pyscf.pbc.df.fft_jk import _format_jks
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.tools import pbc as pbc_tools
from gpu4pyscf.pbc.tools import k2gamma
from gpu4pyscf.lib.cupy_helper import batched_vec_norm2, tag_array
from gpu4pyscf.lib.cupy_helper import ndarray, tag_array, get_avail_mem, vec_dot

from gpu4pyscf.pbc.dft.multigrid_v3 import MultiGridNumInt
from gpu4pyscf.pbc.dft.multigrid_v3 import fft_in_place, ifft_in_place, _apply_Gv_1j, _xc_var_length, _get_coulomb_in_place
from gpu4pyscf.pbc.dft.multigrid_v3 import _wannier_transform_dm, _get_Gv_bases, _eval_density, _density_to_real_space
from gpu4pyscf.pbc.dft.multigrid_v3 import _inverse_wannier_transform_fock, _vxc_to_reciprocal_space, _eval_xc_mat, _contract_Gv_1j

from gpu4pyscf.__config__ import props as gpu_specs

import gpu4pyscf.pbc.dft.multigrid as multigrid_v1


_kernel_registry = {}

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


def gradient_recip(F, Gx, Gy, Gz, out=None):
    """Compute the gradient of a function in reciprocal space.

    Parameters
    ----------
    F : ndarray
        The function values in reciprocal space.
    Gx, Gy, Gz : ndarray
        The reciprocal lattice vectors in each direction.

    Returns
    -------
    ndarray
        The gradient of the function in reciprocal space.
    """
    assert F.ndim == 3, "F must be a 3D array"
    assert F.flags.c_contiguous, "F must be C-contiguous"
    nx, ny, nz = Gx.shape[1], Gy.shape[1], Gz.shape[1]
    assert F.shape == (nx, ny, nz), f"F.shape = {F.shape}, expected {(nx, ny, nz)}"
    assert F.dtype == cp.complex128
    if out is None:
        grad_F = cp.empty((3,) + F.shape, dtype=np.complex128)
    else:
        assert out.shape == (3,) + F.shape
        grad_F = out
    for n in range(3):
        _apply_Gv_1j(F, Gx[n], Gy[n], Gz[n], out=grad_F[n])
    return grad_F

def divergence_recip(Fv, Gx, Gy, Gz, out=None):
    """Compute the divergence of a vector function in reciprocal space.

    Parameters
    ----------
    Fv : ndarray
        The vector function values in reciprocal space.
    Gx, Gy, Gz : ndarray
        The reciprocal lattice vectors in each direction.

    Returns
    -------
    ndarray
        The divergence of the vector function in reciprocal space.
    """
    assert Fv.ndim == 4, "Fv must be a 4D array"
    assert Fv.flags.c_contiguous, "Fv must be C-contiguous"
    nx, ny, nz = Gx.shape[1], Gy.shape[1], Gz.shape[1]
    assert Fv.shape == (3, nx, ny, nz), f"Fv.shape = {Fv.shape}, expected {(3, nx, ny, nz)}"
    assert Fv.dtype == cp.complex128
    if out is None:
        div_F = cp.zeros(Fv.shape[1:], dtype=np.complex128)
    else:
        div_F = out
        div_F.fill(0.0)
    for i in range(3):
        _contract_Gv_1j(div_F, Fv[i], Gx[i], Gy[i], Gz[i])
    # Above function does div_F += -1j * Gv[..., i] * Fv[i, ...]
    # we want +1j.
    div_F *= -1.0
    return div_F


def fft_3d(x):
    return fft.fftn(x.astype(cp.complex128), axes=(-3, -2, -1))
def ifft_3d(x):
    return fft.ifftn(x.astype(cp.complex128), axes=(-3, -2, -1))

def _precond_yukawa_or_coul(rhoG, Gv_bases, eps_r=1.0, ebkappa2=0.0, out=None):
    '''
    Computes
    out = 4*pi*rhoG / (eps_r * |G|^2 + ebkappa2) if ebkappa2 != 0, else
    out = 4*pi*rhoG / (eps_r * |G|^2)
    '''
    fn_name = 'precond_yukawa_or_coul'
    if fn_name not in _kernel_registry:
        kernel_code = ('''\
extern "C" __global__
void ''' + fn_name + r'''(double2* __restrict__ out, double2* __restrict__ rhoG,
    double *Gx, double *Gy, double *Gz, long long nx, long long ny, long long nz,
    double eps_r, double ebkappa2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    size_t nyz = ny * nz;
    size_t ng = nx * nyz;
    for (size_t g = idx; g < ng; g += stride) {
        int ix = g / nyz;
        int iyz = g - nyz * ix;
        int iy = iyz / nz;
        int iz = iyz - nz * iy;
        double GG = 0;
        for (int n = 0; n < 3; ++n) {
            double Gv = Gx[n*nx+ix] + Gy[n*ny+iy] + Gz[n*nz+iz];
            GG += Gv * Gv;
        }
        double2 coul = {0.0, 0.0};
        double2 rho = rhoG[g];


        if (ebkappa2 != 0.0) {
            double fac = 12.566370614359172 / (eps_r * GG + ebkappa2);
            coul = {fac * rho.x, fac * rho.y};
        } else if (GG != 0) {
            double fac = 12.566370614359172 / (eps_r * GG);
            coul = {fac * rho.x, fac * rho.y};
        }

        out[g] = coul;
    }
}''')
        _kernel_registry[fn_name] = cp.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registry[fn_name]
    nx, ny, nz = [x.shape[1] for x in Gv_bases]
    ng = nx * ny * nz
    assert rhoG.size == ng
    out = ndarray(rhoG.shape, buffer=out, dtype=np.complex128)
    workers = gpu_specs['multiProcessorCount']
    kernel((workers*2,), (1024,), (out, rhoG, Gv_bases[0], Gv_bases[1], Gv_bases[2], nx, ny, nz, float(eps_r), float(ebkappa2)))
    return out

def _G2_scale(phiG, Gv_bases, alpha=1.0, out=None):
    '''
    Computes
    out = alpha * |G|^2 * phiG
    '''
    fn_name = 'G2_scale'
    if fn_name not in _kernel_registry:
        kernel_code = ('''\
extern "C" __global__
void ''' + fn_name + r'''(double2* __restrict__ out, double2* __restrict__ phiG,
    double *Gx, double *Gy, double *Gz, long long nx, long long ny, long long nz,
    double alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    size_t nyz = ny * nz;
    size_t ng = nx * nyz;
    for (size_t g = idx; g < ng; g += stride) {
        int ix = g / nyz;
        int iyz = g - nyz * ix;
        int iy = iyz / nz;
        int iz = iyz - nz * iy;
        double GG = 0;
        for (int n = 0; n < 3; ++n) {
            double Gv = Gx[n*nx+ix] + Gy[n*ny+iy] + Gz[n*nz+iz];
            GG += Gv * Gv;
        }
        double2 phi = phiG[g];
        double fac = GG * alpha;
        out[g] = {phi.x * fac, phi.y * fac};
    }
}''')
        _kernel_registry[fn_name] = cp.RawKernel(kernel_code, fn_name)

    kernel = _kernel_registry[fn_name]
    nx, ny, nz = [x.shape[1] for x in Gv_bases]
    ng = nx * ny * nz
    assert phiG.size == ng
    out = ndarray(phiG.shape, buffer=out, dtype=np.complex128)
    workers = gpu_specs['multiProcessorCount']
    kernel((workers*2,), (1024,), (out, phiG, Gv_bases[0], Gv_bases[1], Gv_bases[2], nx, ny, nz, float(alpha)))
    return out


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



def lpbe_inner(ni, rhoG, Gv_bases, options=None, pot_guess=None):

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

    Gv = pbc_tools.get_Gv(ni.cell, ni.mesh)

    Gx, Gy, Gz = Gv_bases

    Gabs2 = cp.einsum('gi,gi->g', Gv, Gv)
    del Gv


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


    vpplocG = ni.vpplocG
    # pseudo_nucdensityG = Gabs2 * vpplocG * (-1.0 / (4*np.pi))
    pseudo_nucdensityG = _G2_scale(vpplocG, Gv_bases, alpha=-1.0/(4*np.pi))

    charges = cell.atom_charges()
    tot_nuc_charge = np.sum(charges)
    pseudo_nucdensityG[0] = tot_nuc_charge
    pseudo_nucdensityR = ifft_3d(pseudo_nucdensityG.reshape(*mesh)) / weight

    rhoR = ifft_3d(rhoG.reshape(*mesh)) / weight
    # Charge sign convention is that electrons are positive.
    solute_chargeR = rhoR - pseudo_nucdensityR

    nelec_by_integration = cp.sum(rhoR) * vol / ngrids
    nuc_charge_by_integration = cp.sum(pseudo_nucdensityR) * vol / ngrids
    qsol = nelec_by_integration - nuc_charge_by_integration

    pseudocore_densityG = ni.pseudocore_densityG
    pseudocore_densityR = ifft_3d(pseudocore_densityG.reshape(*mesh)) / weight

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
            # phiG_3d = phiG.reshape(*mesh)
            # grad_phiG = gradient_recip(phiG_3d, Gx, Gy, Gz).reshape(3, *mesh)
            # grad_phiR = ifft_in_place(grad_phiG)
            # eps_grad_phiR = eps_r_field * grad_phiR
            # eps_grad_phiG = fft_in_place(eps_grad_phiR)
            # div_eps_grad_phiG = divergence_recip(eps_grad_phiG, Gx, Gy, Gz)
            # phi_R = pbc_tools.ifft(phiG, mesh).reshape(*mesh)
            # debye_term_real = Skappa2 * phi_R.reshape(*mesh)
            # debye_term_G = fft_in_place(debye_term_real)
            # return -(div_eps_grad_phiG - debye_term_G).reshape(-1)

            phiG_3d = phiG.reshape(*mesh)
            minus_div_eps_grad_phiG = cp.zeros(mesh, dtype=cp.complex128)
            buf = cp.zeros(mesh, dtype=cp.complex128)
            for i in range(3):
                _apply_Gv_1j(phiG_3d, Gx[i], Gy[i], Gz[i], buf)
                ifft_in_place(buf)
                buf *= eps_r_field
                fft_in_place(buf)
                _contract_Gv_1j(minus_div_eps_grad_phiG, buf, Gx[i], Gy[i], Gz[i])
            buf[:, :, :] = phiG_3d
            ifft_in_place(buf)
            buf *= Skappa2
            fft_in_place(buf)
            buf += minus_div_eps_grad_phiG
            return buf.reshape(-1)
        return Aop

    mean_S = cp.mean(S.reshape(-1))
    mean_eps_r = float(1. + (rel_permittivity - 1.) * mean_S)
    mean_ebkappa2 = float(mean_S * ebkappa2)

    def Mprecond(phiG):
        precond_phiG = _precond_yukawa_or_coul(phiG, Gv_bases, eps_r=mean_eps_r, ebkappa2=mean_ebkappa2)
        return precond_phiG.reshape(-1)


    t0 = log.init_timer()

    A = LinearOperator((ngrids, ngrids), matvec=make_aop(S*ebkappa2), dtype=cp.complex128)
    M = LinearOperator((ngrids, ngrids), matvec=Mprecond, dtype=cp.complex128)
    rhs = fft_3d(4*np.pi*solute_chargeR.reshape(*mesh)).reshape(-1) * weight

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


    RangePush("lpbe_postprocess")
    solution_phi_R = ifft_3d(solution_phi_G.reshape(*mesh)) / weight

    # rho_ion_R = solution_phi_R * S * (ebkappa2 / (4*np.pi))


    # compute solvation potential.
    solute_chargeG = fft_3d(solute_chargeR.reshape(*mesh)).reshape(-1) * weight
    #vac_coulomb_potentialG = coul_kernelG * solute_chargeG
    vac_coulomb_potentialG = _precond_yukawa_or_coul(solute_chargeG, Gv_bases, eps_r=1.0, ebkappa2=0.0)

    vac_coulomb_potentialG = vac_coulomb_potentialG.reshape(-1)
    
    # The next line ensures that the G=0 component of the vacuum Coulomb potential
    # is consistent with vpplocG.
    # vpplocG has a non-zero G=0 component.

    vac_coulomb_potentialG[0] += vpplocG.reshape(-1)[0]

    vac_coulomb_potentialR = ifft_3d(vac_coulomb_potentialG.reshape(*mesh)) / weight


    solvation_potentialR = solution_phi_R - vac_coulomb_potentialR

    solvation_potentialG = fft_3d(solvation_potentialR.reshape(*mesh)).reshape(-1) * weight

    grad_solution_phiG = gradient_recip(solution_phi_G.reshape(*mesh), Gx, Gy, Gz)
    grad_solution_phiR = ifft_in_place(grad_solution_phiG).reshape(3, -1) / weight

    S_grad_solution_phiR = S * grad_solution_phiR.reshape(3, *mesh)

    S_grad_solution_phiG = fft_in_place(S_grad_solution_phiR)
    div_S_grad_solution_phiG = divergence_recip(S_grad_solution_phiG, Gx, Gy, Gz).reshape(-1)

    div_S_grad_solution_phiR = ifft_3d(div_S_grad_solution_phiG.reshape(*mesh)) / weight
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
    rhoR_smoothed = ifft_in_place(rhoG_smoothed.reshape(*mesh)).real * weight
    rhoR_z = rhoR_smoothed.mean(axis=(0, 1))
    dens_min_idx = cp.argmin(rhoR_z)
    vacpot_at_zmin = vac_coulomb_potentialR.reshape(mesh).real.mean(axis=(0, 1))[dens_min_idx]
    solpot_at_zmin = solution_phi_R.reshape(mesh).real.mean(axis=(0, 1))[dens_min_idx]

    del rhoG_smoothed, rhoR_smoothed, rhoR_z




    # V_\mathrm{cav} = \tau \partial_{\rho} S(r) \left( \frac{\nabla^{2}\rho}{|\nabla\rho|} - 
    #                  \frac{1}{|\nabla\rho|^{3}}(\nabla\rho)^{\mathrm{t}}\mathbf{H}_{\rho}(\nabla\rho) \right)
    # tau is self.cav_tension.

    # Cavitation potential.

    # grad_rho_r = pbc_tools.ifft(gradient_recip(rhoG.reshape(*mesh), Gx, Gy, Gz), mesh).real.reshape(3, *mesh) / weight

    # lap_rho_r = pbc_tools.ifft((-Gabs2 * rhoG).reshape(-1), mesh).real.reshape(*mesh) / weight

    grad_rhoG = gradient_recip(rhoG.reshape(*mesh), Gx, Gy, Gz)
    grad_rho_r = ifft_in_place(grad_rhoG).real.reshape(3, *mesh) / weight
    lap_rhoG = _G2_scale(rhoG.reshape(*mesh), Gv_bases, alpha=-1.0)
    lap_rho_r = ifft_in_place(lap_rhoG).real.reshape(*mesh) / weight

    # grad_hess_grad_r = (nabla rho)^t H(rho) (nabla rho)
    grad_hess_grad_r = cp.zeros(mesh, dtype=cp.float64)

    for i in range(3):
        for j in range(3):
            # hij_g = -(Gv[:, i] * Gv[:, j]) * rhoG
            hij_g = _apply_Gv_1j(rhoG, Gx[i], Gy[i], Gz[i])
            hij_g = _apply_Gv_1j(hij_g, Gx[j], Gy[j], Gz[j], out=hij_g)

            hij_r = ifft_in_place(hij_g.reshape(*mesh)).real / weight
            grad_hess_grad_r += grad_rho_r[i] * hij_r * grad_rho_r[j]

    grad_abs_r = cp.sqrt(cp.einsum('i...,i...->...', grad_rho_r, grad_rho_r))
    grad_abs_safe_r = cp.maximum(grad_abs_r, eps)

    vcav_r = cav_tension * Sprime * (
        lap_rho_r / grad_abs_safe_r
        - grad_hess_grad_r / (grad_abs_safe_r ** 3)
    )
    vcav_r = vcav_r.reshape(-1)
    nvcav = cp.linalg.norm(vcav_r)

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
    log.debug(f"Norm of cavitation potential: {nvcav:.3e}")


    if ni.chkfile is not None:
        z = np.arange(mesh[2]) * cell.lattice_vectors(unit='A')[2, 2] / mesh[2]
        solution_phi_z = solution_phi_R.reshape(mesh).mean(axis=(0, 1))
        solvation_potential_z = solvation_potentialR.reshape(mesh).mean(axis=(0, 1))
        vac_coulomb_potential_z = vac_coulomb_potentialR.reshape(mesh).mean(axis=(0, 1))
        vcorr_z = vcorr_r.reshape(mesh).mean(axis=(0, 1))
        rho_z = rhoR.reshape(mesh).mean(axis=(0, 1))
        pseudo_nucdensity_z = pseudo_nucdensityR.reshape(mesh).mean(axis=(0, 1))
        vion_z = vion_r.reshape(mesh).mean(axis=(0, 1))
        vdiel_z = vdiel_r.reshape(mesh).mean(axis=(0, 1))
        vcav_z = vcav_r.reshape(mesh).mean(axis=(0, 1))
        rho_ion_R = solution_phi_R * S * (ebkappa2 / (4*np.pi))
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

        np.savez(ni.chkfile,
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

    results = {
        'Eion': Eion,
        'Ediel': Ediel,
        'Ecav': Ecav,
        'E_coul_corr': E_coul_corr,
        'pot_guess': solution_phi_G,
        'cavity_r': S,
        'solvation_potentialR': solvation_potentialR,
        'vcav_r': vcav_r,
        'vdiel_r': vdiel_r,
        'vion_r': vion_r,
        'vcorr_g': vcorr_g,
        'pseudocore_densityR': pseudocore_densityR,
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
        dump_vesta_prefix : str or path-like, optional
            Prefix for writing LPBE scalar fields in VESTA ``.pgrid`` format.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
    '''
    RangePush("nr_rks_lpbe")
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()

    xctype = ni._xc_type(xc_code)
    nvar = _xc_var_length(xctype)

    if not with_j:
        raise ValueError("Why are you calling this function if you don't want electrostatics?")

    dm_sc = _wannier_transform_dm(ni, dm_kpts, kpts, hermi, xctype)
    assert len(dm_sc) == 1
    dm_sc = dm_sc[0]

    cell = ni.cell
    mesh = ni.mesh
    ngrids = np.prod(mesh)
    vol = cell.vol
    weight = vol / ngrids
    Gv_bases = _get_Gv_bases(mesh, cell.reciprocal_vectors())

    rhoG, tauG = _eval_density(ni, dm_sc, with_tau=xctype=='MGGA')
    n_electrons = float(rhoG[0,0,0].real.get())

    # dm_sc is represented in primitive bases (by sorted_cell). Its size can be
    # much larger than the input dm_kpts. Release its memory if remaining memory
    # is insufficient.
    if (nvar+4)*ngrids*8 > get_avail_mem():
        dm_sc = None

    density = cp.empty((nvar, ngrids))
    _density_to_real_space(rhoG, tauG, Gv_bases, xctype, out=density)
    # *(1./weight) because rhoR is scaled by weight in _eval_density. If
    # computing rhoR with IFFT, the weight factor is not needed.
    density *= 1/weight

    rho_sf = ndarray(ngrids, dtype=np.float64, buffer=tauG)
    rho_sf[:] = density[0].real
    t0 = log.timer_debug1("density", *t0)

    # eval_xc_eff supports float64 only
    xc_for_energy, xc_for_fock = ni.eval_xc_eff(
        xc_code, density, deriv=1, xctype=xctype, spin=0, inplace=True)[:2]

    xc_for_fock = xc_for_fock.reshape(nvar, *mesh)

    xc_energy_sum = float(vec_dot(rho_sf, xc_for_energy).get()) * weight
    xc_for_energy = density = rho_sf = None
    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)
    t0 = log.timer_debug1("eval_xc_eff", *t0)



    lpbe_res = lpbe_inner(
        ni, rhoG.reshape(-1), Gv_bases,
        options=ni.options, pot_guess=ni.pot_guess)

    ecoul, coulomb_on_g_mesh = _get_coulomb_in_place(rhoG, Gv_bases)
    ecoul = (.5 / vol) * float(ecoul.get())
    log.debug('Multigrid Coulomb energy %s', ecoul)

    ni.pot_guess = lpbe_res['pot_guess']
    vcorr_g = lpbe_res['vcorr_g']
    Ecorr = float(cp.real(
        lpbe_res['E_coul_corr'] + lpbe_res['Ecav']).get())

    ecoul += Ecorr
    coulomb_on_g_mesh += vcorr_g.reshape(*mesh)

    xc_for_fock *= weight
    # Now xc_for_fock represents xc on G space
    xc_for_fock = _vxc_to_reciprocal_space(
        xc_for_fock, coulomb_on_g_mesh, Gv_bases, work=tauG)
    coulomb_on_g_mesh = tauG = None

    if kpts_band is None:
        veff = _eval_xc_mat(ni, xc_for_fock, out=dm_sc)
        veff = _inverse_wannier_transform_fock(ni, veff, kpts)
    else:
        kpts_band = kpts_band.reshape(-1, 3)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts_band)
        ni = ni.copy().reset().build(kmesh=kmesh, xctype=xctype)
        # ni.build may alter the mesh. vxc was created with mesh different to
        # this new mesh.
        ni.mesh = mesh
        veff = _eval_xc_mat(ni, xc_for_fock)
        veff = _inverse_wannier_transform_fock(ni, veff, kpts_band)

    veff = _format_jks(veff, dm_kpts, kpts_band, kpts)
    veff = tag_array(veff, ecoul=ecoul, exc=xc_energy_sum)
    t0 = log.timer_debug1("xc matrix", *t0)
    return n_electrons, xc_energy_sum, veff


class LPBEMultiGridNumInt(MultiGridNumInt):
    def __init__(self, cell, **options):
        super().__init__(cell)
        self.options = options
        self.vpplocG = None
        self.pseudocore_densityG = None
        self.pot_guess = None
        self._lpbe_mesh = None
        self.chkfile = None
        self.dump_vesta_prefix = options.get('dump_vesta_prefix', None)

    def reset(self, cell=None):
        super().reset(cell)
        if cell is not None:
            self.mesh = cell.mesh
        self.vpplocG = None
        self.pseudocore_densityG = None
        self.pot_guess = None
        self._lpbe_mesh = None
        return self

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.vpplocG = multigrid_v1.eval_vpplocG(self.cell, self.mesh)
        self.pseudocore_densityG = pseudocore_density(self.cell, self.mesh)

    nr_rks = nr_rks_lpbe
    nr_uks = NotImplemented


def multigrid_lpbe(mf, mesh=None, **kwargs):
    mf2 = mf.copy()
    mf2._numint = LPBEMultiGridNumInt(mf2.cell, **kwargs)
    if mesh is not None:
        mf2._numint.mesh = mesh
    return mf2
