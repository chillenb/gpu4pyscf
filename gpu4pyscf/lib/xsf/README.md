Vendored from scipy/xsf (https://github.com/scipy/xsf), commit
01edf7745fded21f4b5da1d6496ea6f003dd50a4, BSD-3-Clause (see LICENSE).

- faddeeva.h  <- include/xsf/faddeeva.h  (unmodified)
- config.h    <- include/xsf/config.h    (unmodified)
- gpu4pyscf_compat.h: gpu4pyscf shim (cuda::std::tgamma for CCCL 2.x in CuPy 13);
  include it before faddeeva.h.

Used by gpu4pyscf/pbc/tools/stc.py for the complex erfcx (Faddeeva w)
needed by the smooth truncated Coulomb kernels. Compiled with NVRTC through
cupy; thrust::complex is provided by cupy's complex.cuh. The CuPy development
branch ships the same code as <cupy/xsf/erf.h>; once gpu4pyscf requires such a
CuPy, this copy can be removed.
