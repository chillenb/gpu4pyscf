/* Compatibility shim (gpu4pyscf, not part of xsf).
 *
 * xsf/config.h forwards xsf::cxx::tgamma to cuda::std::tgamma, which the
 * CCCL 2.x headers bundled with CuPy 13 do not provide. Include this header
 * before <xsf/faddeeva.h> when compiling with NVRTC through CuPy.
 */
#pragma once

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#include <cuda/std/cmath>
#include <cuda/std/version>
#if defined(CCCL_VERSION) && CCCL_VERSION < 3000000
namespace cuda { namespace std {
__host__ __device__ inline float tgamma(float x) { return ::tgammaf(x); }
__host__ __device__ inline double tgamma(double x) { return ::tgamma(x); }
} }
#endif
#endif
