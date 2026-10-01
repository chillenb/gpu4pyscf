#!/usr/bin/env bash
export CUDACXX=nvcc
export CUDAARCHS="90a;100a"
export CUDAHOSTCXX=g++
export CXX=g++
export CC=gcc

set -e

cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf \
	-DCUDA_ARCHITECTURES=$CUDAARCHS \
	-DCMAKE_CUDA_FLAGS="--fmad=true" \
	-DBUILD_LIBXC=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DFORCE_INTEL_OPENMP=ON 
cmake --build build/temp.gpu4pyscf -j $(nproc) --verbose


