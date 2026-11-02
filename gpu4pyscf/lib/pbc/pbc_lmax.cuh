/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Max angular momentum of orbital basis functions in the PBC integral kernels
// (int3c2e, ft_aopair, contract_ft_ao) and the multigrid v3 kernels. This is
// independent of LMAX in gvhf-rys/vhf.cuh, which is used by the molecular
// integral code. The gout_stride lookup tables built in Python (LMAX in
// pbc/df/int3c2e.py, pbc/df/ft_ao.py, pbc/dft/multigrid_v3.py) must have
// the shape [..., PBC_LMAX+1, PBC_LMAX+1].
#define PBC_LMAX        5
#define PBC_LMAX1       (PBC_LMAX+1)
#define PBC_NCART_MAX   ((PBC_LMAX+1)*(PBC_LMAX+2)/2)
