# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

import cupy
from pyscf import lib
from pyscf.lib import logger

from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.scf import khf


def _for_scf(mf, solvent_obj, dm=None):
    '''Add LPBE solvent model to periodic k-point SCF methods.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen LPBE potential is added to the results.
    '''
    if not isinstance(mf, khf.KSCF):
        raise TypeError('KSCFWithLPBE only supports k-point PBC SCF objects')

    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = KSCFWithLPBE(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (KSCFWithLPBE, mf.__class__), name)


class _Solvation:
    pass


class KSCFWithLPBE(_Solvation):
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, KSCFWithLPBE, name_mixin))
        del obj.with_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, cell=None):
        if hasattr(self.with_solvent, 'reset'):
            self.with_solvent.reset(cell)
        return super().reset(cell)

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        vhf = super().get_veff(cell, dm_kpts, dm_last, vhf_last,
                               hermi, kpts, kpts_band)
        with_solvent = self.with_solvent
        if (not with_solvent.frozen) or getattr(with_solvent, 'v', None) is None:
            if dm_kpts is None:
                dm_kpts = self.make_rdm1()
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm_kpts)
        vhf = tag_array(vhf, e_solvent=with_solvent.e, v_solvent=with_solvent.v)
        return vhf

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        # Add solvent response before DIIS so the extrapolation sees total Fock.
        if getattr(vhf, 'v_solvent', None) is None:
            vhf = self.get_veff(self.cell, dm)
        return super().get_fock(
            h1e, s1e, vhf + vhf.v_solvent, dm, cycle, diis,
            diis_start_cycle, level_shift_factor, damp_factor, fock_last)

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        if getattr(vhf_kpts, 'e_solvent', None) is None:
            vhf_kpts = self.get_veff(self.cell, dm_kpts)
        e_tot, e2 = super().energy_elec(dm_kpts, h1e_kpts, vhf_kpts)
        e_solvent = vhf_kpts.e_solvent
        if isinstance(e_solvent, cupy.ndarray):
            e_solvent = e_solvent.get()[()]
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = e_solvent
        logger.info(self, 'Solvent Energy = %.15g', e_solvent)
        return e_tot, e2
