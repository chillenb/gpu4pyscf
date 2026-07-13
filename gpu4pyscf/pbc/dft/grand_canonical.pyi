from typing import Any, Callable

from pyscf.lib import StreamObject


class GrandCanonicalKRKS(StreamObject):
    mf: Any
    cell: Any
    mu: float | None
    sigma: float
    beta: float
    nelec: float | None

    max_cycle: int
    max_outer_cycle: int
    conv_tol: float
    conv_tol_coarse: float
    conv_tol_mu: float
    conv_tol_nelec: float
    diis_space: int
    damp: float
    diis_backtrack: float
    diis_max_backtrack: int
    diis_min_reduction: float
    diis_trust_shrink: float
    diis_trust_expand: float
    diis_expansion: float
    diis_expand_reduction: float
    min_damp: float
    initial_nelec_step: float
    max_nelec_step_fraction: float
    root_nelec_tol: float
    verify_residual_tol: float
    verify_density_tol: float
    callback: Callable[[dict[str, Any]], None] | None
    converged: bool
    cycles: int
    outer_cycles: int
    nfev: int
    refinements: int
    verification_attempts: int
    message: str
    e_tot: float | None
    free_energy: float | None
    grand_potential: float | None
    electron_number: float | None
    entropy: float | None
    entropy_energy: float | None
    residual_rms: float | None
    mo_energy: Any
    mo_coeff: Any
    mo_occ: Any
    scf_summary: dict[str, Any]

    def __init__(
        self,
        mf: Any,
        mu: float | None = ...,
        sigma: float | None = ...,
        nelec: float | None = ...,
    ) -> None: ...
    def dump_flags(self, verbose: int | None = ...) -> GrandCanonicalKRKS: ...
    def check_sanity(self) -> GrandCanonicalKRKS: ...
    def build(self) -> GrandCanonicalKRKS: ...
    @staticmethod
    def search_mu_root_bracket(
        nelec: list[float], delta_mu: list[float]
    ) -> tuple[int, int] | None: ...
    def kernel(self, dm0: Any = ...) -> float: ...
    def scf(self, dm0: Any = ...) -> float: ...
