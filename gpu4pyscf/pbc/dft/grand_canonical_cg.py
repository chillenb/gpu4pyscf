import cupy as cp
from scipy.special import expit


def omega_gradient_wrt_h(h, f, beta, mu):
    # Gamma = beta[(H + H^H)/2 - mu I]
    gamma_matrix = beta * (0.5 * (h + h.conj().T) - mu * cp.eye(h.shape[0]))

    # Gamma = U diag(gamma) U^H
    gamma, u = cp.linalg.eigh(gamma_matrix)

    # Occupations and divided-difference matrix
    rho = expit(-gamma.get())
    rho_mat = cp.diag(rho)

    gamma_diff = gamma[None, :] - gamma[:, None]
    G = cp.zeros_like(gamma_diff)
    mask = ~cp.eye(len(gamma), dtype=bool)
    G[mask] = 1.0 / gamma_diff[mask]

    # F̃ = U^H F U
    f_tilde = u.conj().T @ f @ u

    # Off-diagonal response terms
    A_plus = (G * (f_tilde.T + f_tilde)) @ rho_mat
    A_minus = (G * (f_tilde.T - f_tilde)) @ rho_mat

    # Diagonal occupation-response contribution
    diag_term = cp.diag(rho * (1.0 - rho) * (mu + gamma / beta - cp.diag(f_tilde)))

    # dOmega/dRe(Gamma)
    grad_re_gamma = (u.conj() @ (A_plus + diag_term) @ u.T).real

    # dOmega/dIm(Gamma)
    grad_im_gamma = (1j * u.conj() @ (A_minus + diag_term) @ u.T).real

    # Chain rule from Gamma to H
    grad_re_h = 0.5 * beta * (grad_re_gamma + grad_re_gamma.T)

    grad_im_h = 0.5 * beta * (grad_im_gamma - grad_im_gamma.T)

    return grad_re_h, grad_im_h


def nlcg(self, dm0=None):
    self.build()
    self.converged = False
    self.cycles = 0
    self.outer_cycles = 0
    self.nfev = 0
    self.refinements = 0
    self.message = ''

    h, unused_nelec = self._initial_h(dm0)
    state = self.calculate_cycle(h, mu=self.mu)
    gradient = [
        2.0 * (grad_re + 1j*grad_im)
        for grad_re, grad_im in (
            omega_gradient_wrt_h(h, f, self.beta, self.mu)
            for h, f in zip(state.h, state.fock)
        )
    ]
    direction = [-g for g in gradient]

    for unused_cycle in range(self.max_cycle):
        if state.residual_rms <= self.conv_tol:
            break

        slope = self._inner(gradient, direction)
        alpha = 1.0
        trial = self.calculate_cycle(
            [h + alpha*d for h, d in zip(state.h, direction)],
            mu=self.mu)
        while trial.grand_potential > state.grand_potential + 1e-4*alpha*slope:
            alpha *= 0.5
            trial = self.calculate_cycle(
                [h + alpha*d for h, d in zip(state.h, direction)],
                mu=self.mu)

        new_gradient = [
            2.0 * (grad_re + 1j*grad_im)
            for grad_re, grad_im in (
                omega_gradient_wrt_h(h, f, self.beta, self.mu)
                for h, f in zip(trial.h, trial.fock)
            )
        ]
        beta = max(0.0, self._inner(
            new_gradient,
            [new-old for new, old in zip(new_gradient, gradient)])
            / self._inner(gradient, gradient))
        direction = [
            -g + beta*d for g, d in zip(new_gradient, direction)]
        state = trial
        gradient = new_gradient
        self.cycles += 1

    converged = state.residual_rms <= self.conv_tol
    self.message = ('converged NLCG residual' if converged else
                    'maximum NLCG cycles reached')
    return self._finalize(state, converged)
