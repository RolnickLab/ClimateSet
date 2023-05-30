import numpy as np
import torch


class ALM:
    """
    Augmented Lagrangian Method
    To use the quadratic penalty method (e.g. for the acyclicity constraint),
    just ignore 'self.lambda'
    """
    def __init__(self,
                 mu_init: float,
                 mu_mult_factor: float,
                 omega_gamma: float,
                 omega_mu: float,
                 h_threshold: float,
                 min_iter_convergence: int,
                 dim_gamma = (1)):
        self.gamma = torch.zeros(*dim_gamma)
        self.delta_gamma = -np.inf
        self.mu = mu_init
        self.min_iter_convergence = min_iter_convergence
        self.h_threshold = h_threshold
        self.omega_mu = omega_mu
        self.omega_gamma = omega_gamma
        self.mu_mult_factor = mu_mult_factor
        self.stop_crit_window = 100
        self.constraint_violation = []
        self.has_converged = False
        self.dim_gamma = dim_gamma

    def _compute_delta_gamma(self, iteration: int, val_loss: list):
        # compute delta for gamma
        if iteration >= 2 * self.stop_crit_window and \
           iteration % (2 * self.stop_crit_window) == 0:
            t0, t_half, t1 = val_loss[-3], val_loss[-2], val_loss[-1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                self.delta_gamma = -np.inf
            else:
                self.delta_gamma = (t1 - t0) / self.stop_crit_window
        else:
            self.delta_gamma = -np.inf  # do not update gamma nor mu

    def update(self, iteration: int, h_list: list, val_loss: list):
        """
        Update the value of mu and gamma. Return True if it has converged.
        Args:
            iteration: number of training iterations completed
            h_list: list of the values of the constraint
            val_loss: list of validation loss
        """
        self.has_increased_mu = False

        if len(val_loss) >= 3:
            h = h_list[-1]
            if len(self.dim_gamma) > 1:
                h_scalar = torch.sum(h)
            else:
                h_scalar = h

            # check if QPM has converged
            if iteration > self.min_iter_convergence and h_scalar <= self.h_threshold:
                self.has_converged = True
            else:
                # update delta_gamma
                self._compute_delta_gamma(iteration, val_loss)

                # if we have found a stationary point of the augmented loss
                if abs(self.delta_gamma) < self.omega_gamma or self.delta_gamma > 0:
                    self.gamma += self.mu * h
                    self.constraint_violation.append(h_scalar)

                    # increase mu if the constraint has sufficiently decreased
                    # since the last subproblem
                    if len(self.constraint_violation) >= 2:
                        if h_scalar > self.omega_mu * self.constraint_violation[-2]:
                            self.mu *= self.mu_mult_factor
                            self.has_increased_mu = True
