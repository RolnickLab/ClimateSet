import torch
import torch.nn as nn
from utils import Mask, MLP


class TSDCD(nn.Module):
    """Differentiable Causal Discovery for time series"""
    def __init__(self,
                 model_type: str,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int,
                 d: int,
                 tau: int,
                 tau_neigh: int,
                 instantaneous: bool,
                 hard_gumbel: bool):
        """
        Args:
            model_type: (fixed, free)
            num_layers: number of layers of each MLP
            num_hidden: number of hidden units of each MLP
            num_input: number of inputs of each MLP
            num_output: number of inputs of each MLP
            d: number of features
            tau: size of the timewindow
            tau_neigh: radius of neighbors to consider
            instantaneous: if True, models instantaneous connections
            hard_gumbel: if True, use hard sampling for the masks
        """
        super().__init__()
        self.distribution_type = "normal"
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output
        self.d = d
        self.tau = tau
        self.tau_neigh = tau_neigh
        self.instantaneous = instantaneous
        self.hard_gumbel = hard_gumbel

        if model_type == "fixed":
            self.cond_models = nn.ModuleList(MLP(num_layers, num_hidden,
                                                 num_input, num_output) for i
                                             in range(self.d))
            self.mask = Mask(d, 2 * tau_neigh + 1, tau,
                             instantaneous=instantaneous, latent=False, drawhard=hard_gumbel)
        elif model_type == "free":
            raise NotImplementedError

    def get_adj(self) -> torch.Tensor:
        """
        Returns:
            Matrices of the probabilities from which the masks are sampled
        """
        return self.mask.get_proba()

    def get_likelihood(self, y: torch.Tensor, mu: torch.Tensor, logvar:
                       torch.Tensor, iteration: int) -> torch.Tensor:
        """
        Returns the log-likelihood of data y given Gaussian distributions
        with parameters mu and logvar
        Args:
            y: data ()
            mu: mean of Gaussian distr
            logvar: log of the variance of Gaussian distr
            iteration: number of iterations performed during the training
        Returns:
            sum of the log-likelihoods
        """
        if self.distribution_type == "normal":
            std = 0.5 * torch.exp(logvar)
            conditionals = torch.distributions.Normal(mu, std)
            log_probs = conditionals.log_prob(y.view(-1, 1))
            return torch.sum(log_probs)
        else:
            raise NotImplementedError()

    def forward(self, x):
        # x shape: batch, time, feature, gridcell

        # sample mask and apply on x
        b = x.shape[0]

        y_hat = torch.zeros((b, self.d, x.shape[-1], 2))
        # TODO: remove loop
        for i_cell in range(x.shape[-1]):
            mask = self.mask(b)  # size: b x tau x d x (d x tau_neigh) x gridcells
            for i in range(self.d):
                # TODO: check if matches data generation
                lower_padding = max(0, -i_cell + self.tau_neigh)
                upper_padding = max(0, i_cell + self.tau_neigh - x.shape[-1] + 1)
                lower_cell = max(0, i_cell - self.tau_neigh)
                upper_cell = min(x.shape[-1], i_cell + self.tau_neigh) + 1
                x_ = x[:, :, :, lower_cell:upper_cell]
                if lower_padding > 0:
                    zeros = torch.zeros((b, x.shape[1], x.shape[2], lower_padding))
                    x_ = torch.cat((zeros, x_), dim=-1)
                elif upper_padding > 0:
                    zeros = torch.zeros((b, x.shape[1], x.shape[2], upper_padding))
                    x_ = torch.cat((x_, zeros), dim=-1)
                # print(y_hat[:, i, i_cell].size())
                # print(self.cond_models[i]((x_ * mask[:, :, i]).view(b, -1)).size())
                # __import__('ipdb').set_trace()
                # print(x_[0] * mask[0, :, i])
                # print(x_[0, -1, 0])
                y_hat[:, i, i_cell] = self.cond_models[i]((x_ * mask[:, :, i]).view(b, -1))
        return y_hat
