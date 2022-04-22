import torch
import torch.nn as nn
import torch.distributions as distr
from collections import OrderedDict


class CausalModel(nn.Module):
    def __init__(self,
                 model_type: str,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int,
                 d: int,
                 tau: int,
                 tau_neigh: int,
                 hard_gumbel: bool):
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
        self.hard_gumbel = hard_gumbel

        if model_type == "fixed":
            self.cond_models = nn.ModuleList(MLP(num_layers, num_hidden,
                                                 num_input, num_output) for i
                                             in range(self.d))
            self.mask = Mask(d, tau_neigh, tau, drawhard=hard_gumbel)
        elif model_type == "free":
            raise NotImplementedError

    def get_adj(self):
        return self.mask.get_proba()

    def get_likelihood(self, y, mu, logvar, iteration):
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
            mask = self.mask(b)  # size: b x d x (d x tau_neigh) x tau
            for i in range(self.d):
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
                y_hat[:, i, i_cell] = self.cond_models[i]((x_ * mask[:, :, i]).view(b, -1))
        return y_hat


class Mask(nn.Module):
    def __init__(self, d: int, tau_neigh: int, tau: int, drawhard: bool):
        super().__init__()

        self.d = d
        self.tau = tau
        self.tau_neigh = tau_neigh
        self.drawhard = drawhard
        self.fixed = False
        self.fixed_output = None
        self.uniform = distr.uniform.Uniform(0, 1)

        # initialize mask as log(mask_ij) = 1
        self.param = nn.Parameter(torch.ones((tau, d, d, (2 * tau_neigh + 1))) * 5)

    def forward(self, b: int, tau: float = 1) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """
        if not self.fixed:
            adj = gumbel_sigmoid(self.param, self.uniform, b, tau=tau, hard=self.drawhard)
            return adj
        else:
            assert self.fixed_output is not None
            return self.fixed_output.repeat(b, 1, 1, 1, 1)

    def get_proba(self) -> torch.Tensor:
        if not self.fixed:
            return torch.sigmoid(self.param)  # XXX
        else:
            return self.fixed_output

    def fix(self, fixed_output):
        self.fixed_output = fixed_output
        self.fixed = True


class MLP(nn.Module):
    def __init__(self,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output

        module_dict = OrderedDict()

        # create model layer by layer
        in_features = num_input
        out_features = num_hidden
        if num_layers == 0:
            out_features = num_output

        module_dict['lin0'] = nn.Linear(in_features, out_features)

        for layer in range(num_layers):
            in_features = num_hidden
            out_features = num_hidden

            if layer == num_layers - 1:
                out_features = num_output

            module_dict[f'nonlin{layer}'] = nn.ReLU()
            module_dict[f'lin{layer+1}'] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft

    return y
