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
                 tau: int,
                 tau_neigh: int):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output
        self.d = num_input
        self.tau = tau
        self.tau_neigh = tau_neigh

        if model_type == "fixed":
            self.cond_model = MLP(num_layers, num_hidden, num_input, num_output)
            self.mask = Mask(num_input, tau_neigh, tau, drawhard=True)
        elif model_type == "free":
            raise NotImplementedError

    def get_adj(self):
        return self.mask.get_proba()

    def get_likelihood(self, x, mu, std, iteration):
        if self.distribution_type == "normal":
            conditionals = torch.distributions.Normal(mu, std)
            log_probs = conditionals.log_prob(x)
            return torch.sum(log_probs)
        else:
            raise NotImplementedError()

    def forward(self, x):
        # sample mask and apply on x
        b = x.shape[0]
        d = x.shape[1]
        __import__('ipdb').set_trace()
        mask = self.mask(b)  # size: b x d x (d x tau_neigh) x tau
        # mask = mask.view(b, d, -1) # size: b x d x (d x tau_neigh x tau)


        # torch.einsum(",bijt->", x, mask)


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
        self.param = nn.Parameter(torch.ones((d, d * tau_neigh, tau)) * 5)

    def forward(self, bs: int, tau: float = 1) -> torch.Tensor:
        """
        :param bs: batch size
        :param tau: temperature constant for sampling
        """
        if not self.fixed:
            adj = gumbel_sigmoid(self.param, self.uniform, bs, tau=tau, hard=self.drawhard)
            return adj
        else:
            assert self.fixed_output is not None
            return self.fixed_output.repeat(bs, 1, 1)

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
        for layer in range(num_layers):
            in_features = num_hidden
            out_features = num_hidden

            if layer == 0:
                in_features = num_input
            if layer == num_layers - 1:
                out_features = num_output

            module_dict[f'lin{layer}'] = nn.Linear(in_features, out_features)
            if layer != num_layers - 1:
                module_dict[f'nonlin{layer}'] = nn.ReLU()

        self.model = nn.Sequential(module_dict)

    def forward(self, x, eta) -> torch.Tensor:
        x_ = torch.cat((x, eta), dim=1)
        return self.model(x_)


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
