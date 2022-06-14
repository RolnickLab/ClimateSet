import torch
import torch.nn as nn
import torch.distributions as distr
from collections import OrderedDict


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
            self.mask = Mask(d, tau_neigh, tau, instantaneous=instantaneous, drawhard=hard_gumbel)
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


class LatentTSDCD(nn.Module):
    """Differentiable Causal Discovery for time series with latent variables"""
    def __init__(self,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int,
                 distr_z0: str,
                 distr_encoder: str,
                 distr_transition: str,
                 distr_decoder: str,
                 d: int,
                 d_x: int,
                 k: int,
                 tau: int,
                 instantaneous: bool,
                 hard_gumbel: bool):
        super().__init__()
        self.num_layers_enc = num_layers
        self.num_hidden_enc = num_hidden
        self.num_input_enc = num_input
        self.num_output_enc = num_output

        self.d = d
        self.d_x = d_x
        self.k = k
        self.tau = tau
        self.instantaneous = instantaneous
        self.hard_gumbel = hard_gumbel

        if distr_z0 == "gaussian":
            self.distr_z0 = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_transition == "gaussian":
            self.distr_transition = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_encoder == "gaussian":
            self.distr_encoder = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_decoder == "gaussian":
            self.distr_decoder = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        self.decoder = Encoder(self.d, self.d_x, self.k)
        self.encoder = Decoder(self.decoder.w)
        self.transition_model = TransitionModel(self.d, self.tau)

        self.mask = Mask(d, k, tau, instantaneous=instantaneous, drawhard=hard_gumbel)

    def get_adj(self):
        return self.mask.get_proba()

    def forward(self, x, y):
        bs = x.size(0)
        t_total = x.size(1)

        z_past = torch.zero(bs, t_total, self.d, self.k)
        elbo = torch.tensor([0])

        # TODO: be more efficient without the loop
        for i in range(self.d):
            for t in range(t_total):
                q_params = self.encoder(x[:, t, i])  # torch.matmul(self.W, x)

                # reparam trick
                q_mu = q_params[:, 0]
                q_std = 0.5 * torch.exp(q_params[:, 1])
                z_past[:, t, i] = q_mu + q_std * self.encoder_distr.sample(0, 1)

            pz_params = self.transition_model(z_past[:, :, i])
            px_params = self.decoder(z_past[:, :, i])
            kl = self.get_kl(q_mu, q_std, pz_params[:, 0], pz_params[:, 1])

            px_std = 0.5 * torch.exp(px_params[:, 1])
            px_distr = self.distr_encoder(px_params[:, 0], px_std)
            recons = torch.sum(px_distr.log_prob(y))

            elbo += recons - kl

        return elbo

    def get_kl(self, mu_q, sigma_q, mu_p, sigma_p) -> float:
        """KL between two multivariate Gaussian Q and P.
        Here, Q is spherical and P is diagonal"""
        d = mu_q.size(1)  # the number of features
        # k = mu_q.size(2)  # the number of cluster
        kl = torch.tensor([0])

        for i in range(d):
            m_p = mu_p[:, i]
            m_q = mu_q[:, i]
            s_p = sigma_p[:, i]
            s_q = sigma_q[:, i]
            kl += 0.5 * (torch.log(torch.prod(s_q) / s_p) +
                         s_q * torch.sum(1 / s_p) +
                         torch.dot(torch.dot((m_p - m_q), 1 / s_p), (m_p - m_q)))

        return kl


class Decoder(nn.Module):
    def __init__(self, d: int, d_x: int, k: int):
        self.d = d
        self.d_x = d_x
        self.k = k
        # make it more general for NN ?
        self.w = torch.rand(d, d_x, k)  # TODO: might have a better initialization
        self.var = torch.rand(d)

    def forward(self, x):
        # TODO: for all i
        mu = nn.Functional.linear(x, self.w)
        return mu, self.var


class Encoder(nn.Module):
    def __init__(self, w: torch.Tensor):
        self.w = w.T  # transpose -1 and -2
        self.d = self.w.size(0)
        self.d_x = self.w.size(1)
        self.k = self.w.size(2)
        self.var = torch.rand(self.d)

    def forward(self, x):
        # TODO: for all i
        mu = nn.Functional.linear(x, self.w)
        return mu, self.var


class TransitionModel(nn.Module):
    def __init__(self, d: int, k: int, tau: int, num_layers: int, num_hidden:
                 int, num_output: int = 2):
        self.d = d
        self.k = k
        self.tau = tau

        # initialize NNs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.nn = nn.ModuleList([MLP(num_layers, num_hidden, d * k * (i+1), self.num_output) for i in range(d)])

    def forward(self, z_past, mask):
        """Returns the params of N(z_t | z_{<t})
        NN(G_{t-k} * z_{t-1}, ..., G_{t-k} * z_{t-k})
        """
        # t_total = torch.max(self.tau, z_past.size(1))  # TODO: find right dim
        param_z = torch.zeros(z_past.size(0), self.d, 2)

        masked_z = mask * z_past
        # TODO: more efficient with einsum ?
        for i in range(self.d):
            param_z[:, i] = self.nn[i](masked_z)

        return param_z


class Mask(nn.Module):
    def __init__(self, d: int, tau_neigh: int, tau: int, instantaneous: bool, drawhard: bool):
        super().__init__()

        self.d = d
        self.instantaneous = instantaneous
        if self.instantaneous:
            self.tau = tau + 1
        else:
            self.tau = tau
        self.tau_neigh = tau_neigh
        self.drawhard = drawhard
        self.fixed = False
        self.fixed_output = None
        self.uniform = distr.uniform.Uniform(0, 1)

        if self.instantaneous:
            # initialize mask as log(mask_ij) = 1
            self.param = nn.Parameter(torch.ones((tau + 1, d, d, (2 * tau_neigh + 1))) * 5)
            self.fixed_mask = torch.ones_like(self.param)
            # set diagonal 0 for G_t0
            self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
            # TODO: set neighbors to 0
            # self.fixed_mask[:, :, :, tau_neigh] = 0
        else:
            # initialize mask as log(mask_ij) = 1
            self.param = nn.Parameter(torch.ones((tau, d, d, (2 * tau_neigh + 1))) * 5)
            self.fixed_mask = torch.ones_like(self.param)

    def forward(self, b: int, tau: float = 1) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """
        if not self.fixed:
            adj = gumbel_sigmoid(self.param, self.uniform, b, tau=tau, hard=self.drawhard)
            adj = adj * self.fixed_mask
            return adj
        else:
            assert self.fixed_output is not None
            return self.fixed_output.repeat(b, 1, 1, 1, 1)

    def get_proba(self) -> torch.Tensor:
        if not self.fixed:
            return torch.sigmoid(self.param) * self.fixed_mask
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
