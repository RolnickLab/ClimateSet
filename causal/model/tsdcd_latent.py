import torch
import time
import torch.nn as nn
import torch.distributions as distr
from .utils import Mask, MLP


class LatentTSDCD(nn.Module):
    """Differentiable Causal Discovery for time series with latent variables"""
    def __init__(self,
                 num_layers: int,
                 num_hidden: int,
                 num_input: int,
                 num_output: int,
                 coeff_kl: float,
                 distr_z0: str,
                 distr_encoder: str,
                 distr_transition: str,
                 distr_decoder: str,
                 d: int,
                 d_x: int,
                 d_z: int,
                 tau: int,
                 instantaneous: bool,
                 hard_gumbel: bool,
                 no_gt: bool,
                 debug_gt_graph: bool,
                 debug_gt_z: bool,
                 debug_gt_w: bool,
                 gt_graph: torch.tensor = None,
                 gt_w: torch.tensor = None,
                 tied_w: bool = False):
        """
        Args:
            num_layers: number of layers of each MLP
            num_hidden: number of hidden units of each MLP
            num_input: number of inputs of each MLP
            num_output: number of inputs of each MLP
            coeff_kl: coefficient of the KL term

            distr_z0: distribution of the first z (gaussian)
            distr_encoder: distribution parametrized by the encoder (gaussian)
            distr_transition: distribution parametrized by the transition model (gaussian)
            distr_decoder: distribution parametrized by the decoder (gaussian)

            d: number of features
            d_x: number of grid locations
            d_z: number of latent variables
            tau: size of the timewindow
            instantaneous: if True, models instantaneous connections
            hard_gumbel: if True, use hard sampling for the masks

            no_gt: if True, do not use any ground-truth data (useful with realworld dataset)
            debug_gt_graph: if True, set the masks to the ground-truth graphes (gt_graph)
            debug_gt_z: if True, use directly the ground-truth z (gt_z sampled with the data)
            debug_gt_w: if True, set the matrices W to the ground-truth W (gt_w)
            gt_graph: Ground-truth graphes, only used if debug_gt_graph is True
            gt_w: Ground-truth W, only used if debug_gt_w is True
        """
        super().__init__()

        # nn encoder hyperparameters
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output
        self.coeff_kl = coeff_kl

        self.d = d
        self.d_x = d_x
        self.d_z = d_z
        self.tau = tau
        self.instantaneous = instantaneous
        self.hard_gumbel = hard_gumbel
        self.no_gt = no_gt
        self.debug_gt_graph = debug_gt_graph
        self.debug_gt_z = debug_gt_z
        self.debug_gt_w = debug_gt_w
        self.tied_w = tied_w

        if self.no_gt:
            self.gt_w = None
            self.gt_graph = None
        else:
            self.gt_w = torch.tensor(gt_w).double()
            self.gt_graph = torch.tensor(gt_graph).double()

        if distr_z0 == "gaussian":
            self.distr_z0 = torch.normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_transition == "gaussian":
            self.distr_transition = torch.normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_encoder == "gaussian":
            self.distr_encoder = torch.normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_decoder == "gaussian":
            self.distr_decoder = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        self.encoder_decoder = EncoderDecoder(self.d, self.d_x, self.d_z, self.debug_gt_w, self.gt_w, self.tied_w)
        self.transition_model = TransitionModel(self.d, self.d_z, self.tau,
                                                self.num_layers,
                                                self.num_hidden,
                                                self.num_output)

        self.mask = Mask(d, d_z, tau, instantaneous=instantaneous, latent=True, drawhard=hard_gumbel)
        if self.debug_gt_graph:
            if self.instantaneous:
                self.mask.fix(self.gt_graph)
            else:
                self.mask.fix(self.gt_graph[:-1])

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def encode(self, x, y):
        """
        encode X and Y into latent variables Z
        """
        b = x.size(0)
        z = torch.zeros(b, self.tau + 1, self.d, self.d_z)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        # sample Zs
        for i in range(self.d):
            # get params from the encoder q(z^t | x^t)
            for t in range(self.tau):
                q_mu, q_logvar = self.encoder_decoder(x[:, t, i], i, encoder=True)  # torch.matmul(self.W, x)

                # reparam trick
                q_std = torch.exp(0.5 * q_logvar)
                z[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

            q_mu, q_logvar = self.encoder_decoder(y[:, i], i, encoder=True)  # torch.matmul(self.W, x)
            q_std = torch.exp(0.5 * q_logvar)
            z[:, -1, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())
            mu[:, i] = q_mu
            std[:, i] = q_std

        return z, mu, std

    def transition(self, z, mask):
        b = z.size(0)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        for i in range(self.d):
            pz_params = torch.zeros(b, self.d_z, 2)
            for k in range(self.d_z):
                pz_params[:, k] = self.transition_model(z, mask[:, :, i * self.d_z + k], i, k)
            mu[:, i] = pz_params[:, :, 0]
            # factor to ensure that doesnt output inf when training is starting
            # factor = 1e-10
            # std[:, i] = 0.5 * torch.exp(factor * pz_params[:, :, 1])
            std[:, i] = torch.exp(0.5 * pz_params[:, :, 1])

        return mu, std

    def decode(self, z):
        mu = torch.zeros(z.size(0), self.d, self.d_x)
        std = torch.zeros(z.size(0), self.d, self.d_x)

        for i in range(self.d):
            px_mu, px_logvar = self.encoder_decoder(z[:, i], i, encoder=False)
            mu[:, i] = px_mu
            std[:, i] = torch.exp(0.5 * px_logvar)

        return mu, std

    def forward(self, x, y, gt_z, iteration):
        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y)

        if self.debug_gt_z:
            z = gt_z

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)
        pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

        # get params from decoder p(x^t | z^t)
        px_mu, px_std = self.decode(z[:, -1])

        # set distribution with obtained parameters
        p = distr.Normal(pz_mu.view(b, -1), pz_std.view(b, -1))
        # test with fixed var: torch.ones_like(pz_mu.view(b, -1)) * 0.01)
        q = distr.Normal(q_mu_y.view(b, -1), q_std_y.view(b, -1))
        px_distr = self.distr_decoder(px_mu, px_std)

        # compute the KL, the reconstruction and the ELBO
        # __import__('ipdb').set_trace()
        # print(pz_mu.shape)
        # print(pz_mu.view(b, -1).shape)
        # kl = distr.kl_divergence(p, q).mean()
        # kl = distr.kl_divergence(q, p).mean()
        kl = torch.mean(0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2)) + 0.5 * (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2 - 0.5, 2).mean()

        # kl2 = self.get_kl(pz_mu, pz_std, q_mu_y, q_std_y)
        # print(kl2.mean())
        assert kl >= 0, f"KL={kl} has to be >= 0"
        recons = torch.mean(px_distr.log_prob(y))

        # __import__('ipdb').set_trace()
        # print(torch.mean((px_mu - y)**2))
        elbo = recons - self.coeff_kl * kl

        return elbo, recons, kl, px_mu

    def get_kl(self, mu1, sigma1, mu2, sigma2) -> float:
        """KL between two multivariate Gaussian Q and P.
        Here, Q is spherical and P is diagonal"""
        kl = 0.5 * (torch.log(torch.prod(sigma2, dim=1) / torch.prod(sigma1, dim=1)) +
                    torch.sum(sigma1 / sigma2, dim=1) - self.d_z +
                    torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / sigma2), mu2 - mu1))
        # kl = 0.5 * (torch.log(torch.prod(sigma2, dim=1) / sigma1 ** self.d_z) +
        #             torch.sum(sigma1 / sigma2, dim=1) - self.d_z +
        #             torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / sigma2), mu2 - mu1))
        if torch.sum(kl) < 0:
            __import__('ipdb').set_trace()
            print(sigma2 ** self.d_z)
            print(torch.prod(sigma1, dim=1))
            print(torch.sum(torch.log(sigma2 ** self.d_z / torch.prod(sigma1, dim=1))))
            print(torch.sum(torch.sum(sigma1 / sigma2, dim=1)))
            # print(torch.sum(torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / s_p), mu2 - mu1)))

        return torch.sum(kl)


class EncoderDecoder(nn.Module):
    """Combine an encoder and a decoder, particularly useful when W is a shared
    parameter."""
    def __init__(self, d: int, d_x: int, d_z: int, debug_gt_w: bool, gt_w:
                 torch.tensor = None, tied_w: bool = False):
        """
        Args:
            d: number of features
            d_x: dimensionality of grid locations
            d_z: dimensionality of latent variables
            debug_gt_w: if True, set W as gt_w
            gt_w: ground-truth W
            tied_w: if True, the encoder use the transposed
            of the matrix W from the decoder
        """
        super().__init__()
        self.use_grad_projection = True
        self.tied_w = tied_w

        self.d = d
        self.d_x = d_x
        self.d_z = d_z
        self.debug_gt_w = debug_gt_w
        self.gt_w = gt_w

        unif = (1 - 0.4) * torch.rand(size=(d, d_x, d_z)) + 0.4
        if self.use_grad_projection:
            self.w = nn.Parameter(unif / torch.tensor(self.d_z))
        else:
            # otherwise, self.w is the log of W
            self.w = nn.Parameter(torch.log(unif) - torch.log(torch.tensor(self.d_z)))

        if self.tied_w:
            self.w_q = None
        else:
            unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
            self.w_q = nn.Parameter(unif / torch.tensor(self.d_z))

        self.logvar_encoder = nn.Parameter(torch.ones(d) * 0.01)
        self.logvar_decoder = nn.Parameter(torch.ones(d) * 0.01)

    def forward(self, x, i, encoder: bool):
        w = self.get_w(encoder)[i]

        if encoder:
            # encoder q(z | x)
            mu = torch.matmul(x, w)
            logvar = self.logvar_encoder[i]
        else:
            # here x is in fact z
            # decoder p(x | z)
            # if self.tied_w:
            w = w.T
            mu = torch.matmul(x, w)
            logvar = self.logvar_decoder[i]
        return mu, logvar

    def get_w(self, encoder: bool = False) -> torch.tensor:
        if self.tied_w:
            if self.debug_gt_w:
                w = self.gt_w
            elif self.use_grad_projection:
                w = self.w
            else:
                w = torch.exp(self.w)
        else:
            if encoder:
                w = self.w_q
            else:
                if self.debug_gt_w:
                    w = self.gt_w
                else:
                    w = self.w

        return w

    def project_gradient(self):
        assert self.use_grad_projection
        with torch.no_grad():
            self.w.clamp_(min=0.)
        assert torch.min(self.w) >= 0.


class TransitionModel(nn.Module):
    """ Models the transitions between the latent variables Z with neural networks.  """
    def __init__(self, d: int, d_z: int, tau: int, num_layers: int, num_hidden:
                 int, num_output: int = 2):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
            num_output: number of outputs
        """
        super().__init__()
        self.d = d
        self.d_z = d_z
        self.tau = tau

        # initialize NNs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.nn = nn.ModuleList(MLP(num_layers, num_hidden, d * d_z * tau, self.num_output) for i in range(d * d_z))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i, k):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k
        NN(G_{tau-1} * z_{t-1}, ..., G_{tau-k} * z_{t-k})
        """
        # t_total = torch.max(self.tau, z_past.size(1))  # TODO: find right dim
        # param_z = torch.zeros(z_past.size(0), 2)
        z = z.view(mask.size())
        masked_z = (mask * z).view(z.size(0), -1)

        param_z = self.nn[i * self.d_z + k](masked_z)
        # param_z = self.nn(masked_z)

        return param_z
