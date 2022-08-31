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
                 distr_z0: str,
                 distr_encoder: str,
                 distr_transition: str,
                 distr_decoder: str,
                 d: int,
                 d_x: int,
                 k: int,
                 tau: int,
                 instantaneous: bool,
                 hard_gumbel: bool,
                 debug_gt_graph: bool,
                 debug_gt_z: bool,
                 debug_gt_w: bool,
                 gt_graph: torch.tensor = None,
                 gt_w: torch.tensor = None):
        """
        Args:
            num_layers: number of layers of each MLP
            num_hidden: number of hidden units of each MLP
            num_input: number of inputs of each MLP
            num_output: number of inputs of each MLP

            distr_z0: distribution of the first z (gaussian)
            distr_encoder: distribution parametrized by the encoder (gaussian)
            distr_transition: distribution parametrized by the transition model (gaussian)
            distr_decoder: distribution parametrized by the decoder (gaussian)

            d: number of features
            d_x: number of grid locations
            k: number of latent variables
            tau: size of the timewindow
            instantaneous: if True, models instantaneous connections
            hard_gumbel: if True, use hard sampling for the masks

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

        self.d = d
        self.d_x = d_x
        self.k = k
        self.tau = tau
        self.instantaneous = instantaneous
        self.hard_gumbel = hard_gumbel
        self.debug_gt_graph = debug_gt_graph
        self.debug_gt_z = debug_gt_z
        self.debug_gt_w = debug_gt_w
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

        self.encoder_decoder = EncoderDecoder(self.d, self.d_x, self.k, self.debug_gt_w, self.gt_w)
        # self.encoder = Encoder(self.d, self.d_x, self.k)  # self.decoder.w
        self.transition_model = TransitionModel(self.d, self.k, self.tau,
                                                self.num_layers,
                                                self.num_hidden,
                                                self.num_output)

        self.mask = Mask(d, k, tau, instantaneous=instantaneous, latent=True, drawhard=hard_gumbel)
        if self.debug_gt_graph:
            self.mask.fix(self.gt_graph)

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def encode(self, x, y):
        """
        encode X and Y into latent variables Z
        size X = (b, t, d, d_x)
        size Y = (b, d, d_x)
        size Z = (b, t, d, k)
        Returns:
            z, mu, std: the Z sampled based on a Gaussian with parameters mu and std
        """
        # concatenate x and y along the time dimension
        x = torch.cat((x, y.unsqueeze(1)), 1)

        # get distr parameters and sample Zs
        mu, logvar = self.encoder_decoder(x, encoder=True)  # torch.matmul(self.W, x)
        std = 0.5 * torch.exp(logvar)
        z = mu + std * self.distr_encoder(0, 1, size=mu.size())

        return z, mu, std

    def transition(self, z, mask):
        """
        size Z = (b, t, d, k)
        """
        b = z.size(0)
        mu = torch.zeros(b, self.d, self.k)
        std = torch.zeros(b, self.d, self.k)

        for i in range(self.d):
            pz_params = torch.zeros(b, self.k, 2)
            for k in range(self.k):
                pz_params[:, k] = self.transition_model(z, mask[:, :, i * self.k + k], i, k)
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = 0.5 * torch.exp(pz_params[:, :, 1])

        return mu, std

    def decode(self, z):
        """
        Decode Z to X: return parameters for p(x^t | z^t)
        size Z = (b, 1, d, k)
        size mu = (b, 1, d, d_x)
        Returns:
            mu, std: parameters for p(x^t | z^t)
        """
        # TODO: check dimension
        mu, logvar = self.encoder_decoder(z, encoder=False)
        std = 0.5 * torch.exp(logvar)

        return mu, std

    def forward(self, x, y, gt_z, iteration):
        """
        Args:
            x: input
            y: output
            gt_z: ground-truth Z (latent variables). If debug_gt_z is True, set z = gt_z
            iteration: number of training iterations
        """
        b = x.size(0)

        # sample Zs (based on X)
        st = time.time()
        z, q_mu, q_std = self.encode(x, y)
        print(f"encode time: {time.time() - st}")

        if self.debug_gt_z:
            z = gt_z

        # get params of the transition model p(z^t | z^{<t})
        st = time.time()
        mask = self.mask(b)
        pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)
        print(f"transition time: {time.time() - st}")

        # get params from decoder p(x^t | z^t)
        st = time.time()
        px_mu, px_std = self.decode(z[:, -1:])
        print(f"decode time: {time.time() - st}")

        # set distribution with obtained parameters
        st = time.time()
        p = distr.Normal(pz_mu.view(b, -1), pz_std.view(b, -1))
        q = distr.Normal(q_mu[:, -1].reshape(b, -1), q_std[:, -1].reshape(b, -1))
        px_distr = self.distr_decoder(px_mu, px_std)
        print(f"sampling time: {time.time() - st}")

        # compute the KL, the reconstruction and the ELBO
        st = time.time()
        kl = distr.kl_divergence(p, q).mean()
        assert kl >= 0, f"KL={kl} has to be >= 0"
        recons = torch.mean(px_distr.log_prob(y))
        elbo = recons - 0.0001 * kl
        print(f"kl time: {time.time() - st}")
        __import__('ipdb').set_trace()

        return elbo, recons, kl

    def get_kl(self, mu1, sigma1, mu2, sigma2) -> float:
        """KL between two multivariate Gaussian Q and P.
        Here, Q is spherical and P is diagonal"""
        kl = 0.5 * (torch.log(torch.prod(sigma2, dim=1) / torch.prod(sigma1, dim=1)) +
                    torch.sum(sigma1 / sigma2, dim=1) - self.k +
                    torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / sigma2), mu2 - mu1))
        # kl = 0.5 * (torch.log(torch.prod(sigma2, dim=1) / sigma1 ** self.k) +
        #             torch.sum(sigma1 / sigma2, dim=1) - self.k +
        #             torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / sigma2), mu2 - mu1))
        if torch.sum(kl) < 0:
            __import__('ipdb').set_trace()
            print(sigma2 ** self.k)
            print(torch.prod(sigma1, dim=1))
            print(torch.sum(torch.log(sigma2 ** self.k / torch.prod(sigma1, dim=1))))
            print(torch.sum(torch.sum(sigma1 / sigma2, dim=1)))
            # print(torch.sum(torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / s_p), mu2 - mu1)))

        return torch.sum(kl)


class EncoderDecoder(nn.Module):
    """Combine an encoder and a decoder, particularly useful when W is a shared
    parameter."""
    def __init__(self, d: int, d_x: int, k: int, debug_gt_w: bool, gt_w:
                 torch.tensor = None):
        """
        Args:
            d: number of features
            d_x: number of grid locations
            k: number of latent variables
            debug_gt_w: if True, set W as gt_w
            gt_w: ground-truth W
        """
        super().__init__()
        self.d = d
        self.d_x = d_x
        self.k = k
        self.debug_gt_w = debug_gt_w
        self.gt_w = gt_w

        self.log_w = nn.Parameter(torch.rand(size=(d, d_x, k)) - 0.5)
        # self.logvar_encoder = nn.Parameter(torch.rand(d) * 0.1)
        # self.logvar_decoder = nn.Parameter(torch.rand(d) * 0.1)
        self.logvar_decoder = torch.log(torch.ones(d, d_x) * 0.001)  # TODO: test
        self.logvar_encoder = torch.log(torch.ones(d, k) * 0.001)  # TODO: test

    def forward(self, x, encoder: bool):
        w = self.get_w()

        if encoder:
            # size X: (b, t, d, d_x), size W: (d, d_x, k),
            # size mu: (b, t, d, k), size logvar: (d, k)
            # sum along the d_x dimension
            # st = time.time()
            mu = torch.einsum("lmij, ijk -> lmik", x, w)
            # print(f"time for einsum: {time.time() - st}")
            # st = time.time()
            # mu = torch.matmul(x[0, 0], w[0])
            # print(f"time for matmul: {(time.time() -st) * x.size(0) * x.size(1)}")
            logvar = self.logvar_encoder
            logvar_repeated = logvar.repeat(x.size(0), x.size(1), 1, 1)
            # print(logvar_repeated.size())
        else:
            # here x is in fact z
            # size Z: (b, t, d, k), size W: (d, d_x, k)
            # size mu: (b, t, d, d_x), size logvar: (d, k)
            # sum on the k dimensions
            mu = torch.einsum("lmij, ikj -> lmik", x, w)
            # mu = torch.matmul(x[i], w[i].T)
            logvar = self.logvar_decoder

        logvar_repeated = logvar.repeat(x.size(0), x.size(1), 1, 1)
        return mu, logvar_repeated

    def get_w(self) -> torch.tensor:
        if self.debug_gt_w:
            w = self.gt_w
        else:
            w = torch.exp(self.log_w)
        return w


class Decoder(nn.Module):
    """ Decode the latent variables Z into the estimation of observable data X
    using a linear model parametrized by W^T """
    def __init__(self, d: int, d_x: int, k: int):
        """
        Args:
            d: number of features
            d_x: number of grid locations
            k: number of latent variables
        """
        # TODO: might want to remove this class and Encoder if we only use EncoderDecoder
        # TODO: make it more general for NN ?
        # TODO: might want to consider alternative initialization for W and var
        super().__init__()
        self.d = d
        self.d_x = d_x
        self.k = k
        self.w = nn.Parameter(torch.rand(size=(d, d_x, k)) - 0.5)
        self.var = torch.rand(d)

    def forward(self, z, i):
        """
        Args:
            z: the latent variables Z
            i: a specific feature
        Returns:
            mu, var: the parameter of a Gaussian from which X can be sampled
        """
        # mu = nn.functional.linear(x, self.w[i])
        mu = torch.matmul(z, self.w[i].T)
        return mu, self.var[i]


class Encoder(nn.Module):
    """ Encode the observable data X into latent variables Z using a linear model parametrized by W """
    def __init__(self, d: int, d_x: int, k: int):
        """
        Args:
            d: number of features
            d_x: number of grid locations
            k: number of latent variables
        """
        super().__init__()
        self.d = d
        self.d_x = d_x
        self.k = k
        self.w = nn.Parameter(torch.rand(size=(d, d_x, k)) - 0.5)
        self.var = torch.rand(self.d)

    def forward(self, x, i):
        """
        Args:
            x: the observable data X
            i: a specific feature
        Returns:
            mu, var: the parameter of a Gaussian from which Z can be sampled
        """
        mu = torch.matmul(x, self.w[i])
        return mu, self.var[i]


class TransitionModel(nn.Module):
    """ Models the transitions between the latent variables Z with neural networks.  """
    def __init__(self, d: int, k: int, tau: int, num_layers: int, num_hidden:
                 int, num_output: int = 2):
        """
        Args:
            d: number of features
            k: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
            num_output: number of outputs
        """
        super().__init__()
        self.d = d
        self.k = k
        self.tau = tau

        # initialize NNs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.nn = nn.ModuleList(MLP(num_layers, num_hidden, d * k * tau, self.num_output) for i in range(d * k))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i, k):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k
        NN(G_{t-k} * z_{t-1}, ..., G_{t-k} * z_{t-k})
        """
        z = z.view(mask.size())
        masked_z = (mask * z).view(z.size(0), -1)
        param_z = self.nn[i * self.k + k](masked_z)

        return param_z
