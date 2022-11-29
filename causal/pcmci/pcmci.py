import numpy as np
from metrics import shd
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, CMIknn
from savar.dim_methods import get_varimax_loadings_standard as varimax


def varimax_pcmci(data, hp):
    """
    Apply Varimax+ to find modes and then PCMCI to recover the causal graph
    relating the different modes.
    args:
        data:
        hp: hyperparameters
    """
    if not hp.latent:
        raise NotImplementedError("The case without is not implemented.")
    if hp.instantaneous:
        raise NotImplementedError("The case with contemporaneous relations is not implemented.")

    # Options: ind_test, tau_max, pc_alpha
    tau_min = hp.tau_min
    tau_max = hp.tau
    d_x = hp.d_x
    d_z = hp.d_z
    pc_alpha = hp.pc_alpha

    if hp.ci_test == "linear"
        ind_test = ParCorr()
    elif hp.ci_test == "nonlinear":
        ind_test = CMIknn()
    else:
        raise ValueError(f"{hp.ci_test} is not valid as a CI test. It should be either 'linear' or 'nonlinear'")

    n = 50

    # data.shape = T x (res in 1D)
    data_raw = np.random.rand(d_x, n)

    # 1 - Apply varimax+ to the data in order to find W
    # (the relations from the grid locations to the modes)
    modes = varimax(data_raw, max_comps=d_z)

    # Get matrix W, apply it to grid-level
    W = modes['weights']
    data = data_raw @ W

    # TODO: plot found modes...

    # 2 - Apply PCMCI to the modes
    data = pp.DataFrame(data)
    pcmci = PCMCI(dataframe=data, cond_ind_test=ind_test)
    results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    str_graph = results['graph']
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']

    # 3 - convert output of PCMCI
    graph = np.zeros((tau_max + 1, d_z, d_z))
    for t in range(tau_max + 1):
    for i in range(d_z):
        for j in range(d_z):
            if str_graph[i, j, t] == "":
                graph[t, i, j] = 0
            elif str_graph[i, j, t] == "-->":
                graph[t, i, j] = 1
            else:
                raise ValueError("the output of PCMCI contains unknown symbols")

    return graph
