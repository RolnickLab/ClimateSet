import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

# Code for MCC adapted from https://github.com/slachapelle/disentanglement_via_mechanism_sparsity/blob/main/metrics.py
# def get_linear_score(x, y):
#     reg = LinearRegression().fit(x, y)
#     return reg.score(x, y)
# 
# 
# def linear_regression_metric(model, data_loader, device, num_samples=int(1e5), indices=None, opt=None):
#     with torch.no_grad():
#         if model.latent_model.z_block_size != 1:
#             raise NotImplementedError("This function is implemented only for z_block_size == 1")
#         model.eval()
#         z_list = []
#         z_hat_list = []
#         sample_counter = 0
#         for batch in data_loader:
#             x, y, z = batch
#             z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
#             z_hat = z_hat.view(z_hat.shape[0], -1)
# 
#             z_list.append(z)
#             z_hat_list.append(z_hat)
#             sample_counter += obs.shape[0]
#             if sample_counter >= num_samples:
#                 break
# 
#         z = torch.cat(z_list, 0)[:int(num_samples)]
#         z_hat = torch.cat(z_hat_list, 0)[:int(num_samples)]
# 
#         z, z_hat = z.cpu().numpy(), z_hat.cpu().numpy()
# 
#         score = get_linear_score(z_hat, z)
# 
#         # masking z_hat
#         # TODO: this does not take into account case where z_block_size > 1
#         if indices is not None:
#             z_hat_m = z_hat[:, indices[-z.shape[0]:]]
#             score_m = get_linear_score(z_hat_m, z)
#         else:
#             score_m = 0
# 
#         return score, score_m


def mean_corr_coef_np(x: np.ndarray, y: np.ndarray, method: str = 'pearson',
                      indices: list = None) -> float:
    """
    Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py
    A numpy implementation of the mean correlation coefficient metric.
    Args:
        x: numpy.ndarray
        y: numpy.ndarray
        method: The method used to compute the correlation coefficients ['pearson', 'spearman']
        indices: list of indices to consider, if None consider all variables
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))

    cc = np.abs(cc)
    if indices is not None:
        cc_program = cc[:, indices[-d:]]
    else:
        cc_program = cc

    assignments = linear_sum_assignment(-1 * cc_program)
    score = cc_program[assignments].mean()

    perm_mat = np.zeros((d, d))
    perm_mat[assignments] = 1
    # cc_program_perm = np.matmul(perm_mat.transpose(), cc_program)
    cc_program_perm = np.matmul(cc_program, perm_mat.transpose())  # permute the learned latents

    return score, cc_program_perm, assignments


def mean_corr_coef(model, data_loader, num_samples=int(1e5), method='pearson', indices=None):
    """Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py"""
    with torch.no_grad():
        model.eval()
        z_list = []
        z_hat_list = []
        sample_counter = 0

        # if num_samples is greater than number of examples in dataset
        n = data_loader.x.shape[0]
        if sample_counter < n:
            num_samples = n

        # Load data
        while sample_counter < num_samples:
            x, y, z = data_loader.sample_train(64)
            z_hat, _, _ = model.encode(x, y)

            z_list.append(z)
            z_hat_list.append(z_hat)

            sample_counter += x.shape[0]

        z = torch.cat(z_list, 0)[:int(num_samples)]
        z_hat = torch.cat(z_hat_list, 0)[:int(num_samples)]
        z, z_hat = z.cpu().numpy(), z_hat.cpu().numpy()

        z = z.reshape(z.shape[0] * z.shape[1], z.shape[2] * z.shape[3])
        z_hat = z_hat.reshape(z_hat.shape[0] * z_hat.shape[1], z_hat.shape[2] * z_hat.shape[3])

        score, cc_program_perm, assignments = mean_corr_coef_np(z, z_hat, method, indices)
        return score, cc_program_perm, assignments, z, z_hat


def edge_errors(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Counts all types of sensitivity/specificity metrics (true positive (tp),
    true negative (tn), false negatives (fn), false positives (fp), reversed edges (rev))

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
    Returns:
        tp, tn, fp, fn, fp_rev, fn_rev, rev
    """
    tp = ((pred == 1) & (pred == target)).sum()
    tn = ((pred == 0) & (pred == target)).sum()

    # errors
    diff = target - pred
    diff_t = np.swapaxes(diff, 1, 2)
    rev = (((diff + diff_t) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum()
    fp = (diff == -1).sum()
    fn_rev = fn - rev
    fp_rev = fp - rev

    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
            "fp_rev": float(fp_rev), "fn_rev": float(fn_rev), "rev": float(rev)}


def shd(pred: np.ndarray, target: np.ndarray, rev_as_double: bool = False) -> float:
    """
    Calculates the Structural Hamming Distance (SHD)

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
        rev_as_double: if True, reversed edges count for 2 mistakes
    Returns: shd
    """
    if rev_as_double:
        m = edge_errors(pred, target)
        m["rev"] = 0
    else:
        m = edge_errors(pred, target)
    shd = sum([m["fp"], m["fn"], m["rev"]])
    return float(shd)


def f1_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates the F1 score, ie the harmonic mean of
    the precision and recall.

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
    Returns: f1_score
    """
    m = edge_errors(pred, target)
    f1_score = m["tp"] / (m["tp"] + 0.5 * (m["fp"] + m["fn"]))
    return float(f1_score)


if __name__ == "__main__":
    from sklearn.metrics import f1_score as sk_f1_score
    from scipy.spatial.distance import hamming

    pred = np.asarray([[0, 1, 0],
                       [1, 0, 1],
                       [0, 0, 1]])
    good_pred = np.asarray([[1, 1, 0],
                            [0, 0, 0],
                            [1, 1, 0]])
    true = np.asarray([[1, 1, 0],
                       [0, 0, 0],
                       [1, 1, 1]])

    print(f"F1 score: {f1_score(pred, true)}")
    print(f"F1 score: {f1_score(good_pred, true)}")
    print(f"F1 score: {f1_score(true, true)}")
    print("----------")
    average_type = 'weighted'
    print(f"F1 score (sklearn): {sk_f1_score(pred.flatten(), true.flatten(), average=average_type)}")
    print(f"F1 score (sklearn): {sk_f1_score(good_pred.flatten(), true.flatten(), average=average_type)}")
    print(f"F1 score (sklearn): {sk_f1_score(true.flatten(), true.flatten(), average=average_type)}")
    print("==============================")
    print(f"SHD: {shd(pred, true)}")
    print(f"SHD: {shd(good_pred, true)}")
    print(f"SHD: {shd(true, true)}")
    print("----------")
    print(f"SHD (sklearn): {9 * hamming(pred.flatten(), true.flatten())}")
    print(f"SHD (sklearn): {9 * hamming(good_pred.flatten(), true.flatten())}")
    print(f"SHD (sklearn): {9 * hamming(true.flatten(), true.flatten())}")
