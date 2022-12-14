import numpy as np
import glob
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from typing import Tuple


def mean_corr_coef(x: np.ndarray, y: np.ndarray, method: str = 'pearson',
                   indices: list = None) -> float:
    """
    Source: https://github.com/ilkhem/icebeem/blob/master/metrics/mcc.py
    A numpy implementation of the mean correlation coefficient (MCC) metric.
    Args:
        x: numpy.ndarray
        y: numpy.ndarray
        method: The method used to compute the correlation coefficients
        ['pearson', 'spearman']
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


def assignment_l1(x, y):
    d = x.shape[1]
    dist = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            dist[i, j] = np.linalg.norm(x[i] - y[j], ord=1)

    assignments = linear_sum_assignment(dist)
    score = dist[assignments].mean()

    return score, assignments


def clustering_consistency(path: str):
    """
    Test how consistent the clusters at the grid-location level
    are consistent (find the best permutation since labels are arbitrary)
    """
    # find all the experiments in the given directory
    files = glob.glob(f"{path}/exp*/w_tensor.npy")
    print(files)

    ws = []
    n = len(files)
    score = np.zeros((n, n))

    # loop over all tensors W
    for file in files:
        ws.append(np.load(file)[0])

    for i in range(n):
        for j in range(i+1, n):
            # find the best cluster alignment for each pair
            score[i, j], assignments = assignment_l1(ws[i], ws[j])
            score[j, i] = score[i, j]
    print(np.mean(score) / 2)

    return np.mean(score) / 2


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
    diff_t = np.swapaxes(diff, -2, -1)
    rev = (((diff + diff_t) == 0) & (diff != 0)).sum() // 2
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
        shd = sum([m["fp"], m["fn"]])
    else:
        m = edge_errors(pred, target)
        shd = sum([m["fp_rev"], m["fn_rev"], m["rev"]])
    return float(shd)

def precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    tp = ((pred == 1) & (pred == target)).sum()
    # tn = ((pred == 0) & (pred == target)).sum()
    diff = target - pred
    fn = (diff == 1).sum()
    fp = (diff == -1).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


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
    pass
    # clustering_consistency("./test")

    # simple tests
    # from sklearn.metrics import f1_score as sk_f1_score
    # from scipy.spatial.distance import hamming

    # pred = np.asarray([[0, 1, 0],
    #                    [0, 0, 1],
    #                    [0, 0, 1]])
    # good_pred = np.asarray([[1, 1, 0],
    #                         [0, 0, 0],
    #                         [1, 1, 0]])
    # true = np.asarray([[1, 1, 0],
    #                    [0, 0, 0],
    #                    [1, 1, 1]])

    # print(f"F1 score: {f1_score(pred, true)}")
    # print(f"F1 score: {f1_score(good_pred, true)}")
    # print(f"F1 score: {f1_score(true, true)}")
    # print("----------")
    # average_type = 'weighted'
    # print(f"F1 score (sklearn): {sk_f1_score(pred.flatten(), true.flatten(), average=average_type)}")
    # print(f"F1 score (sklearn): {sk_f1_score(good_pred.flatten(), true.flatten(), average=average_type)}")
    # print(f"F1 score (sklearn): {sk_f1_score(true.flatten(), true.flatten(), average=average_type)}")
    # print("==============================")
    # print(f"SHD: {shd(pred, true)}")
    # print(f"SHD: {shd(good_pred, true)}")
    # print(f"SHD: {shd(true, true)}")
    # print("----------")
    # print(f"SHD (sklearn): {9 * hamming(pred.flatten(), true.flatten())}")
    # print(f"SHD (sklearn): {9 * hamming(good_pred.flatten(), true.flatten())}")
    # print(f"SHD (sklearn): {9 * hamming(true.flatten(), true.flatten())}")
