import numpy as np


def edge_errors(pred: np.ndarray, target: np.ndarray):
    """
    Counts all types of sensitivity/specificity metrics (true positive (tp),
    true negative (tn), false negatives (fn), false positives (fp), reversed edges (rev))

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
    Returns:
        tp, tn, fp, fn, fp_rev, fn_rev, rev
    """
    total_edges = (target).sum()
    tp = ((pred == 1) & (pred == target)).sum()
    tn = ((pred == 0) & (pred == target)).sum()

    # errors
    diff = target - pred
    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum()
    fp = (diff == -1).sum()
    fn_rev = fn - rev
    fp_rev = fp - rev

    return tp, tn, fp, fn, fp_rev, fn_rev, rev


def shd(pred: np.ndarray, target: np.ndarray, rev_as_double: bool = False):
    """
    Calculates the Structural Hamming Distance (SHD)

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
        rev_as_double: if True, reversed edges count for 2 mistakes
    Returns: shd
    """
    if rev_as_double:
        _, _, fp, fn, _, _, _ = edge_errors(pred, target)
        rev = 0
    else:
        _, _, _, _, fp, fn, rev = edge_errors(pred, target)
    return sum([fp, fn, rev])


def f1_score(pred: np.ndarray, target: np.ndarray):
    """
    Calculates the F1 score, ie the harmonic mean of
    the precision and recall.

    Args:
        pred: The predicted adjacency matrix
        target: The true adjacency matrix
    Returns: f1_score
    """
    tp, _, fp, fn, _, _, _ = edge_errors(pred, target)
    f1_score = tp / (tp + 0.5 * (fp + fn))
    return f1_score


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
