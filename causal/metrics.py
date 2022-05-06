import numpy as np


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
    total_edges = (target).sum()
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
