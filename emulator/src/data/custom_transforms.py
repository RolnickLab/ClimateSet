import numpy as np
import torch


class Normalize_z:
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (tuple): means for each variable.
        std (tuple): standard deviations for each variable.
    """

    def __init__(self, stats_path):
        self.mean_in = np.expand_dims(mean_in, axis=(1, 1, 1))
        self.std_in = np.expand_dims(std_in, axis=(1, 1, 1))
        self.eps = 1e-9

    def __call__(self, X, y):
        X_norm = (X - self.mean_in) / (self.std_in + self.eps)
        y_norm = (y - self.mean_out) / (self.std_out + self.eps)

        return X_norm, y_norm


class Normalize_minmax:
    """Normalize a tensor image with mean and standard deviation.

    Args:
        mean (tuple): means for each variable.
        std (tuple): standard deviations for each variable.
    """

    def __init__(self, min_pr=0.0, max_pr=0.0, min_t=0.0, max_t=0.0):
        self.min_pr = min_pr
        self.max_pr = max_pr
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, X, y):
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # Image norm
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        mask = sample["label"]

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.uint8)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {"image": img, "label": mask}
