# Copied from climart
# Needs to be adjusted TODO


import logging
from abc import ABC
from typing import Optional, Union, Dict, Iterable, Sequence, List, Callable
import numpy as np
import torch
from torch import Tensor, nn

NP_ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]


class NormalizationMethod(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def normalize(self, data: np.ndarray, axis=0, *args, **kwargs):
        return data

    def inverse_normalize(self, normalized_data: np.ndarray):
        return normalized_data

    def stored_values(self):
        return dict()

    def __copy__(self):
        return type(self)(**self.stored_values())

    def copy(self):
        return self.__copy__()

    def change_input_type(self, new_type):
        for attribute, value in self.__dict__.items():
            if new_type in [torch.Tensor, torch.TensorType]:
                if isinstance(value, np.ndarray):
                    setattr(self, attribute, torch.from_numpy(value).float())
            elif new_type == np.ndarray:
                if torch.is_tensor(value):
                    setattr(self, attribute, value.numpy().cpu())
            else:
                setattr(self, attribute, new_type(value))

    def apply_torch_func(self, fn):
        """
        Function to be called to apply a torch function to all tensors of this class, e.g. apply .to(), .cuda(), ...,
        Just call this function from within the model's nn.Module._apply()
        """
        for attribute, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, attribute, fn(value))


class Z_Normalizer(NormalizationMethod):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def normalize(self, data, axis=None, *args, **kwargs):
        return self(data)

    def inverse_normalize(self, normalized_data):
        data = normalized_data * self.std + self.mean
        return data

    def stored_values(self):
        return {'mean': self.mean, 'std': self.std}

    def __call__(self, data):
        return (data - self.mean) / self.std


class MinMax_Normalizer(NormalizationMethod):
    def __init__(self, min=None, max_minus_min=None, max=None, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        if min:
            assert max_minus_min or max
            self.max_minus_min = max_minus_min or max - min

    def normalize(self, data, axis=None, *args, **kwargs):
        # self.min = np.min(data, axis=axis)
        # self.max_minus_min = (np.max(data, axis=axis) - self.min)
        return self(data)

    def inverse_normalize(self, normalized_data):
        shapes = normalized_data.shape
        if len(shapes) >= 2:
            normalized_data = normalized_data.reshape(normalized_data.shape[0], -1)
        data = normalized_data * self.max_minus_min + self.min
        if len(shapes) >= 2:
            data = data.reshape(shapes)
        return data

    def stored_values(self):
        return {'min': self.min, 'max_minus_min': self.max_minus_min}

    def __call__(self, data):
        return (data - self.min) / self.max_minus_min


class LogNormalizer(NormalizationMethod):
    def normalize(self, data, *args, **kwargs):
        normalized_data = self(data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = np.exp(normalized_data)
        return data

    def __call__(self, data: np.ndarray, *args, **kwargs):
        return np.log(data)


class LogZ_Normalizer(NormalizationMethod):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.z_normalizer = Z_Normalizer(mean, std)

    def normalize(self, data, *args, **kwargs):
        normalized_data = np.log(data + 1e-5)
        normalized_data = self.z_normalizer.normalize(normalized_data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = self.z_normalizer.inverse_normalize(normalized_data)
        data = np.exp(data) - 1e-5
        return data

    def stored_values(self):
        return self.z_normalizer.stored_values()

    def change_input_type(self, new_type):
        self.z_normalizer.change_input_type(new_type)

    def apply_torch_func(self, fn):
        self.z_normalizer.apply_torch_func(fn)

    def __call__(self, data, *args, **kwargs):
        normalized_data = np.log(data + 1e-5)
        return self.z_normalizer(normalized_data)


class MinMax_LogNormalizer(NormalizationMethod):
    def __init__(self, min=None, max_minus_min=None, **kwargs):
        super().__init__(**kwargs)
        self.min_max_normalizer = MinMax_Normalizer(min, max_minus_min)

    def normalize(self, data, *args, **kwargs):
        normalized_data = self.min_max_normalizer.normalize(data)
        normalized_data = np.log(normalized_data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = np.exp(normalized_data)
        data = self.min_max_normalizer.inverse_normalize(data)
        return data

    def stored_values(self):
        return self.min_max_normalizer.stored_values()

    def change_input_type(self, new_type):
        self.min_max_normalizer.change_input_type(new_type)

    def apply_torch_func(self, fn):
        self.min_max_normalizer.apply_torch_func(fn)


def get_normalizer(normalizer='z', *args, **kwargs) -> NormalizationMethod:
    normalizer = normalizer.lower().strip().replace('-', '_').replace('&', '+')
    supported_normalizers = ['z',
                             'min_max',
                             'min_max+log', 'min_max_log',
                             'log_z',
                             'log',
                             'none']
    assert normalizer in supported_normalizers, f"Unsupported Normalization {normalizer} not in {str(supported_normalizers)}"
    if normalizer == 'z':
        return Z_Normalizer(*args, **kwargs)
    elif normalizer == 'min_max':
        return MinMax_Normalizer(*args, **kwargs)
    elif normalizer in ['min_max+log', 'min_max_log']:
        return MinMax_LogNormalizer(*args, **kwargs)
    elif normalizer in ['logz', 'log_z']:
        return LogZ_Normalizer(*args, **kwargs)
    elif normalizer == 'log':
        return LogNormalizer(*args, **kwargs)
    else:
        return NormalizationMethod(*args, **kwargs)  # like no normalizer


class Normalizer:
    def __init__(
            self,
            #datamodule_config: DictConfig,
            input_normalization: Optional[str] = None,
            output_normalization: Optional[str] = None,
            spatial_normalization_in: bool = False,
            spatial_normalization_out: bool = False,
            log_scaling: Union[bool, List[str]] = False,
            data_dir: Optional[str] = None,
            verbose: bool = True
    ):
        """
        input_normalization (str): "z" for z-scaling (zero mean and unit standard deviation)
        """
        
        print(f"INFO: Initializing Normalizer.")
        print(f"WARNING: Normalizer not yet implemented")

        self._feature_by_var = None
        self.input_normalization = input_normalization
        self.output_normalization=output_normalization
        self.spatial_normalization_in=spatial_normalization_in
        self.spatial_normalization_out=spatial_normalization_out
        self.log_scaling=log_scaling
        self.data_dir=data_dir
        self.verbose=verbose

    @property
    def feature_by_var(self):
        return self._feature_by_var

    def get_input_normalizer(self, data_type: str) -> Union[NP_ARRAY_MAPPING, NormalizationMethod]:
        return self._input_normalizer[data_type]

    def get_input_normalizers(self) -> Dict[str, Union[NP_ARRAY_MAPPING, NormalizationMethod]]:
        return {
            data_type: self.get_input_normalizer(data_type)
            for data_type in constants.INPUT_TYPES
        }

    def set_normalizer(self, data_type: str, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        if new_normalizer is None:
            new_normalizer = identity
        if data_type in constants.INPUT_TYPES:
            self._input_normalizer[data_type] = new_normalizer
        else:
            print(f"INFO: Setting output normalizer, after calling set_normalizer with data_type={data_type}")
            self._output_normalizer = new_normalizer

    def set_input_normalizers(self, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        for data_type in constants.INPUT_TYPES:
            self.set_normalizer(data_type, new_normalizer)

    @property
    def output_normalizer(self):
        return self._output_normalizer

    def normalize(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for input_type, rawX in X.items():
            X[input_type] = self._input_normalizer[input_type].normalize(rawX)

        X = self._log_scaler_func(X)
        return X

    def __call__(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.normalize(X)