from typing import List, Sequence, Union, Dict
import torch
import numpy as np


class PredictionPostProcessCallback:
    # PostProcessing Outputs
    def __init__(self,
                 variables: List[str],
                 sizes: Union[int, Sequence[int]] # if multiple features belong to a variable e.g. stretching over multiple leevls
                 ):
        self.variable_to_channel = dict()
        cur = 0
        sizes = [sizes for _ in range(len(variables))] if isinstance(sizes, int) else sizes
        for var, size in zip(variables, sizes):
            self.variable_to_channel[var] = {'start': cur, 'end': cur + size}
            cur += size

    def split_vector_by_variable(self,
                                 vector: Union[np.ndarray, torch.Tensor]
                                 ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if isinstance(vector, dict):
            return vector
        splitted_vector = dict()
        for var_name, var_channel_limits in self.variable_to_channel.items():
            splitted_vector[var_name] = vector[..., var_channel_limits['start']:var_channel_limits['end']]
        return splitted_vector

    def __call__(self, vector, *args, **kwargs):
        return self.split_vector_by_variable(vector)
