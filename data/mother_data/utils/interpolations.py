# interpolation classes
from abc import ABC, abstractmethod
from pathlib import Path

# see data/mother_data/utils/aggregations.py
# same thing, just other interpolation methods

### ABSTRACT CLASS #############################################################
class Interpolation(ABC):
    def __init__(self, var_path: Path, store_path: Path):
        """Init method that is the same for all subclasses
        Parameters:
            var_path (pathlib.Path)
            store_path (pathlib.Path)
        """
        # TODO
        # self.var_path = var_path
        # self.store_path = store_path
        # self.new_res_var= None
        raise NotImplementedError

    # every subclass must implement the abstract methods
    @abstractmethod
    def aggregate(self):
        """Some description what the function must be able to do
        # TODO
        """
        pass

    # not abstract method because all subclasses can use that one
    def __store__(self):
        """Stores the interpolated data in the right place"""
        # TODO store results, use self.store_path
        raise NotImplementedError


### SPECIFIC CLASSES ###########################################################

# DISCUSS:
# linear
# Nearest-neighbor
# Polynomial
# Splines
# Inverse Distance weighted [new for me]
# Kriging [new for me]
