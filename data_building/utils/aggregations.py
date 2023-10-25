# aggregation classes
from pathlib import Path
from abc import ABC, abstractmethod


### ABSTRACT CLASS #############################################################
class Aggregation(ABC):
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
        """Stores the aggregated data in the right place"""
        # TODO store results, use self.store_path
        raise NotImplementedError


### SPECIFIC CLASSES ###########################################################
# must implement the abstract methods!!!


# Note: all those aggregation methods were mentioned as the usual
# CMIP6 aggregation methods
class MeanAggregation(Aggregation):
    """Calculates mean over aggregation period"""

    def aggregate(self):
        # TODO: some mean calculations on self.var_path,
        # results go into self.new_res_var
        # TODO store results with self.__store__()
        raise NotImplementedError


class MaxAggregation(Aggregation):
    """Takes maximum over aggregation period"""

    def aggregate(self):
        raise NotImplementedError


class MinAggregation(Aggregation):
    """Takes minimum over aggregation period"""

    def aggregate(self):
        raise NotImplementedError


class InstAggregation(Aggregation):
    """Takes instantaneous value within aggregation period
    # TODO: not sure what it means exactly
    """

    def aggregate(self):
        raise NotImplementedError
