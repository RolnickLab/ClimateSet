# just a file to test and run processing stuff

# TODO some directories that can be used to test the single functions

# functions that check if this was done right

import unittest
from pathlib import Path

from data_building.parameters.data_paths import TEST_CASES
from data_building.utils.helper_funcs import get_single_example

class TestUtils(unittest.TestCase):

    def test_get_single_example(self):
        dir = Path(TEST_CASES / "test_subdir" / "another_subdir")
        ground_truth = Path(dir / "another_file.nc")
        self.assertEqual(get_single_example(dir), ground_truth)
        
if __name__ == "__main__":
    unittest.main()
