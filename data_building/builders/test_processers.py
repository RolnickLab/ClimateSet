# just a file to test and run processing stuff

# TODO some directories that can be used to test the single functions

# functions that check if this was done right
import unittest
from pathlib import Path

from data_building.parameters.data_paths import TEST_CASES, ROOT
from data_building.utils.helper_funcs import get_single_example
from data_building.builders.preprocesser import SpatResProcesser, TempResProcesser

example_file = TEST_CASES / "test_subdir" / "another_subdir" / "another_file.nc"
example_file_long = TEST_CASES / "test_subdir" / "another_subdir" / "CMIP6_NorESM2-LM_r1i1p1f1_ssp126_tas_250_km_mon_gn_2015.nc"
example_dir = TEST_CASES / "test_subdir"

class TestUtils(unittest.TestCase):

    def test_get_single_example(self):
        dir = Path(TEST_CASES / "test_subdir" / "another_subdir")
        ground_truth = Path(dir / "another_file.nc")
        self.assertEqual(get_single_example(dir), ground_truth)

class TestResProcesser(unittest.TestCase):

    def test_init(self):
        res_processer = SpatResProcesser(example_file, "interpolate")
        self.assertEqual(res_processer.example, example_file)
        self.assertEqual(res_processer.task, "interpolate")
        self.assertEqual(res_processer.input_dir, None)
        self.assertEqual(res_processer.output_dir, None)
        self.assertEqual(res_processer.finished_preprocessing, False)

    def test_choose_alg(self):
        res_processer = SpatResProcesser(example_file, "interpolate")
        json_res_file = ROOT / "data_building" / "parameters" / "config_res_processing.json"

        # more can be added if desired
        # TODO test aggregation ones as well
        alg = res_processer.choose_alg(
            task="interpolate",
            climate_var="tas",
            json_path=json_res_file
        )
        self.assertEqual(alg, "remapbil")

        alg = res_processer.choose_alg(
            task="interpolate",
            climate_var="pr",
            json_path=json_res_file
        )
        self.assertEqual(alg, "remapcon")

        try:
            alg = res_processer.choose_alg(
                task="interpolate",
                climate_var="BC_em_biomassburning",
                json_path=json_res_file
            )
        except ValueError:
            alg = "null"
        self.assertEqual(alg, "null")


    def test_read_var(self):
        res_processer = SpatResProcesser(example=example_file, task="interpolate")
        var = res_processer.read_var(TEST_CASES / "full_names", test=False)
        self.assertEqual(var, "tas")
        var = res_processer.read_var(TEST_CASES / "full_names", test=True)
        self.assertEqual(var, "tas")

if __name__ == "__main__":
    unittest.main()
