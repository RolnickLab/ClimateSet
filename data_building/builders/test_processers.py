# just a file to test and run processing stuff

# TODO some directories that can be used to test the single functions

# functions that check if this was done right
import unittest
import xarray as xr
from pathlib import Path

from data_building.parameters.data_paths import TEST_CASES, ROOT
from data_building.utils.helper_funcs import get_single_example, read_gridfile
from data_building.builders.preprocesser import SpatResProcesser, TempResProcesser

example_file = TEST_CASES / "test_subdir" / "another_subdir" / "another_file.nc"
example_file_long = TEST_CASES / "test_subdir" / "another_subdir" / "CMIP6_NorESM2-LM_r1i1p1f1_ssp126_tas_250_km_mon_gn_2015.nc"
example_dir = TEST_CASES / "test_subdir"
flagged_file = TEST_CASES / "test_subdir" / "flagged_cmip6_file.nc"
example_cmip6_dir = TEST_CASES / "example_cmip6" / "CMIP6"
gridfile_125km = ROOT / "tmp" / "grid_files" / "125km_grid.txt"
output_dir = TEST_CASES / "output"
low_res_file = TEST_CASES / "test_res" / "spat" / "500km.nc"
med_res_file = TEST_CASES / "test_res" / "spat" / "250km.nc"
high_res_file = TEST_CASES / "test_res" / "spat" / "100km.nc"

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

    # only relevant for spatial resolution processer
    def test_create_grid_from_example(self):
        res_processer = SpatResProcesser(example=example_file)
        grid_attrs = read_gridfile(res_processer.grid_file)

        compare_dict = {
            'gridtype': 'lonlat',
            'xsize': '144', 'ysize': '96',
            'xfirst': '0.0', 'yfirst': '-90.0',
            'xinc': '2.5', 'yinc': '1.8947368421052602'
        }

        self.assertEqual(grid_attrs, compare_dict)

    def test_choose_alg(self):
        res_processer = SpatResProcesser(example_file, "remap")
        json_res_file = ROOT / "data_building" / "parameters" / "config_res_processing.json"

        # more can be added if desired
        # TODO test aggregation ones as well
        alg = res_processer.choose_alg(
            task="remap",
            climate_var="tas",
            json_path=json_res_file
        )
        self.assertEqual(alg, "remapbil")

        alg = res_processer.choose_alg(
            task="remap",
            climate_var="pr",
            json_path=json_res_file
        )
        self.assertEqual(alg, "remapcon")

        try:
            alg = res_processer.choose_alg(
                task="remap",
                climate_var="BC_em_biomassburning",
                json_path=json_res_file
            )
        except ValueError:
            alg = "null"
        self.assertEqual(alg, "null")


    def test_read_var(self):
        res_processer = SpatResProcesser(example=example_file)
        var = res_processer.read_var(TEST_CASES / "full_names", test=False)
        self.assertEqual(var, "tas")
        var = res_processer.read_var(TEST_CASES / "full_names", test=True)
        self.assertEqual(var, "tas")

    def test_raw_processed(self):
        res_processer = SpatResProcesser(example=example_file)
        is_processed = res_processer.check_raw_processed(flagged_file)
        self.assertEqual(is_processed, False)

    def test_create_weights(self):
        # we wanna get low res files and feed in some high res files
        res_processer = SpatResProcesser(example=low_res_file)
        weights_path = res_processer.create_weights(alg="remapbil", input_file=high_res_file)
        weights = xr.open_dataset(weights_path)
        self.assertEqual(weights.attrs["source_grid"], "lonlat")
        self.assertEqual(weights.attrs["dest_grid"], "lonlat")
        self.assertEqual(weights.attrs["map_method"], "Bilinear remapping")
        self.assertEqual(weights["src_grid_dims"][0].item(), 360)
        self.assertEqual(weights["src_grid_dims"][1].item(), 180)
        self.assertEqual(weights["dst_grid_dims"][0].item(), 72)
        self.assertEqual(weights["dst_grid_dims"][1].item(), 72)
    #
    # # def test_applysubdir(self):
    # #     res_processer = SpatResProcesser(example=None, task="interpolate", grid_file=gridfile_125km)
    # #     res_processer.interpolate(alg="remapbil", input=example_cmip6_dir, output=output_dir)
    # #     exit(0)
    # #     pass

    # different tests:
    # both with and without example file
    # apply_subdir and remap directly
    # cmip6 data
    # input4mips
    # input4mips historical (openburning historical e.g.)
    def test_applysubdir(self):
        res_processer = SpatResProcesser(example=None, grid_file=gridfile_125km)
        res_processer.apply_subdir(sub_dir=example_cmip6_dir, output_dir=output_dir, threads=1)
        # TODO asserts

    def test_remap(self):
        pass




if __name__ == "__main__":
    unittest.main()
