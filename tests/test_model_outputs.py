"""Small numerical checks; run with the module environment's Python and unittest."""

# The isolated module environment has unittest, but deliberately excludes pytest.
# ruff: noqa: PT009, PT027

import copy
import importlib.util
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    import rasterio
    import xarray as xr
    import yaml
    from rasterio.transform import from_origin

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

ROOT = Path(__file__).resolve().parents[1]


def run_calculation(config, cycle, folder):
    """Run a tiny cube through the actual calculation script."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temperature = folder / "temperature.nc"
    sediment = folder / "sediment.nc"
    output = folder / "cube.nc"
    coords = {"depth_m": [1000, 3000, 6000], "y": [15000, 5000], "x": [5000, 15000]}
    xr.Dataset(
        {
            "temperature": (
                ("depth_m", "y", "x"),
                np.array(
                    [
                        [[180, 200], [220, np.nan]],
                        [[200, 220], [240, 260]],
                        [[220, 240], [260, 280]],
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords=coords,
        attrs={"crs": "EPSG:3857"},
    ).to_netcdf(temperature)
    xr.Dataset(
        {"sediment_thickness_m": (("y", "x"), [[2000, np.nan], [5000, 1000]])},
        coords={"y": coords["y"], "x": coords["x"]},
        attrs={"crs": "EPSG:3857"},
    ).to_netcdf(sediment)
    inputs = SimpleNamespace(
        nc=str(temperature),
        sed=str(sediment),
        eff=str(ROOT / "workflow/internal/efficiency_curves.csv"),
        drilling_costs_2017=str(ROOT / "workflow/internal/baseline_drilling_2017.csv"),
        drilling_costs_2025=str(ROOT / "workflow/internal/baseline_drilling_2025.csv"),
    )

    class Params(dict):
        """Match Snakemake's attribute and mapping access."""

        __getattr__ = dict.__getitem__

    job = SimpleNamespace(
        config=config,
        input=inputs,
        params=Params(cycle=cycle, model=config),
        output=SimpleNamespace(cube=str(output)),
        log=[str(folder / "calculation.log")],
    )
    stdout, stderr = sys.stdout, sys.stderr
    try:
        runpy.run_path(
            str(ROOT / "workflow/scripts/calculate_LCOE.py"),
            init_globals={"snakemake": job},
            run_name="__main__",
        )
    finally:
        if sys.stdout is not stdout:
            sys.stdout.close()
        sys.stdout, sys.stderr = stdout, stderr
    with xr.open_dataset(output) as result:
        return result.load()


@unittest.skipUnless(HAS_RUNTIME, "Requires the module runtime environment")
class ModelOutputsTest(unittest.TestCase):
    """Verify direct config use and study-wide raster summaries."""

    def setUp(self):
        """Create isolated synthetic inputs."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
        spec = importlib.util.spec_from_file_location(
            "overall_potential", ROOT / "workflow/scripts/overall_potential.py"
        )
        self.summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.summary)

    def test_model_obeys_nested_temperatures_and_egs_settings(self):
        """Apply nested temperatures, depth limits, and sediment policies."""
        for cycle, name in [("orc", "ORC"), ("flash", "Flash"), ("kalina", "Kalina")]:
            with self.subTest(cycle=cycle):
                base = run_calculation(
                    self.config, cycle, self.folder / f"{cycle}_base"
                )
                self.assertTrue(np.isfinite(base.lcoe).any())
                changed = copy.deepcopy(self.config)
                changed["temp_turbine_out"][name] += 10
                other = run_calculation(changed, cycle, self.folder / f"{cycle}_temp")
                self.assertFalse(
                    np.allclose(base.potential, other.potential, equal_nan=True)
                )
        no_egs = copy.deepcopy(self.config)
        no_egs["include_egs"] = False
        result = run_calculation(no_egs, "orc", self.folder / "no_egs")
        self.assertEqual(float(result.depth_m.max()), 3000)
        self.assertEqual(result.attrs["include_egs"], 0)
        depth_only = copy.deepcopy(self.config)
        depth_only["SED_NAN_POLICY"] = "depth_only"
        result = run_calculation(depth_only, "orc", self.folder / "depth_only")
        self.assertEqual(
            float(result.egs_cutoff_depth_m.isel(depth_m=0, y=0, x=1)), 3000
        )

    def write_raster(self, name, values, offset=0):
        """Write a tiny raster with a known grid."""
        path = self.folder / "documents" / name
        path.parent.mkdir(exist_ok=True)
        values = np.array(values, dtype="float64")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=values.dtype,
            crs="EPSG:3857",
            transform=from_origin(offset, 2, 1, 1),
            nodata=np.nan,
        ) as dst:
            dst.write(values, 1)
        return path

    def test_totals_and_least_cost_portfolio(self):
        """Sum depth bands without summing competing technologies in a cell."""
        for categories in [[None], ["0-3km", "3-6km"]]:
            with self.subTest(categories=categories):
                lcoe, potential = [], []
                for category in categories:
                    suffix = f"{category}_" if category else ""
                    for tech, costs, capacities in [
                        (
                            "orc",
                            [[10, 30, np.nan, np.nan]],
                            [[1e6, 2e6, np.nan, np.nan]],
                        ),
                        ("flash", [[20, 20, 40, np.nan]], [[3e6, 4e6, 5e6, np.nan]]),
                    ]:
                        lcoe.append(
                            self.write_raster(f"{tech}_2d_{suffix}lcoe.tif", costs)
                        )
                        potential.append(
                            self.write_raster(
                                f"{tech}_2d_{suffix}potential.tif", capacities
                            )
                        )
                result = self.summary.potential_summary_table(lcoe, potential)
                self.assertEqual(list(result.columns), ["technology", "potential_tw"])
                totals = result.set_index("technology")["potential_tw"].to_dict()
                self.assertEqual(
                    totals,
                    {
                        "orc": 3 * len(categories),
                        "flash": 12 * len(categories),
                        "lowest_lcoe_technology": 10 * len(categories),
                    },
                )

    def test_missing_pairs_and_mismatched_grids(self):
        """Reject incomplete or misaligned raster inputs."""
        lcoe = self.write_raster("orc_2d_lcoe.tif", [[10]])
        with self.assertRaisesRegex(ValueError, "Missing matching"):
            self.summary.potential_summary_table([lcoe], [])
        potential = self.write_raster("orc_2d_potential.tif", [[1e6]], offset=1)
        with self.assertRaisesRegex(ValueError, "grids differ"):
            self.summary.potential_summary_table([lcoe], [potential])
        with self.assertRaisesRegex(ValueError, "Duplicate raster"):
            self.summary.potential_summary_table([lcoe, lcoe], [potential])


if __name__ == "__main__":
    unittest.main()
