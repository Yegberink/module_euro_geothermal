"""Numerical and geospatial heat checks in the module runtime environment."""

# The runtime contains unittest rather than pytest.
# ruff: noqa: PT009, PT027

import copy
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow/scripts"))

try:
    import aggregate_to_regions as regional
    import allocate_heat as allocation
    import calculate_LCOH as heat
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import xarray as xr
    import yaml
    from _heat import DEMAND_COLUMN, align_to_grid, load_networks, read_raster
    from affine import Affine
    from shapely.geometry import box
except ImportError:
    HAS_RUNTIME = False
else:
    HAS_RUNTIME = True


class AttrDict(dict):
    """Provide the two access styles of Snakemake named inputs and parameters."""

    __getattr__ = dict.__getitem__


def run_script(name, inputs, outputs, params, folder):
    """Execute a real workflow script with isolated files and restore logging."""
    job = AttrDict(
        input=AttrDict(inputs),
        output=AttrDict(outputs),
        params=AttrDict(params),
        log=[str(folder / f"{name}.log")],
    )
    stdout, stderr = sys.stdout, sys.stderr
    try:
        runpy.run_path(
            str(ROOT / "workflow/scripts" / f"{name}.py"),
            init_globals={"snakemake": job},
            run_name="__main__",
        )
    finally:
        if sys.stdout is not stdout:
            sys.stdout.close()
        sys.stdout, sys.stderr = stdout, stderr


@unittest.skipUnless(HAS_RUNTIME, "Requires the module runtime environment")
class HeatOutputsTest(unittest.TestCase):
    """Check physics, joint allocation, regional alternatives and script outputs."""

    def setUp(self):
        """Create a workspace and the published model settings."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = yaml.safe_load((ROOT / "config/config.yaml").read_text())

    def raster(self, values, name=None):
        """Create a projected 10 km test grid with optional raster output."""
        values = np.asarray(values, dtype=float)
        rows, cols = values.shape
        data = xr.DataArray(
            values,
            dims=("y", "x"),
            coords={
                "x": 4005000 + np.arange(cols) * 10000,
                "y": 3005000 - np.arange(rows) * 10000,
            },
        )
        data = data.rio.write_crs("EPSG:3035").rio.write_transform(
            Affine(10000, 0, 4000000, 0, -10000, 3010000)
        )
        if name:
            data.rio.write_nodata(np.nan).rio.to_raster(self.folder / name)
        return data

    def test_temperature_crossings_and_gaps(self):
        """Interpolate shallowest adjacent crossings, honoring exact depth limits."""
        profiles = np.array(
            [
                [20, 60, 100, 120],
                [20, 80, 100, 120],
                [90, 100, 110, 120],
                [20, np.nan, 100, 120],
                [20, 40, 60, 70],
                [20, 100, 40, 100],
                [20, 40, 60, 100],
                [80, 90, 100, 110],
            ]
        ).T[:, None, :]
        result = heat.shallowest_depth(profiles, [0, 1000, 2000, 3000], 80, 2500)
        np.testing.assert_allclose(
            result,
            [[1500, 1000, np.nan, np.nan, np.nan, 750, 2500, np.nan]],
            equal_nan=True,
        )
        self.assertTrue(
            np.isnan(
                heat.shallowest_depth(
                    profiles[:, :, 6:7], [0, 1000, 2000, 3000], 80, 2499
                )
            ).all()
        )

    def test_extent_and_lcoh_scripts(self):
        """Produce real aligned rasters and reuse extents when only demand changes."""
        networks = gpd.GeoDataFrame(
            {"area_id": [1], DEMAND_COLUMN: [100.0]},
            geometry=[box(4001000, 3001000, 4009000, 3009000)],
            crs=3035,
        )
        shapes = networks.rename(columns={"area_id": "shape_id"}).assign(
            shape_id="region"
        )
        networks.to_parquet(self.folder / "networks.parquet")
        shapes.to_parquet(self.folder / "shapes.parquet")
        coords = {
            "x": [4005000, 4015000],
            "y": [2995000, 3005000],
            "depth_m": [0, 2000],
        }
        values = np.array(
            [[[20.0, 20.0], [20.0, 20.0]], [[100.0, 60.0], [100.0, np.nan]]]
        )
        xr.Dataset(
            {"temperature": (("depth_m", "y", "x"), values)},
            coords=coords,
            attrs={"crs": "EPSG:3035"},
        ).to_netcdf(self.folder / "temperature.nc")
        xr.Dataset(
            {"sediment_thickness_m": (("y", "x"), np.full((2, 2), 3000.0))},
            coords={k: v for k, v in coords.items() if k != "depth_m"},
            attrs={"crs": "EPSG:3035"},
        ).to_netcdf(self.folder / "sediment.nc")
        inputs = {
            "nc": str(self.folder / "temperature.nc"),
            "shapes": str(self.folder / "shapes.parquet"),
            "networks": str(self.folder / "networks.parquet"),
        }
        extent = self.folder / "extent.json"
        run_script(
            "prepare_heat_extent",
            inputs,
            {"extent": str(extent)},
            {"buffer_km": 10},
            self.folder,
        )
        self.assertEqual(json.loads(extent.read_text())["x"], [0, 2])
        original_mtime = extent.stat().st_mtime_ns
        networks[DEMAND_COLUMN] = 200.0
        networks.to_parquet(self.folder / "networks.parquet")
        run_script(
            "prepare_heat_extent",
            inputs,
            {"extent": str(extent)},
            {"buffer_km": 10},
            self.folder,
        )
        self.assertEqual(extent.stat().st_mtime_ns, original_mtime)
        heat_inputs = {
            "nc": inputs["nc"],
            "extent": str(extent),
            "sed": str(self.folder / "sediment.nc"),
            **{
                f"drilling_costs_{year}": str(
                    ROOT / f"workflow/internal/baseline_drilling_{year}.csv"
                )
                for year in [2017, 2025]
            },
        }
        outputs = {
            name: str(self.folder / f"{name}.tif")
            for name in ["lcoh", "potential", "depth"]
        }
        run_script(
            "calculate_LCOH", heat_inputs, outputs, {"model": self.model}, self.folder
        )
        depth = read_raster(outputs["depth"])
        np.testing.assert_allclose(
            depth.values, [[1500, np.nan], [1500, np.nan]], equal_nan=True
        )
        self.assertLess(depth.rio.transform().e, 0)
        for name in ["lcoh", "potential"]:
            self.assertEqual(
                read_raster(outputs[name]).rio.transform(), depth.rio.transform()
            )
        # A completely out-of-source extent produces NaN maps, not remote supply.
        extent.write_text(json.dumps({"x": [0, 2], "y": [0, 2], "empty": True}))
        run_script(
            "calculate_LCOH", heat_inputs, outputs, {"model": self.model}, self.folder
        )
        self.assertTrue(np.isnan(read_raster(outputs["potential"]).values).all())

    def test_input_validation_empty_networks_and_grid_alignment(self):
        """Reject invalid demands/IDs and preserve exact centres across cropped grids."""
        networks = gpd.GeoDataFrame(
            {"area_id": [1, 2], DEMAND_COLUMN: [1.0, np.nan]},
            geometry=[box(0, 0, 1, 1)] * 2,
            crs=3035,
        )
        path = self.folder / "networks.parquet"
        networks.to_parquet(path)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            load_networks(path)
        networks[DEMAND_COLUMN] = [-1.0, 1.0]
        networks.to_parquet(path)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            load_networks(path)
        networks[DEMAND_COLUMN] = [1.0, 1.0]
        networks["area_id"] = [1, 1]
        networks.to_parquet(path)
        with self.assertRaises(Exception):
            load_networks(path)
        networks.iloc[:0].to_parquet(path)
        empty = load_networks(path)
        regions = gpd.GeoDataFrame(
            {"shape_id": ["a"]}, geometry=[box(0, 0, 1, 1)], crs=3035
        )
        self.assertEqual(len(allocation.assign_network_regions(empty, regions)), 0)
        grid = self.raster([[0.2, 0.5], [0.0, 0.0]])
        self.assertEqual(
            allocation.candidate_edges(empty, grid, grid, 10).shape, (0, 2)
        )
        target = self.raster([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        aligned = align_to_grid(grid, target)
        np.testing.assert_allclose(aligned.values, [[0.2, 0.5, 0], [0, 0, 0]])
        shifted = target.assign_coords(x=target.x + 1)
        with self.assertRaisesRegex(ValueError, "origins differ"):
            align_to_grid(grid, shifted)

    def test_heat_economics_and_egs(self):
        """Compare output and additive costs with an independently computed doublet."""
        model = copy.deepcopy(self.model)
        model["well_spacing"] = np.sqrt(50)  # Exactly one doublet per cell.
        temperature = np.array([[[20.0]], [[100.0]]])
        result = heat.calculate_heat(temperature, [0, 2000], [[3000]], model)
        capacity = 100 * 4250 * 40 / 1e6
        energy = capacity * 8760 * 0.88
        drilling = 2 * (250000 + 700 * 1500 + 0.2 * 1500**2)
        investment = drilling + 600000 + capacity * 1000 * 234
        annuity = 0.075 / (1 - 1.075**-30)
        self.assertAlmostEqual(result["potential"].item(), energy)
        self.assertAlmostEqual(
            result["lcoh"].item(), investment * (annuity + 0.015) / energy
        )
        self.assertAlmostEqual(
            result["lcoh"].item(),
            sum(v.item() for k, v in result.items() if k.startswith("lcoh_")),
        )
        model["parasitic_fraction"] = 0.99
        self.assertEqual(
            result["potential"].item(),
            heat.calculate_heat(temperature, [0, 2000], [[3000]], model)[
                "potential"
            ].item(),
        )
        model["include_egs"] = False
        self.assertTrue(
            np.isnan(
                heat.calculate_heat(temperature, [0, 2000], [[1000]], model)["lcoh"]
            ).all()
        )
        self.assertTrue(
            np.isnan(
                heat.calculate_heat(temperature, [0, 2000], [[np.nan]], model)["lcoh"]
            ).all()
        )
        model["SED_NAN_POLICY"] = "depth_only"
        self.assertTrue(
            np.isfinite(
                heat.calculate_heat(temperature, [0, 2000], [[np.nan]], model)["lcoh"]
            ).all()
        )
        model["include_egs"] = True
        egs = heat.calculate_heat(temperature, [0, 2000], [[1000]], model)
        self.assertAlmostEqual(egs["capacity"].item(), capacity * 0.6)
        self.assertGreater(egs["lcoh_stimulation"].item(), 0)

    def test_joint_allocation_conservation_and_cost(self):
        """Serve a constrained network while routing flexible demand elsewhere."""
        edges = np.array([[0, 0], [0, 1], [1, 0], [2, 2], [3, 2]])
        flow = allocation.solve_allocation(
            edges, [10, 10, 4, 0], [10, 20, 2], [5, 10, 1]
        )
        np.testing.assert_allclose(flow, [0, 10, 10, 2, 0])
        flow = allocation.solve_allocation([[0, 0], [1, 0]], [2, 3], [10], [5])
        np.testing.assert_allclose(flow, [2, 3])
        self.assertEqual(
            len(allocation.solve_allocation(np.empty((0, 2), dtype=int), [10], [], [])),
            0,
        )

    def test_candidates_centroids_and_fallback(self):
        """Use polygon buffers, stable boundary ties, and nearest intersecting regions."""
        regions = gpd.GeoDataFrame(
            {"shape_id": ["b", "a"]},
            geometry=[
                box(4010000, 3000000, 4020000, 3010000),
                box(4000000, 3000000, 4010000, 3010000),
            ],
            crs=3035,
        )
        networks = gpd.GeoDataFrame(
            {"area_id": ["tie", "gap"], DEMAND_COLUMN: [10, 10]},
            geometry=[
                box(4009000, 3004000, 4011000, 3006000),
                box(3998000, 3004000, 4000100, 3006000),
            ],
            crs=3035,
        )
        self.assertEqual(
            allocation.assign_network_regions(networks, regions).tolist(), ["a", "a"]
        )
        costs = self.raster([[10, 20]])
        edges = allocation.candidate_edges(networks, costs, costs, 5)
        self.assertEqual(edges.tolist(), [[0, 0], [0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "no intersecting region"):
            allocation.assign_network_regions(
                networks.assign(geometry=networks.geometry.translate(xoff=-1e6)),
                regions,
            )

    def test_regional_costs_and_alternatives(self):
        """Classify retained power only and count contested heat once across cycles."""
        costs = self.raster([[50, 76, 75]])
        capacity = self.raster([[100, 200, 300]])
        eligible = regional.power_eligibility(costs, capacity, "cutoff", 4, [], 75)
        self.assertEqual(eligible.values.tolist(), [[True, False, True]])
        ledger = pd.DataFrame(
            {
                "region_id": ["consumer", "consumer", "consumer"],
                "row": [0, 0, 0],
                "col": [0, 1, 2],
                "cell_id": [0, 1, 2],
                "heat_mwh_per_year": [10, 30, 20],
                "lcoh_eur_per_mwh": [10, 20, 40],
            }
        )
        result = regional.aggregate_heat(
            ledger, eligible, pd.Index(["consumer", "empty"]), True
        ).set_index(["region_id", "category"])
        self.assertEqual(result.loc[("consumer", "contested"), "potential"], 30)
        self.assertEqual(result.loc[("consumer", "contested"), "mean_lcoh"], 30)
        self.assertEqual(result.loc[("consumer", "heat"), "potential"], 30)
        self.assertEqual(result.loc[("empty", "heat"), "potential"], 0)
        self.assertTrue(np.isnan(result.loc[("empty", "heat"), "mean_lcoh"]))
        for method, bins in [("quantiles", []), ("fixed_bins", [(0, 75)])]:
            mask = regional.power_eligibility(costs, capacity, method, 2, bins, None)
            self.assertTrue(mask.values[0, 0])
            self.assertEqual(bool(mask.values[0, 1]), method == "quantiles")

    def test_real_allocation_and_regional_scripts(self):
        """Exercise raster files, weighted network results, manifests and strict schemas."""
        networks = gpd.GeoDataFrame(
            {
                "area_id": [1, 2],
                DEMAND_COLUMN: [20.0, 0.0],
                "levelized_cost_eur_per_mwh": [99.0, 99.0],
            },
            geometry=[
                box(4001000, 3001000, 4009000, 3009000),
                box(4011000, 3001000, 4019000, 3009000),
            ],
            crs=3035,
        )
        regions = gpd.GeoDataFrame(
            {"shape_id": ["region"]},
            geometry=[box(4000000, 3000000, 4020000, 3010000)],
            crs=3035,
        )
        networks.to_parquet(self.folder / "networks.parquet")
        regions.to_parquet(self.folder / "shapes.parquet")
        self.raster([[10, 20]], "heat_lcoh.tif")
        self.raster([[100, 100]], "heat_potential.tif")
        output = {
            name: str(self.folder / filename)
            for name, filename in {
                "networks": "supplied.parquet",
                "allocations": "allocations.parquet",
                "allocated": "allocated.tif",
                "fraction": "fraction.tif",
            }.items()
        }
        run_script(
            "allocate_heat",
            {
                "networks": str(self.folder / "networks.parquet"),
                "shapes": str(self.folder / "shapes.parquet"),
                "lcoh": str(self.folder / "heat_lcoh.tif"),
                "potential": str(self.folder / "heat_potential.tif"),
            },
            output,
            {"buffer_km": 0},
            self.folder,
        )
        supplied = gpd.read_parquet(output["networks"])
        self.assertEqual(supplied.geothermal_supplied_mwh_per_year.tolist(), [20, 0])
        self.assertEqual(supplied.levelized_cost_eur_per_mwh.tolist(), [99, 99])
        self.assertTrue(np.isnan(supplied.geothermal_lcoh_eur_per_mwh.iloc[1]))
        self.raster([[50, 76]], "orc_2d_lcoe.tif")
        self.raster([[100, 200]], "orc_2d_potential.tif")
        inputs = {
            "shapes": str(self.folder / "shapes.parquet"),
            "lcoe_tifs": [str(self.folder / "orc_2d_lcoe.tif")],
            "pot_tifs": [str(self.folder / "orc_2d_potential.tif")],
            "heat_fraction": [output["fraction"]],
            "heat_allocations": [output["allocations"]],
        }
        params = {
            "carrier_out": "heat_and_power",
            "cycles": ["orc"],
            "method_region_aggregation": "cutoff",
            "lcoe_cutoff": 75,
        }
        regional_outputs = {
            "geoparquet": str(self.folder / "regions.parquet"),
            "manifest": str(self.folder / "manifest.tsv"),
        }
        run_script(
            "aggregate_to_regions", inputs, regional_outputs, params, self.folder
        )
        result = gpd.read_parquet(regional_outputs["geoparquet"]).set_index(
            ["category", "carrier"]
        )
        self.assertAlmostEqual(result.loc[("power", "power"), "potential"], 80)
        self.assertAlmostEqual(result.loc[("contested", "power"), "potential"], 20)
        self.assertAlmostEqual(result.loc[("contested", "heat"), "potential"], 20)
        self.assertEqual(result.loc[("contested", "heat"), "mean_lcoh"], 10)
        # A second power alternative must not duplicate the contested heat.
        self.raster([[60, 90]], "flash_2d_lcoe.tif")
        self.raster([[50, 300]], "flash_2d_potential.tif")
        inputs["lcoe_tifs"].append(str(self.folder / "flash_2d_lcoe.tif"))
        inputs["pot_tifs"].append(str(self.folder / "flash_2d_potential.tif"))
        params["cycles"] = ["orc", "flash"]
        run_script(
            "aggregate_to_regions", inputs, regional_outputs, params, self.folder
        )
        result = gpd.read_parquet(regional_outputs["geoparquet"])
        contested_heat = result[
            (result.category == "contested") & (result.carrier == "heat")
        ]
        self.assertEqual(contested_heat.potential.tolist(), [20])
        self.assertEqual(
            result[
                (result.category == "contested") & (result.carrier == "power")
            ].potential.sum(),
            30,
        )
        # Heat-only aggregation needs no power rasters or price cutoff.
        inputs.update(lcoe_tifs=[], pot_tifs=[])
        params.update(carrier_out="heat", cycles=[], lcoe_cutoff=None)
        run_script(
            "aggregate_to_regions", inputs, regional_outputs, params, self.folder
        )
        result = gpd.read_parquet(regional_outputs["geoparquet"])
        self.assertEqual(result.category.tolist(), ["heat"])
        self.assertEqual(result.potential.tolist(), [20])

    def test_cross_region_resource_links(self):
        """Keep heat in its consumer region and contested electricity at its source."""
        networks = gpd.GeoDataFrame(
            {"area_id": [1], DEMAND_COLUMN: [20.0]},
            geometry=[box(4011000, 3001000, 4019000, 3009000)],
            crs=3035,
        )
        regions = gpd.GeoDataFrame(
            {"shape_id": ["source", "consumer"]},
            geometry=[
                box(4000000, 3000000, 4010000, 3010000),
                box(4010000, 3000000, 4020000, 3010000),
            ],
            crs=3035,
        )
        networks.to_parquet(self.folder / "networks.parquet")
        regions.to_parquet(self.folder / "shapes.parquet")
        for name, values in [
            ("heat_lcoh", [[10, 20]]),
            ("heat_potential", [[100, 100]]),
            ("orc_2d_lcoe", [[50, 100]]),
            ("orc_2d_potential", [[100, 100]]),
        ]:
            self.raster(values, f"{name}.tif")
        outputs = {
            name: str(self.folder / filename)
            for name, filename in {
                "networks": "served.parquet",
                "allocations": "ledger.parquet",
                "allocated": "used.tif",
                "fraction": "fraction.tif",
            }.items()
        }
        inputs = {
            "networks": str(self.folder / "networks.parquet"),
            "shapes": str(self.folder / "shapes.parquet"),
            "lcoh": str(self.folder / "heat_lcoh.tif"),
            "potential": str(self.folder / "heat_potential.tif"),
        }
        run_script("allocate_heat", inputs, outputs, {"buffer_km": 10}, self.folder)
        ledger = pd.read_parquet(outputs["allocations"])
        self.assertEqual(ledger.region_id.tolist(), ["consumer"])
        self.assertEqual(ledger.col.tolist(), [0])
        regional_inputs = {
            "shapes": inputs["shapes"],
            "lcoe_tifs": [str(self.folder / "orc_2d_lcoe.tif")],
            "pot_tifs": [str(self.folder / "orc_2d_potential.tif")],
            "heat_fraction": [outputs["fraction"]],
            "heat_allocations": [outputs["allocations"]],
        }
        params = {
            "carrier_out": "heat_and_power",
            "cycles": ["orc"],
            "method_region_aggregation": "cutoff",
            "lcoe_cutoff": 75,
        }
        regional_outputs = {
            "geoparquet": str(self.folder / "regions.parquet"),
            "manifest": str(self.folder / "manifest.tsv"),
        }
        run_script(
            "aggregate_to_regions",
            regional_inputs,
            regional_outputs,
            params,
            self.folder,
        )
        result = gpd.read_parquet(regional_outputs["geoparquet"])
        contested = result[
            (result.category == "contested") & (result.potential > 0)
        ].set_index("carrier")
        self.assertEqual(contested.loc["heat", "region_id"], "consumer")
        self.assertEqual(contested.loc["power", "region_id"], "source")
        self.assertAlmostEqual(contested.loc["power", "potential"], 20)


if __name__ == "__main__":
    unittest.main()
