"""Validation at external input and final output boundaries."""

# ruff: noqa: PT009, PT027

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandera.errors
import xarray as xr
from affine import Affine
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/scripts"))
from _schemas import (  # noqa: E402
    DEMAND_COLUMN,
    GeothermalRegionsSchema,
    HeatNetworksSchema,
    ShapeSchema,
    empty_geothermal_regions,
    validate_area_potential,
    validate_fractions,
    validate_gst1,
    validate_temperature,
)


class BoundaryValidationTests(unittest.TestCase):
    """Keep useful failures for supplied and downloaded data."""

    def test_polygons(self):
        """Require unique IDs, polygon geometry and a CRS."""
        shapes = gpd.GeoDataFrame(
            {"shape_id": [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326
        )
        self.assertEqual(ShapeSchema.validate(shapes).shape_id.tolist(), ["1", "2"])
        for invalid in (
            shapes.assign(shape_id=[1, 1]),
            shapes.set_geometry([Point(0, 0), Point(1, 1)]),
            shapes.set_crs(None, allow_override=True),
            shapes.set_geometry([None, box(0, 0, 1, 1)]),
        ):
            with self.assertRaises((ValueError, pandera.errors.SchemaError)):
                ShapeSchema.validate(invalid)

    def test_network_demand(self):
        """Demand must be finite and nonnegative."""
        networks = gpd.GeoDataFrame(
            {"area_id": ["a"], DEMAND_COLUMN: [0.0]},
            geometry=[box(0, 0, 1, 1)],
            crs=4326,
        )
        HeatNetworksSchema.validate(networks)
        for value in (-1, np.nan, np.inf):
            with self.assertRaises(pandera.errors.SchemaError):
                HeatNetworksSchema.validate(networks.assign(**{DEMAND_COLUMN: value}))

    def test_downloaded_temperature(self):
        """Source axes support georeferencing and depth interpolation."""
        source = xr.Dataset(
            {"temperature": (("depth_m", "y", "x"), np.full((2, 2, 2), np.nan))},
            coords={"depth_m": [100, 200], "x": [0.0, 1.0], "y": [1.0, 0.0]},
            attrs={"crs": "EPSG:4326"},
        )
        validate_temperature(source)
        for invalid in (
            source.assign_coords(depth_m=[200, 100]),
            source.assign_coords(depth_m=[100, np.nan]),
            source.assign_coords(x=[0, 0]),
        ):
            with self.assertRaises(ValueError):
                validate_temperature(invalid)

    def test_gst1_interpretation(self):
        """Preserve XYZ ordering without requiring a fixed release extent."""
        records = np.array(
            [[lat, lon, lat + lon] for lat in (0, 1) for lon in (2, 3, 4)]
        )
        lat, lon, thickness = validate_gst1(records)
        np.testing.assert_array_equal(lat, [0, 1])
        np.testing.assert_array_equal(lon, [2, 3, 4])
        np.testing.assert_array_equal(thickness, [[2, 3, 4], [3, 4, 5]])
        for invalid in (
            records[:, :2],
            records[:-1],
            records[::-1],
            records[:3],
            records * np.nan,
            np.column_stack((records[:, :2], -np.ones(6))),
            np.array([[lat, lon, 1] for lat in (0, 1) for lon in (0, 1, 3)]),
        ):
            with self.assertRaises(ValueError):
                validate_gst1(invalid)

    def test_supplied_area_potential(self):
        """Check metadata and normalized values of external area rasters."""
        source = dict(count=1, crs="EPSG:4326", transform=Affine(1, 0, 0, 0, -1, 2))
        validate_area_potential(SimpleNamespace(**source))
        for changes in (
            {"count": 2},
            {"crs": "EPSG:3857"},
            {"transform": Affine.identity()},
        ):
            with self.assertRaises(ValueError):
                validate_area_potential(SimpleNamespace(**(source | changes)))
        validate_fractions(np.array([0, 1, 1 + 1e-6]))
        for value in (-1e-6, 1.01, np.nan, np.inf):
            with self.assertRaises(ValueError):
                validate_fractions(np.array([value]))

    def test_empty_final_output(self):
        """Empty output retains the published columns, types and CRS."""
        result = empty_geothermal_regions("EPSG:4326")
        GeothermalRegionsSchema.schema.validate(result)
        self.assertEqual(
            list(result.columns), list(GeothermalRegionsSchema.schema.columns)
        )
        self.assertEqual(result.crs.to_epsg(), 4326)


if __name__ == "__main__":
    unittest.main()
