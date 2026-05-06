"""Focused checks for schema-shaped empty outputs."""

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr
from shapely.geometry import Polygon

SCRIPTS = Path(__file__).parents[1] / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _schemas  # noqa: E402
from make_2d import write_geotiff  # noqa: E402


def test_empty_table_helpers(tmp_path):
    """Empty table helpers should write readable schema-shaped files."""
    gdf = _schemas.GeothermalRegionsSchema.validate(
        _schemas.empty_geothermal_regions("EPSG:4326")
    )
    out = tmp_path / "regions.parquet"
    gdf.to_parquet(out)
    assert out.exists()

    manifest = pd.DataFrame(
        [{"kind": "combined_parquet", "depth_category": "all", "bin_id": "all", "price_bin": "all", "path": str(out)}]
    )
    _schemas.write_manifest(manifest, tmp_path / "manifest.tsv")
    assert pd.read_csv(tmp_path / "manifest.tsv", sep="\t").columns.tolist() == _schemas.MANIFEST_COLUMNS


def test_shape_schema_accepts_valid_geometry():
    """Shape schema should accept a minimal valid user shape."""
    gdf = gpd.GeoDataFrame(
        {"shape_id": ["a"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]},
        crs="EPSG:4326",
    )
    assert len(_schemas.ShapeSchema.validate(gdf)) == 1


def test_empty_binary_helpers(tmp_path):
    """Empty binary helpers should write readable placeholder artifacts."""
    ds = xr.Dataset(
        {"temperature": (("depth_m", "y", "x"), np.ones((1, 1, 1), dtype=np.float32))},
        coords={"depth_m": [0.0], "y": [0.0], "x": [0.0]},
    )
    _schemas.write_empty_netcdf_like(ds, tmp_path / "empty.nc")
    assert np.isnan(xr.open_dataset(tmp_path / "empty.nc")["temperature"].values).all()

    da = xr.DataArray(
        np.full((2, 2), np.nan, dtype=np.float32),
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]},
        dims=("y", "x"),
    )
    write_geotiff(da, tmp_path / "empty.tif", crs="EPSG:4326")
    with rasterio.open(tmp_path / "empty.tif") as src:
        assert src.read(1).shape == (2, 2)

    _schemas.write_empty_html(tmp_path / "empty.html")
    assert (tmp_path / "empty.html").read_text().startswith("<!doctype html>")

    _schemas.write_blank_png(tmp_path / "empty.png")
    assert (tmp_path / "empty.png").read_bytes().startswith(b"\x89PNG")
