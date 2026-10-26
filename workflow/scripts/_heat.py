"""Shared heat input validation and raster I/O, independent of model economics."""

from pathlib import Path

import _schemas
import geopandas as gpd
import numpy as np
import rioxarray as rxr
from affine import Affine

METRIC_CRS = "EPSG:3035"
DEMAND_COLUMN = _schemas.DEMAND_COLUMN


def assign_cell_regions(cells, regions):
    """Assign centres to regions, breaking boundary/overlap ties by region ID.

    This ownership defines which source cells may supply each consumer region
    when cross-region supply is disabled.
    """
    regions = regions.to_crs(cells.crs)[["shape_id", "geometry"]].copy()
    regions["shape_id"] = regions.shape_id.astype(str)
    matches = gpd.sjoin(
        cells[["geometry"]], regions, how="left", predicate="intersects"
    )
    return (
        matches.sort_values("shape_id")
        .groupby(level=0)
        .shape_id.first()
        .reindex(cells.index)
    )


def read_raster(path):
    """Load one raster band without collapsing singleton spatial dimensions."""
    with rxr.open_rasterio(path, masked=True) as source:
        data = source.isel(band=0, drop=True).load().astype("float64")
        transform = source.rio.transform()
    data = data.sortby("x").sortby("y", ascending=False)
    dx, dy = abs(transform.a), -abs(transform.e)
    return data.rio.write_transform(
        Affine(dx, 0, float(data.x[0]) - dx / 2, 0, dy, float(data.y[0]) - dy / 2)
    )


def check_grid(left, right):
    """Require identical grids before combining costs, quantities, or fractions."""
    if (
        left.rio.crs != right.rio.crs
        or left.shape != right.shape
        or not left.rio.transform().almost_equals(right.rio.transform())
        or not np.array_equal(left.x, right.x)
        or not np.array_equal(left.y, right.y)
    ):
        raise ValueError("Heat raster grids differ.")


def write_raster(data, path, units):
    """Write a north-up compressed raster with explicit units and NaN nodata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = data.sortby("y", ascending=False).transpose("y", "x")
    data.attrs["units"] = units
    data.rio.write_nodata(np.nan).rio.to_raster(path, compress="deflate")


def align_to_grid(source, target, fill=0.0):
    """Select matching source-grid centres, allowing different cropped extents.

    No interpolation is permitted for resource fractions or eligibility masks.
    Both grids must be crops of the same original temperature grid.
    """
    if source.rio.crs != target.rio.crs:
        raise ValueError("Heat and power raster CRS differ.")
    a, b = source.rio.transform(), target.rio.transform()
    if not np.allclose([a.a, a.e, a.b, a.d], [b.a, b.e, b.b, b.d]):
        raise ValueError("Heat and power grid resolutions differ.")
    offsets = [(a.c - b.c) / a.a, (a.f - b.f) / a.e]
    if not np.allclose(offsets, np.round(offsets)):
        raise ValueError("Heat and power grid origins differ.")
    return source.reindex(x=target.x, y=target.y, fill_value=fill).fillna(fill)
