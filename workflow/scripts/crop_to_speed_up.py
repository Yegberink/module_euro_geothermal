"""Crop the European temperature voxel to a supplied study area."""

import sys
from pathlib import Path

import _schemas
import geopandas as gpd
import xarray as xr

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

CRS_TARGET = "EPSG:3857"


def load_shapes(path: Path) -> gpd.GeoDataFrame:
    """Load and validate study-area polygons."""
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    return _schemas.ShapeSchema.validate(gdf)


def bbox_3857_from_shapes(gdf: gpd.GeoDataFrame):
    """Return the polygon bounds in the temperature model's CRS."""
    gdf_3857 = gdf.to_crs(CRS_TARGET)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    return (minx, miny, maxx, maxy)


def crop_ds_bbox_projected(ds, x_name="x", y_name="y", bbox=None, pad_m=20_000):
    """Crop a projected dataset to a padded bounding box."""
    minx, miny, maxx, maxy = bbox
    minx -= pad_m
    maxx += pad_m
    miny -= pad_m
    maxy += pad_m

    x = ds[x_name]
    y = ds[y_name]

    # Handle ascending/descending coords
    y_slice = slice(miny, maxy) if float(y[0]) < float(y[-1]) else slice(maxy, miny)
    x_slice = slice(minx, maxx) if float(x[0]) < float(x[-1]) else slice(maxx, minx)

    return ds.sel({x_name: x_slice, y_name: y_slice})


def main():
    """Crop the temperature voxel requested by Snakemake."""
    voxel_nc = Path(snakemake.input.nc)
    shapes_path = Path(snakemake.input.shapes)
    out_nc = Path(snakemake.output.clipped_temp_voxel)

    pad_m = float(snakemake.params.get("pad_m", 25_000))

    ds = xr.open_dataset(voxel_nc, chunks="auto")

    gdf = load_shapes(shapes_path)
    bbox = bbox_3857_from_shapes(gdf)

    ds_small = crop_ds_bbox_projected(
        ds, x_name="x", y_name="y", bbox=bbox, pad_m=pad_m
    )
    if ds_small.sizes.get("x", 1) == 0 or ds_small.sizes.get("y", 1) == 0:
        ds_small = _schemas.empty_dataset_like(ds.isel(x=slice(0, 1), y=slice(0, 1)))

    ds_small.to_netcdf(out_nc)


if __name__ == "__main__":
    main()
