"""Crop the European temperature voxel to a supplied study area."""

import sys
from pathlib import Path

import _schemas
import geopandas as gpd
import numpy as np
import xarray as xr

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


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

    pad_m = float(snakemake.params.pad_m)

    ds = xr.open_dataset(voxel_nc, chunks="auto")

    gdf = _schemas.ShapeSchema.validate(
        gpd.read_parquet(shapes_path)
        if shapes_path.suffix.lower() == ".parquet"
        else gpd.read_file(shapes_path)
    )
    _schemas.SourceGridSchema.validate(ds)
    bbox = gdf.to_crs(ds.attrs["crs"]).total_bounds

    ds_small = crop_ds_bbox_projected(
        ds, x_name="x", y_name="y", bbox=bbox, pad_m=pad_m
    )
    if ds_small.sizes.get("x", 1) == 0 or ds_small.sizes.get("y", 1) == 0:
        template = ds.isel(x=slice(0, 1), y=slice(0, 1))
        ds_small = xr.Dataset(
            {
                name: xr.full_like(var.astype("float32"), np.nan).assign_attrs(
                    var.attrs
                )
                for name, var in template.data_vars.items()
            },
            coords=template.coords,
            attrs=dict(template.attrs),
        )

    ds_small.to_netcdf(out_nc)


if __name__ == "__main__":
    main()
