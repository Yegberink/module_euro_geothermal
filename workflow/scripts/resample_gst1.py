"""Resample the GST-1 sediment-thickness model to the temperature grid."""

import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xarray as xr
from _schemas import validate_gst1, validate_source_grid
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

GST1_TITLE = "GST-1: A high-resolution Global Sediment Thickness model"


def read_gst1_xyz(archive_path, member):
    """Return ascending latitude, longitude, and thickness arrays from GST-1.

    The published header says ``Lon Lat meter``, but the first two columns are
    latitude and longitude respectively. Rows are ordered by latitude, with
    longitude varying fastest.
    """
    with ZipFile(archive_path) as archive:
        with archive.open(member) as xyz_file:
            records = np.loadtxt(xyz_file, skiprows=1, dtype=np.float32)

    return validate_gst1(records)


def reproject_nearest(thickness, latitudes, longitudes, x, y, target_crs):
    """Reproject validated regular grids onto exact target centers."""
    longitude_spacing = float(longitudes[1] - longitudes[0])
    latitude_spacing = float(latitudes[1] - latitudes[0])
    source_transform = from_origin(
        float(longitudes[0] - 0.5 * longitude_spacing),
        float(latitudes[-1] + 0.5 * latitude_spacing),
        longitude_spacing,
        latitude_spacing,
    )

    x, y = np.asarray(x), np.asarray(y)
    x_spacing = float(abs(x[1] - x[0]))
    y_spacing = float(abs(y[1] - y[0]))
    target_transform = from_origin(
        float(x.min() - 0.5 * x_spacing),
        float(y.max() + 0.5 * y_spacing),
        x_spacing,
        y_spacing,
    )

    north_up_west_to_east = np.full((y.size, x.size), np.nan, dtype=np.float32)
    reproject(
        source=np.asarray(thickness, dtype=np.float32)[::-1, :],
        destination=north_up_west_to_east,
        src_transform=source_transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=target_transform,
        dst_crs=CRS.from_user_input(target_crs),
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    result = north_up_west_to_east
    if y[0] < y[-1]:
        result = result[::-1, :]
    if x[0] > x[-1]:
        result = result[:, ::-1]
    return result


def resample_gst1_to_temperature_grid(
    archive_path, member, temperature_grid, *, doi="10.5281/zenodo.15320474"
):
    """Build the aligned GST-1 dataset for a temperature-grid dataset."""
    validate_source_grid(temperature_grid)
    target_crs = str(temperature_grid.attrs["crs"]).strip()

    latitudes, longitudes, thickness = read_gst1_xyz(archive_path, member)
    x = temperature_grid["x"].values
    y = temperature_grid["y"].values
    aligned = reproject_nearest(thickness, latitudes, longitudes, x, y, target_crs)

    sediment = xr.DataArray(
        aligned,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name="sediment_thickness_m",
        attrs={
            "long_name": "GST-1 sediment thickness",
            "units": "m",
            "crs": target_crs,
        },
    )
    return sediment.to_dataset().assign_attrs(
        title="GST-1 sediment thickness resampled to the temperature grid",
        source_dataset=GST1_TITLE,
        source_version="Version 1",
        source_doi=doi,
        source_file=member,
        resampling="nearest",
        crs=target_crs,
    )


def main():
    """Run GST-1 alignment using paths supplied by Snakemake."""
    archive_path = Path(snakemake.input.archive)
    temperature_path = Path(snakemake.input.temperature)
    output_path = Path(snakemake.output.sediment)

    with xr.open_dataset(temperature_path) as temperature_grid:
        result = resample_gst1_to_temperature_grid(
            archive_path,
            snakemake.params.member,
            temperature_grid,
            doi=snakemake.params.doi,
        )
        result.to_netcdf(output_path)


if __name__ == "__main__":
    main()
