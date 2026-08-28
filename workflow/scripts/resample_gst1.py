"""Resample the GST-1 sediment-thickness model to the temperature grid."""

import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

GST1_SHAPE = (1441, 2881)
GST1_RESOLUTION_DEGREES = 0.125
GST1_BOUNDS = (-90.0, 90.0, -180.0, 180.0)
GST1_TITLE = "GST-1: A high-resolution Global Sediment Thickness model"


def read_gst1_xyz(
    archive_path,
    member,
    *,
    expected_shape=GST1_SHAPE,
    expected_resolution=GST1_RESOLUTION_DEGREES,
    expected_bounds=GST1_BOUNDS,
):
    """Return ascending latitude, longitude, and thickness arrays from GST-1.

    The published header says ``Lon Lat meter``, but the first two columns are
    latitude and longitude respectively. Rows are ordered by latitude, with
    longitude varying fastest.
    """
    with ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"GST-1 archive does not contain {member!r}")
        with archive.open(member) as xyz_file:
            records = np.loadtxt(xyz_file, skiprows=1, dtype=np.float32)

    if records.ndim != 2 or records.shape[1] != 3:
        raise ValueError("GST-1 XYZ data must contain exactly three columns")

    latitude_column = records[:, 0]
    longitude_column = records[:, 1]
    thickness_column = records[:, 2]

    latitude_starts = np.r_[0, np.flatnonzero(np.diff(latitude_column) != 0.0) + 1]
    latitudes = latitude_column[latitude_starts]
    longitude_count = (
        int(latitude_starts[1]) if latitude_starts.size > 1 else records.shape[0]
    )
    longitudes = longitude_column[:longitude_count]

    shape = (latitudes.size, longitudes.size)
    if records.shape[0] != shape[0] * shape[1]:
        raise ValueError("GST-1 rows do not form a complete rectangular grid")
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f"Expected GST-1 grid shape {expected_shape}, found {shape}")

    latitude_grid = latitude_column.reshape(shape)
    longitude_grid = longitude_column.reshape(shape)
    if not np.all(latitude_grid == latitudes[:, np.newaxis]):
        raise ValueError("GST-1 latitude ordering is inconsistent")
    if not np.all(longitude_grid == longitudes[np.newaxis, :]):
        raise ValueError("GST-1 longitude ordering is inconsistent")
    if not np.all(np.diff(latitudes) > 0.0) or not np.all(np.diff(longitudes) > 0.0):
        raise ValueError("GST-1 coordinates must be strictly ascending")

    if expected_resolution is not None:
        if not np.allclose(np.diff(latitudes), expected_resolution) or not np.allclose(
            np.diff(longitudes), expected_resolution
        ):
            raise ValueError(
                f"GST-1 coordinates must use a {expected_resolution} degree grid"
            )

    if expected_bounds is not None:
        actual_bounds = (
            float(latitudes[0]),
            float(latitudes[-1]),
            float(longitudes[0]),
            float(longitudes[-1]),
        )
        if not np.allclose(actual_bounds, expected_bounds):
            raise ValueError(
                f"Expected GST-1 bounds {expected_bounds}, found {actual_bounds}"
            )

    if not np.all(np.isfinite(thickness_column)):
        raise ValueError("GST-1 sediment thickness contains non-finite values")
    if np.any(thickness_column < 0.0):
        raise ValueError("GST-1 sediment thickness cannot be negative")

    return latitudes, longitudes, thickness_column.reshape(shape)


def _regular_grid_spacing(coordinates, name):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(
            f"Temperature-grid coordinate {name!r} must be one-dimensional"
        )

    differences = np.diff(coordinates)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(f"Temperature-grid coordinate {name!r} must be monotonic")
    if not np.allclose(np.abs(differences), abs(differences[0])):
        raise ValueError(
            f"Temperature-grid coordinate {name!r} must be regularly spaced"
        )
    return float(abs(differences[0]))


def reproject_nearest(thickness, latitudes, longitudes, x, y, target_crs):
    """Reproject a regular WGS84 thickness grid onto exact target centers."""
    longitude_spacing = float(longitudes[1] - longitudes[0])
    latitude_spacing = float(latitudes[1] - latitudes[0])
    source_transform = from_origin(
        float(longitudes[0] - 0.5 * longitude_spacing),
        float(latitudes[-1] + 0.5 * latitude_spacing),
        longitude_spacing,
        latitude_spacing,
    )

    x = np.asarray(x)
    y = np.asarray(y)
    x_spacing = _regular_grid_spacing(x, "x")
    y_spacing = _regular_grid_spacing(y, "y")
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
    archive_path,
    member,
    temperature_grid,
    *,
    doi="10.5281/zenodo.15320474",
    expected_shape=GST1_SHAPE,
    expected_resolution=GST1_RESOLUTION_DEGREES,
    expected_bounds=GST1_BOUNDS,
):
    """Build the aligned GST-1 dataset for a temperature-grid dataset."""
    if "x" not in temperature_grid.coords or "y" not in temperature_grid.coords:
        raise ValueError("Temperature grid must provide x and y coordinates")
    target_crs = str(temperature_grid.attrs.get("crs", "")).strip()
    if not target_crs:
        raise ValueError("Temperature grid must define its CRS in the 'crs' attribute")

    latitudes, longitudes, thickness = read_gst1_xyz(
        archive_path,
        member,
        expected_shape=expected_shape,
        expected_resolution=expected_resolution,
        expected_bounds=expected_bounds,
    )
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
