"""Schemas and empty output helpers."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.pandas as pa
import xarray as xr

DEMAND_COLUMN = "heat_potential_mwh_per_year"


def _valid_geometry(series):
    return series.notna() & ~series.is_empty & series.is_valid


class ShapeSchema:
    """Contract for user-provided aggregation polygons."""

    schema = pa.DataFrameSchema(
        {
            "shape_id": pa.Column(str, unique=True, coerce=True),
            "geometry": pa.Column(checks=pa.Check(_valid_geometry)),
        },
        strict=False,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        """Validate tabular, CRS, and polygon-geometry requirements."""
        if not isinstance(df, gpd.GeoDataFrame):
            raise TypeError("Shapes must be supplied as a GeoDataFrame.")
        if df.crs is None:
            raise ValueError("Shapes must declare a coordinate reference system (CRS).")
        validated = cls.schema.validate(df)
        polygon_types = validated.geometry.geom_type.isin({"Polygon", "MultiPolygon"})
        if not polygon_types.all():
            raise ValueError(
                "Shapes must contain only Polygon or MultiPolygon geometry."
            )
        return validated


class HeatNetworksSchema(ShapeSchema):
    """Network polygons with unique IDs and finite, nonnegative annual demand."""

    schema = ShapeSchema.schema.rename_columns({"shape_id": "area_id"}).add_columns(
        {
            DEMAND_COLUMN: pa.Column(
                float,
                checks=pa.Check(
                    lambda s: np.isfinite(s) & (s >= 0),
                    error="Network heat demand must be finite and nonnegative.",
                    ignore_na=False,
                ),
                nullable=True,
                coerce=True,
            )
        }
    )


class SourceGridSchema:
    """Coordinate contract for the regular subsurface grid."""

    @staticmethod
    def validate(source):
        """Require at least two finite, regularly spaced centres per axis."""
        for axis in (source.x.values, source.y.values):
            if len(axis) < 2:
                raise ValueError(
                    "The source grid must contain at least two cells per axis."
                )
            spacing = np.diff(axis)
            if (
                not np.isfinite(axis).all()
                or spacing[0] == 0
                or not np.allclose(spacing, spacing[0])
            ):
                raise ValueError("Eligibility requires a regular source grid.")
        return source


class TemperatureDepthSchema:
    """Depth coordinates used for temperature interpolation."""

    @staticmethod
    def validate(depths):
        """Require finite, strictly increasing depths."""
        if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
            raise ValueError(
                "Temperature depths must be finite and strictly increasing."
            )
        return depths


class EligibilityFractionSchema:
    """Available-area fractions on the model grid."""

    @staticmethod
    def validate(values):
        """Require finite fractions in the closed unit interval."""
        array = np.asarray(values)
        if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError("Eligibility fractions must be finite and within [0,1].")
        return values


class GeothermalRegionsSchema:
    schema = pa.DataFrameSchema(
        {
            "category": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["power", "heat", "contested"]))
            ),
            "carrier": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["power", "heat"]))
            ),
            "potential_unit": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["MW_e", "MWh_th/year"]))
            ),
            "mean_lcoh": pa.Column(float, nullable=True, coerce=True),
            "technology": pa.Column(str, coerce=True),
            "depth_category": pa.Column(str, coerce=True),
            "region_id": pa.Column(str, coerce=True),
            "price_bin": pa.Column(str, coerce=True),
            "bin_id": pa.Column(str, coerce=True),
            "bin_left": pa.Column(float, nullable=True, coerce=True),
            "bin_right": pa.Column(float, nullable=True, coerce=True),
            "potential": pa.Column(float, nullable=True, coerce=True),
            "mean_lcoe": pa.Column(float, nullable=True, coerce=True),
            "n_cells": pa.Column(int, coerce=True),
            "geometry": pa.Column(checks=pa.Check(_valid_geometry)),
        },
        strict=True,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        return cls.schema.validate(df)


class GeothermalRegionManifestSchema:
    schema = pa.DataFrameSchema(
        {
            col: pa.Column(str, coerce=True)
            for col in ["kind", "depth_category", "bin_id", "price_bin", "path"]
        },
        strict=True,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        return cls.schema.validate(df)


GEOTHERMAL_REGION_COLUMNS = [
    "category",
    "carrier",
    "potential_unit",
    "mean_lcoh",
    "technology",
    "depth_category",
    "region_id",
    "price_bin",
    "bin_id",
    "bin_left",
    "bin_right",
    "potential",
    "mean_lcoe",
    "n_cells",
]
MANIFEST_COLUMNS = ["kind", "depth_category", "bin_id", "price_bin", "path"]


def empty_geothermal_regions(crs=None) -> gpd.GeoDataFrame:
    df = pd.DataFrame(
        {
            "category": pd.Series(dtype="object"),
            "carrier": pd.Series(dtype="object"),
            "potential_unit": pd.Series(dtype="object"),
            "mean_lcoh": pd.Series(dtype="float64"),
            "technology": pd.Series(dtype="object"),
            "depth_category": pd.Series(dtype="object"),
            "region_id": pd.Series(dtype="object"),
            "price_bin": pd.Series(dtype="object"),
            "bin_id": pd.Series(dtype="object"),
            "bin_left": pd.Series(dtype="float64"),
            "bin_right": pd.Series(dtype="float64"),
            "potential": pd.Series(dtype="float64"),
            "mean_lcoe": pd.Series(dtype="float64"),
            "n_cells": pd.Series(dtype="int64"),
        }
    )
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in MANIFEST_COLUMNS})


def ensure_parent(path) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def empty_dataarray_like(da: xr.DataArray) -> xr.DataArray:
    out = xr.full_like(da.astype("float32"), np.nan)
    out.attrs.update(da.attrs)
    return out


def empty_dataset_like(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(
        {name: empty_dataarray_like(var) for name, var in ds.data_vars.items()},
        coords=ds.coords,
        attrs=dict(ds.attrs),
    )
    for name in out.data_vars:
        out[name].attrs.update(ds[name].attrs)
    return out


def write_empty_netcdf_like(ds: xr.Dataset, path) -> None:
    ensure_parent(path)
    empty_dataset_like(ds).to_netcdf(path)


def empty_raster_like(da: xr.DataArray) -> xr.DataArray:
    out = empty_dataarray_like(da)
    if hasattr(da, "rio"):
        out = out.rio.write_crs(da.rio.crs, inplace=False)
        out = out.rio.write_transform(da.rio.transform(), inplace=False)
        out = out.rio.write_nodata(np.nan, inplace=False)
    return out


def empty_raster_dataarray(
    da: xr.DataArray, *, x_name: str = "x", y_name: str = "y"
) -> xr.DataArray:
    return xr.DataArray(
        np.full((1, 1), np.nan, dtype=np.float32),
        coords={y_name: [0.0], x_name: [0.0]},
        dims=(y_name, x_name),
        attrs=dict(da.attrs),
    )


def write_manifest(df: pd.DataFrame, path) -> None:
    ensure_parent(path)
    GeothermalRegionManifestSchema.validate(df).to_csv(path, sep="\t", index=False)
