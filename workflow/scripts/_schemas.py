"""Schemas and empty output helpers."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.pandas as pa
import xarray as xr


def _valid_geometry(series):
    return series.notna() & ~series.is_empty & series.is_valid


class ShapeSchema:
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
        return cls.schema.validate(df)


class GeothermalRegionsSchema:
    schema = pa.DataFrameSchema(
        {
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
        {col: pa.Column(str, coerce=True) for col in ["kind", "depth_category", "bin_id", "price_bin", "path"]},
        strict=True,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        return cls.schema.validate(df)


class SensitivityMetricsSchema:
    schema = pa.DataFrameSchema(
        {
            "metric": pa.Column(str, coerce=True),
            "value": pa.Column(str, nullable=True, coerce=True),
        },
        strict=True,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        return cls.schema.validate(df)


class SensitivityBoxplotSchema:
    schema = pa.DataFrameSchema(
        {
            "scenario": pa.Column(str, coerce=True),
            "baseline": pa.Column(str, coerce=True),
            "row": pa.Column(int, coerce=True),
            "col": pa.Column(int, coerce=True),
            "tech": pa.Column(str, coerce=True),
            "lcoe_base": pa.Column(float, nullable=True, coerce=True),
            "lcoe_scen": pa.Column(float, nullable=True, coerce=True),
            "potential_base": pa.Column(float, nullable=True, coerce=True),
            "potential_scen": pa.Column(float, nullable=True, coerce=True),
            "pct_change_lcoe": pa.Column(float, nullable=True, coerce=True),
        },
        strict=True,
        coerce=True,
    )

    @classmethod
    def validate(cls, df):
        return cls.schema.validate(df)


GEOTHERMAL_REGION_COLUMNS = [
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
BOXPLOT_COLUMNS = [
    "scenario",
    "baseline",
    "row",
    "col",
    "tech",
    "lcoe_base",
    "lcoe_scen",
    "potential_base",
    "potential_scen",
    "pct_change_lcoe",
]


def validate_scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario" not in df.columns:
        raise ValueError("Scenario table must contain a 'scenario' column.")
    return df


def empty_geothermal_regions(crs=None) -> gpd.GeoDataFrame:
    df = pd.DataFrame(
        {
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


def empty_boxplot_table() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in BOXPLOT_COLUMNS})


def ensure_parent(path) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def write_empty_html(path) -> None:
    ensure_parent(path)
    Path(path).write_text("<!doctype html><html><body></body></html>\n")


def write_blank_png(path, figsize=(6, 4)) -> None:
    import matplotlib.pyplot as plt

    ensure_parent(path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    da: xr.DataArray,
    *,
    x_name: str = "x",
    y_name: str = "y",
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


def write_metrics(rows: list[tuple[str, object]], path) -> None:
    df = pd.DataFrame(rows, columns=["metric", "value"]).astype({"value": "object"})
    ensure_parent(path)
    SensitivityMetricsSchema.validate(df).to_csv(path, sep="\t", index=False)


def write_boxplot_table(df: pd.DataFrame, path) -> None:
    ensure_parent(path)
    df = SensitivityBoxplotSchema.validate(df)
    if str(path).endswith(".tsv"):
        df.to_csv(path, sep="\t", index=False)
    elif str(path).endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
