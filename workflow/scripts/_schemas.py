"""Schemas and empty output helpers."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pyproj import CRS

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


def validate_source_grid(source):
    """Check downloaded temperature coordinates before cropping or resampling."""
    CRS.from_user_input(source.attrs["crs"])
    for name in ("x", "y"):
        values = source[name].values
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError(
                f"Temperature coordinate {name!r} needs two finite centres."
            )
        steps = np.diff(values)
        if steps[0] == 0 or not np.allclose(steps, steps[0]):
            raise ValueError(
                f"Temperature coordinate {name!r} must be regularly spaced."
            )


def validate_temperature(source):
    """Check the downloaded temperature cube used for depth interpolation."""
    validate_source_grid(source)
    if set(source.temperature.dims) != {"depth_m", "y", "x"}:
        raise ValueError("Temperature must have depth_m, y and x dimensions.")
    depths = source.depth_m.values
    if not depths.size or not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("Temperature depths must be finite and strictly increasing.")


def validate_gst1(records):
    """Interpret downloaded latitude/longitude/thickness rows as a regular grid."""
    if records.ndim != 2 or records.shape[1] != 3 or not records.shape[0]:
        raise ValueError("GST-1 XYZ data must contain exactly three columns")
    if not np.isfinite(records).all() or (records[:, 2] < 0).any():
        raise ValueError("GST-1 records must be finite with nonnegative thickness")

    latitudes = np.unique(records[:, 0])
    longitudes = np.unique(records[:, 1])
    shape = (latitudes.size, longitudes.size)
    if records.shape[0] != latitudes.size * longitudes.size:
        raise ValueError("GST-1 rows do not form a complete rectangular grid")
    grid = records.reshape(*shape, 3)
    if not (
        np.all(grid[:, :, 0] == latitudes[:, None])
        and np.all(grid[:, :, 1] == longitudes[None, :])
    ):
        raise ValueError("GST-1 rows must be ordered by latitude then longitude")
    for coordinates in (latitudes, longitudes):
        steps = np.diff(coordinates)
        if not steps.size or not np.allclose(steps, steps[0]):
            raise ValueError(
                "GST-1 axes must have at least two regularly spaced centres"
            )
    return latitudes, longitudes, grid[:, :, 2]


def validate_area_potential(source):
    """Check supplied raster metadata required for pixel-area calculations."""
    transform = source.transform
    if (
        source.count != 1
        or source.crs != CRS.from_epsg(4326)
        or transform.b != 0
        or transform.d != 0
        or transform.a <= 0
        or transform.e >= 0
        or not np.isclose(transform.a, -transform.e)
    ):
        raise ValueError(
            "Area potential must be a single-band, square, north-up WGS84 raster "
            "containing available square metres per pixel."
        )


def validate_fractions(values):
    """Reject invalid supplied areas after normalization, allowing float32 roundoff."""
    if not np.isfinite(values).all() or ((values < 0) | (values > 1 + 1e-5)).any():
        raise ValueError("Area fractions must be finite and within [0,1].")


class GeothermalRegionsSchema:
    schema = pa.DataFrameSchema(
        {
            "region_id": pa.Column(str, coerce=True),
            "category": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["power", "heat", "contested"]))
            ),
            "carrier": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["power", "heat"]))
            ),
            "technology": pa.Column(str, coerce=True),
            "price_bin": pa.Column(str, coerce=True),
            "bin_left": pa.Column(float, nullable=True, coerce=True),
            "bin_right": pa.Column(float, nullable=True, coerce=True),
            "potential": pa.Column(float, nullable=True, coerce=True),
            "potential_unit": pa.Column(
                str, checks=pa.Check(lambda s: s.isin(["MW_e", "MWh_th/year"]))
            ),
            "mean_specific_capex_power": pa.Column(float, nullable=True, coerce=True),
            "mean_specific_capex_heat": pa.Column(float, nullable=True, coerce=True),
            "mean_network_distance_m": pa.Column(
                float, checks=pa.Check(lambda s: s >= 0), nullable=True, coerce=True
            ),
            "n_cells": pa.Column(int, coerce=True),
            "geometry": pa.Column(checks=pa.Check(_valid_geometry)),
        },
        strict=True,
        ordered=True,
        coerce=True,
    )


def empty_geothermal_regions(crs=None) -> gpd.GeoDataFrame:
    df = pd.DataFrame(
        {
            name: pd.Series(dtype=str(column.dtype))
            for name, column in GeothermalRegionsSchema.schema.columns.items()
            if name != "geometry"
        }
    )
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
