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


class SourceGridSchema:
    """Coordinate contract for the regular subsurface grid."""

    @staticmethod
    def validate(source):
        """Require at least two finite, regularly spaced centres per axis."""
        if not source.attrs.get("crs"):
            raise ValueError("Temperature grid must declare its CRS.")
        CRS.from_user_input(source.attrs["crs"])
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
