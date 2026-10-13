#!/usr/bin/env python3
"""Aggregate 2D geothermal LCOE and potential rasters to user-defined regions.

Consumes outputs of make_2d.py (one or more LCOE + potential GeoTIFF pairs)
and produces a GeoParquet with per-region potential and mean-LCOE statistics,
grouped by technology, depth category, and price bin.

Supported aggregation methods (method_region_aggregation):
  - quantiles    : bin LCOE into N global quantile bins
  - fixed_bins   : bin LCOE into user-supplied (left, right) intervals
  - cutoff       : single bin containing all cells with LCOE <= lcoe_cutoff
"""

import os
import re
import sys

import _schemas
import affine
import geopandas as gpd
import gregor
import numpy as np
import pandas as pd
import xarray as xr
from _heat import align_to_grid, check_grid, read_raster

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def write_empty_outputs(out_path: str, manifest_path: str, crs=None):
    """Write schema-valid empty regional outputs."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_gdf = _schemas.GeothermalRegionsSchema.validate(
        _schemas.empty_geothermal_regions(crs)
    )
    out_gdf.to_parquet(out_path, index=False)
    _schemas.write_manifest(
        pd.DataFrame(
            [
                {
                    "kind": "combined_parquet",
                    "depth_category": "all",
                    "bin_id": "all",
                    "price_bin": "all",
                    "path": out_path,
                }
            ]
        ),
        manifest_path,
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _parse_2d_tif_name(path: str):
    """Parse a 2D output tif filename into (tech, depth_category, kind).

    Accepts both forms produced by make_2d.py:
      <tech>_2d_lcoe.tif                    → depth_category = "all_depths"
      <tech>_2d_potential.tif               → depth_category = "all_depths"
      <tech>_2d_<depthcat>_lcoe.tif         → depth_category = <depthcat>
      <tech>_2d_<depthcat>_potential.tif    → depth_category = <depthcat>
    """
    name = os.path.basename(path)

    # Depth-category form  (tech_2d_<cat>_<kind>.tif)
    m = re.match(
        r"^(?P<tech>[A-Za-z0-9]+)_2d_(?P<cat>.+)_(?P<kind>lcoe|potential)\.tif$", name
    )
    if m:
        return m.group("tech").lower(), m.group("cat"), m.group("kind")

    # No-category form  (tech_2d_<kind>.tif)
    m = re.match(r"^(?P<tech>[A-Za-z0-9]+)_2d_(?P<kind>lcoe|potential)\.tif$", name)
    if m:
        return m.group("tech").lower(), "all_depths", m.group("kind")

    raise ValueError(f"Unrecognized 2-D raster filename: {path}")


def _safe_name(s: str) -> str:
    """Sanitise a string for use in a file name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    return s[:200]


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------


def ensure_north_up(da: xr.DataArray) -> xr.DataArray:
    """Guarantee that the raster is north-up (negative pixel height).

    Flips the y dimension and rewrites the affine transform when needed.
    """
    t = da.rio.transform()
    if t.e < 0:
        return da  # already north-up

    height = da.sizes["y"]
    da2 = da.isel(y=slice(None, None, -1))

    # Recompute top-left y origin (use height-1 to stay on the last pixel centre)
    y_top = t.f + t.e * (height - 1)
    t2 = affine.Affine(t.a, t.b, t.c, t.d, -abs(t.e), y_top)

    da2 = da2.rio.write_transform(t2, inplace=False)
    da2 = da2.rio.write_crs(da.rio.crs, inplace=False)
    nd = da.rio.nodata
    if nd is not None:
        da2 = da2.rio.write_nodata(nd, inplace=False)

    return da2


def inherit_rio_meta(dst: xr.DataArray, src: xr.DataArray) -> xr.DataArray:
    """Copy CRS, transform, and nodata from *src* to *dst*."""
    dst = dst.rio.write_crs(src.rio.crs, inplace=False)
    dst = dst.rio.write_transform(src.rio.transform(), inplace=False)
    nd = src.rio.nodata
    if nd is not None:
        dst = dst.rio.write_nodata(nd, inplace=False)
    return dst


def load_aligned_rasters(lcoe_path: str, pot_path: str):
    """Open LCOE and potential rasters, verify alignment, and return as float64 DataArrays."""
    lcoe, pot = read_raster(lcoe_path), read_raster(pot_path)
    check_grid(lcoe, pot)

    return lcoe, pot


# ---------------------------------------------------------------------------
# Bin computation
# ---------------------------------------------------------------------------


def parse_fixed_bins(raw):
    """Parse fixed_bins from config into a sorted list of (left, right) float tuples.

    Accepts either ["0-50", "50-100"] or [[0, 50], [50, 100]].
    """
    out = []
    for b in raw or []:
        if isinstance(b, str) and "-" in b:
            a, c = b.split("-", 1)
            out.append((float(a), float(c)))
        else:
            out.append((float(b[0]), float(b[1])))
    out.sort(key=lambda x: x[0])
    return out


def compute_quantile_bins(lcoe_da: xr.DataArray, n_quantiles: int):
    """Compute *n_quantiles* global quantile bins from finite LCOE values.

    Returns a list of (left, right) float tuples.
    """
    vals = lcoe_da.values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []

    edges = np.nanquantile(vals, np.linspace(0, 1, n_quantiles + 1))
    edges = np.maximum.accumulate(edges)  # ensure monotone

    if np.allclose(edges[0], edges[-1]):
        edges = np.array([edges[0], edges[0] + 1e-6])

    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def _bins_for_method(method, lcoe_da, n_quantiles, fixed_bins, lcoe_cutoff):
    """Return the list of (left, right) bins for the chosen method."""
    if method == "quantiles":
        return compute_quantile_bins(lcoe_da, n_quantiles=n_quantiles)
    if method == "fixed_bins":
        return list(fixed_bins)
    if method == "cutoff":
        return [(np.nan, float(lcoe_cutoff))]
    return []


# ---------------------------------------------------------------------------
# Region dissolution
# ---------------------------------------------------------------------------


def dissolve_regions(gdf: gpd.GeoDataFrame, region_id_col: str) -> gpd.GeoDataFrame:
    """Dissolve all geometries to unique region IDs (removes sub-polygon detail)."""
    dissolved = (
        gdf[[region_id_col, gdf.geometry.name]]
        .dissolve(by=region_id_col, as_index=False)
        .set_index(region_id_col)
    )
    return dissolved


# ---------------------------------------------------------------------------
# Core aggregation for one (tech, depth_category) pair
# ---------------------------------------------------------------------------


def aggregate_one_tech(
    tech: str,
    lcoe_path: str,
    pot_path: str,
    regions_gdf: gpd.GeoDataFrame,
    method: str,
    n_quantiles: int,
    fixed_bins: list,
    lcoe_cutoff: float | None,
    depth_category: str = "all_depths",
    used_fraction=None,
) -> tuple[pd.DataFrame, object]:
    """Aggregate LCOE and potential for a single (tech, depth_category) pair.

    Returns:
    -------
    df : pd.DataFrame
        Rows of (technology, depth_category, region_id, price_bin, …).
    crs : CRS
        CRS of the input rasters (for writing the output GeoDataFrame).
    """
    lcoe, pot = load_aligned_rasters(lcoe_path, pot_path)
    lcoe = ensure_north_up(lcoe)
    pot = ensure_north_up(pot)

    fraction = (
        align_to_grid(used_fraction, pot).clip(0, 1)
        if used_fraction is not None
        else xr.zeros_like(pot)
    )
    regions = regions_gdf.to_crs(lcoe.rio.crs)
    bins = _bins_for_method(method, lcoe, n_quantiles, fixed_bins, lcoe_cutoff)

    rows = []
    for bi, (bin_left, bin_right) in enumerate(bins):
        # Build the per-bin validity mask
        if method == "cutoff":
            in_bin = lcoe <= bin_right
            bin_label = f"<= {bin_right:g}"
            bin_id = f"cutoff_{bin_right:g}"
            bin_left_v = np.nan
            bin_right_v = float(bin_right)
        else:
            # Last bin is inclusive on the right edge
            if bi < len(bins) - 1:
                in_bin = (lcoe >= bin_left) & (lcoe < bin_right)
            else:
                in_bin = (lcoe >= bin_left) & (lcoe <= bin_right)
            bin_label = f"{bin_left:g}-{bin_right:g}"
            bin_id = (
                f"quantile_{bi + 1:02d}"
                if method == "quantiles"
                else f"bin_{bin_label}"
            )
            bin_left_v = float(bin_left)
            bin_right_v = float(bin_right)

        valid = in_bin & xr.ufuncs.isfinite(lcoe) & xr.ufuncs.isfinite(pot) & (pot > 0)

        original_valid = valid
        categories = [("power", 1 - fraction)]
        if used_fraction is not None:
            categories.append(("contested", fraction))
        for category, share in categories:
            category_pot = pot * share
            valid = original_valid & (category_pot > 0)
            # Fast skip when no valid cells exist
            n_valid = int(valid.sum().values)
            if n_valid == 0:
                continue

            # Weighted-sum rasters (potential, lcoe*potential, cell count)
            pot_bin = inherit_rio_meta(category_pot.where(valid, other=0.0), pot)
            lp_bin = inherit_rio_meta(
                (lcoe * category_pot).where(valid, other=0.0), pot
            )
            cnt_bin = inherit_rio_meta(xr.where(valid, 1.0, 0.0), pot)

            agg_p = gregor.aggregate.aggregate_raster_to_polygon(
                pot_bin, regions, stats="sum"
            )
            agg_lp = gregor.aggregate.aggregate_raster_to_polygon(
                lp_bin, regions, stats="sum"
            )
            agg_n = gregor.aggregate.aggregate_raster_to_polygon(
                cnt_bin, regions, stats="sum"
            )

            if agg_p is None or len(agg_p) == 0 or "sum" not in agg_p.columns:
                continue

            total_pot = agg_p["sum"].astype(float)
            if (total_pot.fillna(0.0) <= 0.0).all():
                continue

            mean_lcoe = agg_lp["sum"].astype(float) / total_pot.replace(0.0, np.nan)

            df = pd.DataFrame(
                {
                    "category": category,
                    "carrier": "power",
                    "potential_unit": "MW_e",
                    "mean_lcoh": np.nan,
                    "technology": tech,
                    "depth_category": str(depth_category),
                    "bin_id": bin_id,
                    "region_id": agg_p.index.astype(str),
                    "price_bin": bin_label,
                    "bin_left": bin_left_v,
                    "bin_right": bin_right_v,
                    "potential": total_pot.values,
                    "mean_lcoe": mean_lcoe.values,
                    "n_cells": agg_n["sum"].fillna(0).round().astype(np.int64).values,
                }
            )
            df = df[np.isfinite(df["potential"]) & (df["potential"] > 0)]
            if not df.empty:
                rows.append(df)

    if rows:
        out_df = pd.concat(rows, ignore_index=True)
    else:
        out_df = pd.DataFrame(
            columns=[
                "technology",
                "depth_category",
                "bin_id",
                "region_id",
                "price_bin",
                "bin_left",
                "bin_right",
                "potential",
                "mean_lcoe",
                "n_cells",
            ]
        )

    return out_df, lcoe.rio.crs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def power_eligibility(costs, capacities, method, n_quantiles, fixed_bins, cutoff):
    """Identify retained power before reservation, using the original bin edges."""
    bins = _bins_for_method(method, costs, n_quantiles, fixed_bins, cutoff)
    eligible = xr.zeros_like(costs, dtype=bool)
    for index, (left, right) in enumerate(bins):
        if method == "cutoff":
            inside = costs <= right
        elif index == len(bins) - 1:
            inside = (costs >= left) & (costs <= right)
        else:
            inside = (costs >= left) & (costs < right)
        eligible |= inside
    return eligible & np.isfinite(costs) & np.isfinite(capacities) & (capacities > 0)


def aggregate_heat(ledger, eligible, region_ids, combined):
    """Attribute allocated heat to consumers, without duplicating power options."""
    ledger = ledger.copy()
    ledger["category"] = "heat"
    if combined and len(ledger):
        contested = eligible.values[ledger.row.to_numpy(), ledger.col.to_numpy()]
        ledger.loc[contested, "category"] = "contested"
    ledger["annual_cost"] = ledger.heat_mwh_per_year * ledger.lcoh_eur_per_mwh
    grouped = ledger.groupby(["region_id", "category"]).agg(
        potential=("heat_mwh_per_year", "sum"),
        annual_cost=("annual_cost", "sum"),
        n_cells=("cell_id", "nunique"),
    )
    categories = ["heat", "contested"] if combined else ["heat"]
    index = pd.MultiIndex.from_product(
        [region_ids.astype(str), categories], names=["region_id", "category"]
    )
    grouped = grouped.reindex(index, fill_value=0).reset_index()
    grouped["mean_lcoh"] = grouped.annual_cost / grouped.potential.replace(0, np.nan)
    grouped = grouped.drop(columns="annual_cost")
    grouped["carrier"] = "heat"
    grouped["potential_unit"] = "MWh_th/year"
    grouped["technology"] = "geothermal_heat"
    grouped["depth_category"] = "target_temperature"
    grouped["price_bin"] = "all"
    grouped["bin_id"] = "all"
    grouped["bin_left"] = np.nan
    grouped["bin_right"] = np.nan
    grouped["mean_lcoe"] = np.nan
    return grouped


def main():
    """Aggregate all configured technology rasters to the supplied regions."""
    # --- Inputs / outputs from Snakemake ---------------------------------
    lcoe_tifs = list(snakemake.input.lcoe_tifs)
    pot_tifs = list(snakemake.input.pot_tifs)
    shapes_path = snakemake.input["shapes"]
    out_path = snakemake.output["geoparquet"]
    manifest_path = snakemake.output["manifest"]
    resultdir = os.path.dirname(out_path)
    heat_enabled = snakemake.params.get("carrier_out", "power") != "power"
    used_fraction = (
        read_raster(snakemake.input.heat_fraction[0]) if heat_enabled else None
    )
    heat_eligible = xr.zeros_like(used_fraction, dtype=bool) if heat_enabled else None

    # --- Parameters -------------------------------------------------------
    cycles = [str(c).strip().lower() for c in (snakemake.params.get("cycles") or [])]
    method = str(snakemake.params["method_region_aggregation"]).lower()
    n_quantiles = (
        int(snakemake.params["n_quantiles"]) if cycles and method == "quantiles" else None
    )
    fixed_bins = parse_fixed_bins(snakemake.params.get("fixed_bins", []))
    lcoe_cutoff = snakemake.params.get("lcoe_cutoff", None)
    if lcoe_cutoff is not None:
        lcoe_cutoff = float(lcoe_cutoff)

    # --- Build (tech, depth_category) → path maps -------------------------
    lcoe_map: dict[tuple, str] = {}
    pot_map: dict[tuple, str] = {}

    for p in lcoe_tifs:
        tech, cat, _ = _parse_2d_tif_name(p)
        lcoe_map[(tech, cat)] = p

    for p in pot_tifs:
        tech, cat, _ = _parse_2d_tif_name(p)
        pot_map[(tech, cat)] = p

    # Optionally filter to the cycles listed in params
    if cycles:
        lcoe_map = {k: v for k, v in lcoe_map.items() if k[0] in cycles}
        pot_map = {k: v for k, v in pot_map.items() if k[0] in cycles}

    pairs = sorted(set(lcoe_map) & set(pot_map))
    # --- Load and prepare regions -----------------------------------------

    gdf = _schemas.ShapeSchema.validate(gpd.read_parquet(shapes_path))
    if "shape_class" in gdf.columns:
        gdf = gdf[gdf["shape_class"] != "maritime"].copy()
    if gdf.empty:
        write_empty_outputs(out_path, manifest_path, gdf.crs)
        return

    regions = dissolve_regions(gdf, region_id_col="shape_id")
    if regions.empty:
        write_empty_outputs(out_path, manifest_path, gdf.crs)
        return

    # --- Aggregate each (tech, depth_category) pair -----------------------
    frames = []
    for tech, depth_cat in pairs:
        if heat_enabled:
            costs, capacities = load_aligned_rasters(
                lcoe_map[(tech, depth_cat)], pot_map[(tech, depth_cat)]
            )
            eligible = power_eligibility(
                costs, capacities, method, n_quantiles, fixed_bins, lcoe_cutoff
            )
            heat_eligible = heat_eligible | align_to_grid(
                eligible, used_fraction, fill=False
            )
        df, _ = aggregate_one_tech(
            tech=tech,
            lcoe_path=lcoe_map[(tech, depth_cat)],
            pot_path=pot_map[(tech, depth_cat)],
            regions_gdf=regions,
            method=method,
            n_quantiles=n_quantiles,
            fixed_bins=fixed_bins,
            lcoe_cutoff=lcoe_cutoff,
            depth_category=depth_cat,
            used_fraction=used_fraction,
        )
        if not df.empty:
            frames.append(df)

    if heat_enabled:
        ledger = pd.read_parquet(snakemake.input.heat_allocations[0])
        frames.append(
            aggregate_heat(
                ledger,
                heat_eligible,
                regions.index,
                snakemake.params["carrier_out"] == "heat_and_power",
            )
        )

    # --- Assemble output GeoDataFrame -------------------------------------
    regions2 = regions.copy()
    regions2.index = regions2.index.astype(str)
    geom_col = regions2.geometry.name

    if frames:
        out_df = pd.concat(frames, ignore_index=True)
        out_gdf = gpd.GeoDataFrame(
            out_df.join(regions2[[geom_col]], on="region_id"),
            geometry=geom_col,
            crs=regions2.crs,
        )
    else:
        write_empty_outputs(out_path, manifest_path, regions2.crs)
        return

    os.makedirs(resultdir, exist_ok=True)
    out_gdf = _schemas.GeothermalRegionsSchema.validate(out_gdf)
    out_gdf.to_parquet(out_path, index=False)
    output_rows = [
        {
            "kind": "combined_parquet",
            "depth_category": "all",
            "bin_id": "all",
            "price_bin": "all",
            "path": out_path,
        }
    ]

    for (depth_category, bin_id), group in out_gdf.groupby(
        ["depth_category", "bin_id"], dropna=False
    ):
        price_bin = ", ".join(sorted(group["price_bin"].astype(str).unique()))
        split_path = os.path.join(
            resultdir,
            f"geothermal_regions_{_safe_name(depth_category)}_{_safe_name(bin_id)}.parquet",
        )
        group = _schemas.GeothermalRegionsSchema.validate(group)
        group.to_parquet(split_path, index=False)
        output_rows.append(
            {
                "kind": "split_parquet",
                "depth_category": depth_category,
                "bin_id": bin_id,
                "price_bin": price_bin,
                "path": split_path,
            }
        )

    _schemas.write_manifest(pd.DataFrame(output_rows), manifest_path)


if "snakemake" in dir():
    main()
