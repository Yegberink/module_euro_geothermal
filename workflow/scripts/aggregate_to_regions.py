#!/usr/bin/env python3
"""
Aggregate 2D geothermal LCOE and potential rasters to user-defined regions.

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
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gregor
import affine


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_2d_tif_name(path: str):
    """
    Parse a 2D output tif filename into (tech, depth_category, kind).

    Accepts both forms produced by make_2d.py:
      <tech>_2d_lcoe.tif                    → depth_category = "all_depths"
      <tech>_2d_potential.tif               → depth_category = "all_depths"
      <tech>_2d_<depthcat>_lcoe.tif         → depth_category = <depthcat>
      <tech>_2d_<depthcat>_potential.tif    → depth_category = <depthcat>
    """
    name = os.path.basename(path)

    # Depth-category form  (tech_2d_<cat>_<kind>.tif)
    m = re.match(
        r"^(?P<tech>[A-Za-z0-9]+)_2d_(?P<cat>.+)_(?P<kind>lcoe|potential)\.tif$",
        name,
    )
    if m:
        return m.group("tech").lower(), m.group("cat"), m.group("kind")

    # No-category form  (tech_2d_<kind>.tif)
    m = re.match(
        r"^(?P<tech>[A-Za-z0-9]+)_2d_(?P<kind>lcoe|potential)\.tif$",
        name,
    )
    if m:
        return m.group("tech").lower(), "all_depths", m.group("kind")

    raise ValueError(f"Unexpected tif filename: {name!r}")


def _safe_name(s: str) -> str:
    """Sanitise a string for use in a file name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    return s[:200]


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------

def ensure_north_up(da: xr.DataArray) -> xr.DataArray:
    """
    Guarantee that the raster is north-up (negative pixel height).
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
    try:
        nd = da.rio.nodata
        if nd is not None:
            da2 = da2.rio.write_nodata(nd, inplace=False)
    except Exception:
        pass

    return da2


def inherit_rio_meta(dst: xr.DataArray, src: xr.DataArray) -> xr.DataArray:
    """Copy CRS, transform, and nodata from *src* to *dst*."""
    dst = dst.rio.write_crs(src.rio.crs, inplace=False)
    dst = dst.rio.write_transform(src.rio.transform(), inplace=False)
    try:
        nd = src.rio.nodata
        if nd is not None:
            dst = dst.rio.write_nodata(nd, inplace=False)
    except Exception:
        pass
    return dst


def load_aligned_rasters(lcoe_path: str, pot_path: str):
    """
    Open LCOE and potential rasters, verify alignment, and return as float64 DataArrays.
    """
    lcoe = rxr.open_rasterio(lcoe_path).squeeze(drop=True).astype("float64")
    pot  = rxr.open_rasterio(pot_path).squeeze(drop=True).astype("float64")

    if (
        lcoe.rio.crs != pot.rio.crs
        or lcoe.rio.transform() != pot.rio.transform()
        or lcoe.shape != pot.shape
    ):
        raise ValueError(
            f"Potential raster is not aligned with LCOE raster "
            f"(CRS/transform/shape mismatch).\n  lcoe: {lcoe_path}\n  pot: {pot_path}"
        )

    return lcoe, pot


# ---------------------------------------------------------------------------
# Bin computation
# ---------------------------------------------------------------------------

def parse_fixed_bins(raw):
    """
    Parse fixed_bins from config into a sorted list of (left, right) float tuples.
    Accepts either ["0-50", "50-100"] or [[0, 50], [50, 100]].
    """
    out = []
    for b in (raw or []):
        if isinstance(b, str) and "-" in b:
            a, c = b.split("-", 1)
            out.append((float(a), float(c)))
        else:
            out.append((float(b[0]), float(b[1])))
    out.sort(key=lambda x: x[0])
    return out


def compute_quantile_bins(lcoe_da: xr.DataArray, n_quantiles: int):
    """
    Compute *n_quantiles* global quantile bins from finite LCOE values.
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
    raise ValueError(f"Unknown method_region_aggregation: {method!r}")


# ---------------------------------------------------------------------------
# Choropleth helper
# ---------------------------------------------------------------------------

def choropleth_png(gdf: gpd.GeoDataFrame, column: str, path: str, title: str):
    """Write a simple choropleth PNG; silently skip if data are missing."""
    if gdf is None or len(gdf) == 0 or column not in gdf.columns:
        return
    if gdf[column].notna().sum() == 0:
        return

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        gdf.plot(column=column, ax=ax, legend=True)
        gdf.geometry.boundary.plot(ax=ax, linewidth=0.5, color="black")
        ax.set_title(title)
        ax.set_axis_off()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


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
    figdir: str,
    lcoe_cutoff: float | None,
    depth_category: str = "all_depths",
) -> tuple[pd.DataFrame, object]:
    """
    Aggregate LCOE and potential for a single (tech, depth_category) pair.

    Returns
    -------
    df : pd.DataFrame
        Rows of (technology, depth_category, region_id, price_bin, …).
    crs : CRS
        CRS of the input rasters (for writing the output GeoDataFrame).
    """
    lcoe, pot = load_aligned_rasters(lcoe_path, pot_path)
    lcoe = ensure_north_up(lcoe)
    pot  = ensure_north_up(pot)

    regions = regions_gdf.to_crs(lcoe.rio.crs)
    bins = _bins_for_method(method, lcoe, n_quantiles, fixed_bins, lcoe_cutoff)

    rows = []

    for bi, (bin_left, bin_right) in enumerate(bins):

        # Build the per-bin validity mask
        if method == "cutoff":
            in_bin      = lcoe <= bin_right
            bin_label   = f"<= {bin_right:g}"
            bin_left_v  = np.nan
            bin_right_v = float(bin_right)
        else:
            # Last bin is inclusive on the right edge
            if bi < len(bins) - 1:
                in_bin = (lcoe >= bin_left) & (lcoe < bin_right)
            else:
                in_bin = (lcoe >= bin_left) & (lcoe <= bin_right)
            bin_label   = f"{bin_left:g}-{bin_right:g}"
            bin_left_v  = float(bin_left)
            bin_right_v = float(bin_right)

        valid = (
            in_bin
            & xr.ufuncs.isfinite(lcoe)
            & xr.ufuncs.isfinite(pot)
            & (pot > 0)
        )

        # Fast skip when no valid cells exist
        try:
            n_valid = int(valid.sum().values)
        except Exception:
            n_valid = 0
        if n_valid == 0:
            continue

        # Weighted-sum rasters (potential, lcoe*potential, cell count)
        pot_bin = inherit_rio_meta(pot.where(valid, other=0.0), pot)
        lp_bin  = inherit_rio_meta((lcoe * pot).where(valid, other=0.0), pot)
        cnt_bin = inherit_rio_meta(xr.where(valid, 1.0, 0.0), pot)

        agg_p  = gregor.aggregate.aggregate_raster_to_polygon(pot_bin,  regions, stats="sum")
        agg_lp = gregor.aggregate.aggregate_raster_to_polygon(lp_bin,   regions, stats="sum")
        agg_n  = gregor.aggregate.aggregate_raster_to_polygon(cnt_bin,  regions, stats="sum")

        if agg_p is None or len(agg_p) == 0 or "sum" not in agg_p.columns:
            continue

        total_pot = agg_p["sum"].astype(float)
        if (total_pot.fillna(0.0) <= 0.0).all():
            continue

        mean_lcoe = agg_lp["sum"].astype(float) / total_pot.replace(0.0, np.nan)

        choropleth_png(
            agg_p,
            column="sum",
            path=os.path.join(
                figdir,
                f"{tech}_{_safe_name(depth_category)}_potential_bin_{_safe_name(bin_label)}.png",
            ),
            title=f"{tech} ({depth_category}): potential per region — LCOE bin {bin_label}",
        )

        df = pd.DataFrame(
            {
                "technology":    tech,
                "depth_category": str(depth_category),
                "region_id":     agg_p.index.astype(str),
                "price_bin":     bin_label,
                "bin_left":      bin_left_v,
                "bin_right":     bin_right_v,
                "potential":     total_pot.values,
                "mean_lcoe":     mean_lcoe.values,
                "n_cells":       agg_n["sum"].fillna(0).round().astype(np.int64).values,
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
                "technology", "depth_category", "region_id", "price_bin",
                "bin_left", "bin_right", "potential", "mean_lcoe", "n_cells",
            ]
        )

    return out_df, lcoe.rio.crs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Inputs / outputs from Snakemake ---------------------------------
    lcoe_tifs  = list(snakemake.input.lcoe_tifs)
    pot_tifs   = list(snakemake.input.pot_tifs)
    shapes_path = snakemake.input["shapes"]
    out_path   = snakemake.output["geoparquet"]
    figdir     = snakemake.output["figdir"]

    # --- Parameters -------------------------------------------------------
    cycles     = [str(c).strip().lower() for c in (snakemake.params.get("cycles") or [])]
    method     = str(snakemake.params["method_region_aggregation"]).lower()
    n_quantiles = int(snakemake.params.get("n_quantiles", 5))
    fixed_bins = parse_fixed_bins(snakemake.params.get("fixed_bins", []))
    lcoe_cutoff = snakemake.params.get("lcoe_cutoff", None)
    if lcoe_cutoff is not None:
        lcoe_cutoff = float(lcoe_cutoff)

    # Validate method-specific requirements up front
    if method == "fixed_bins" and not fixed_bins:
        raise ValueError("fixed_bins required when method_region_aggregation='fixed_bins'")
    if method == "cutoff" and lcoe_cutoff is None:
        raise ValueError("lcoe_cutoff required when method_region_aggregation='cutoff'")

    # --- Build (tech, depth_category) → path maps -------------------------
    lcoe_map: dict[tuple, str] = {}
    pot_map:  dict[tuple, str] = {}

    for p in lcoe_tifs:
        tech, cat, kind = _parse_2d_tif_name(p)
        assert kind == "lcoe", f"Expected lcoe tif, got: {p}"
        lcoe_map[(tech, cat)] = p

    for p in pot_tifs:
        tech, cat, kind = _parse_2d_tif_name(p)
        assert kind == "potential", f"Expected potential tif, got: {p}"
        pot_map[(tech, cat)] = p

    # Optionally filter to the cycles listed in params
    if cycles:
        lcoe_map = {k: v for k, v in lcoe_map.items() if k[0] in cycles}
        pot_map  = {k: v for k, v in pot_map.items()  if k[0] in cycles}

    pairs = sorted(set(lcoe_map) & set(pot_map))
    if not pairs:
        raise ValueError(
            "No matching (tech, depth_category) pairs found between lcoe_tifs and pot_tifs.\n"
            f"  lcoe keys : {sorted(lcoe_map)}\n"
            f"  pot  keys : {sorted(pot_map)}"
        )

    # --- Load and prepare regions -----------------------------------------
    os.makedirs(figdir, exist_ok=True)

    gdf = gpd.read_parquet(shapes_path)
    if "shape_class" in gdf.columns:
        gdf = gdf[gdf["shape_class"] != "maritime"].copy()

    regions = dissolve_regions(gdf, region_id_col="shape_id")

    # --- Aggregate each (tech, depth_category) pair -----------------------
    frames  = []
    out_crs = None

    for tech, depth_cat in pairs:
        df, crs = aggregate_one_tech(
            tech           = tech,
            lcoe_path      = lcoe_map[(tech, depth_cat)],
            pot_path       = pot_map[(tech, depth_cat)],
            regions_gdf    = regions,
            method         = method,
            n_quantiles    = n_quantiles,
            fixed_bins     = fixed_bins,
            figdir         = figdir,
            lcoe_cutoff    = lcoe_cutoff,
            depth_category = depth_cat,
        )
        if not df.empty:
            frames.append(df)
        out_crs = out_crs or crs

    # --- Assemble output GeoDataFrame -------------------------------------
    _EMPTY_COLS = [
        "technology", "depth_category", "region_id", "price_bin",
        "bin_left", "bin_right", "potential", "mean_lcoe", "n_cells",
    ]

    regions2 = regions.copy()
    regions2.index = regions2.index.astype(str)
    geom_col = regions2.geometry.name

    if not frames:
        if method == "cutoff":
            raise ValueError(
                f"No geothermal potential passed the LCOE cutoff "
                f"(lcoe_cutoff={lcoe_cutoff})."
            )
        if method == "fixed_bins":
            raise ValueError(
                f"No geothermal potential fell into any fixed bin "
                f"(fixed_bins={fixed_bins})."
            )
        raise ValueError(
            "Region aggregation produced an empty result. "
            "No raster cells overlapped the region shapes or passed filtering."
        )

    out_df = pd.concat(frames, ignore_index=True)
    out_gdf = gpd.GeoDataFrame(
        out_df.join(regions2[[geom_col]], on="region_id"),
        geometry=geom_col,
        crs=regions2.crs,
    )

    out_gdf.to_parquet(out_path, index=False)


if "snakemake" in dir():
    main()
