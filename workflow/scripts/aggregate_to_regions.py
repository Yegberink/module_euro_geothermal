#!/usr/bin/env python3
"""Aggregate 2D geothermal LCOE and potential rasters to user-defined regions.

Consumes power LCOE, potential and specific CAPEX rasters, and allocated heat,
and produces a GeoParquet with per-region potential and mean specific CAPEX,
grouped by technology, category, and price bin.

Supported aggregation methods (method_region_aggregation):
  - quantiles    : bin LCOE into N global quantile bins
  - fixed_bins   : bin LCOE into user-supplied (left, right) intervals
  - cutoff       : single bin containing all cells with LCOE <= lcoe_cutoff
"""

import sys

import _plots
import _schemas
import geopandas as gpd
import gregor
import numpy as np
import pandas as pd
import xarray as xr
from _heat import align_to_grid, check_grid, read_raster, write_raster

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def inherit_rio_meta(dst: xr.DataArray, src: xr.DataArray) -> xr.DataArray:
    """Copy CRS, transform, and nodata from *src* to *dst*."""
    dst = dst.rio.write_crs(src.rio.crs, inplace=False)
    dst = dst.rio.write_transform(src.rio.transform(), inplace=False)
    nodata = src.rio.nodata
    if nodata is not None:
        dst = dst.rio.write_nodata(nodata, inplace=False)
    return dst


# ---------------------------------------------------------------------------
# Bin computation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Electricity aggregation
# ---------------------------------------------------------------------------


def aggregate_one_tech(
    tech: str,
    lcoe: xr.DataArray,
    pot: xr.DataArray,
    specific_capex: xr.DataArray,
    regions_gdf: gpd.GeoDataFrame,
    method: str,
    n_quantiles: int,
    fixed_bins: list,
    lcoe_cutoff: float | None,
    used_fraction=None,
    power_path=None,
    contested_path=None,
) -> tuple[pd.DataFrame, object]:
    """Write eligible power alternatives and aggregate each price bin."""
    check_grid(lcoe, pot)
    check_grid(specific_capex, pot)
    fraction = (
        align_to_grid(used_fraction, pot).clip(0, 1)
        if used_fraction is not None
        else xr.zeros_like(pot)
    )
    regions = regions_gdf.to_crs(lcoe.rio.crs)
    if method == "quantiles":
        bins = compute_quantile_bins(lcoe, n_quantiles)
    elif method == "fixed_bins":
        bins = fixed_bins
    elif method == "cutoff":
        bins = [(np.nan, float(lcoe_cutoff))]
    else:
        bins = []

    eligible = power_eligibility(lcoe, pot, bins, method)
    feasible = np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)
    # Price exclusions are zero potential; physically infeasible cells are nodata.
    retained = pot.where(eligible, 0).where(feasible)
    if power_path is not None:
        write_raster(retained * (1 - fraction), power_path, "MW_e")
    if contested_path is not None:
        write_raster(retained * fraction, contested_path, "MW_e")

    rows = []
    if regions.empty:
        return pd.DataFrame(), eligible
    for bi, (bin_left, bin_right) in enumerate(bins):
        # Build the per-bin validity mask
        if method == "cutoff":
            in_bin = lcoe <= bin_right
            bin_label = f"<= {bin_right:g}"
            bin_left_v = np.nan
            bin_right_v = float(bin_right)
        else:
            # Last bin is inclusive on the right edge
            if bi < len(bins) - 1:
                in_bin = (lcoe >= bin_left) & (lcoe < bin_right)
            else:
                in_bin = (lcoe >= bin_left) & (lcoe <= bin_right)
            bin_label = f"{bin_left:g}-{bin_right:g}"
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

            # Weight specific CAPEX by the capacity assigned to each category.
            pot_bin = inherit_rio_meta(category_pot.where(valid, other=0.0), pot)
            capex_bin = inherit_rio_meta(
                (specific_capex * category_pot).where(valid, other=0.0), pot
            )
            cnt_bin = inherit_rio_meta(xr.where(valid, 1.0, 0.0), pot)

            agg_p = gregor.aggregate.aggregate_raster_to_polygon(
                pot_bin, regions, stats="sum"
            )
            agg_capex = gregor.aggregate.aggregate_raster_to_polygon(
                capex_bin, regions, stats="sum"
            )
            agg_n = gregor.aggregate.aggregate_raster_to_polygon(
                cnt_bin, regions, stats="sum"
            )

            if agg_p is None or len(agg_p) == 0 or "sum" not in agg_p.columns:
                continue

            total_pot = agg_p["sum"].astype(float)
            if (total_pot.fillna(0.0) <= 0.0).all():
                continue

            mean_specific_capex_power = agg_capex["sum"].astype(
                float
            ) / total_pot.replace(0.0, np.nan)

            df = pd.DataFrame(
                {
                    "category": category,
                    "carrier": "power",
                    "potential_unit": "MW_e",
                    "mean_specific_capex_heat": np.nan,
                    "mean_network_distance_m": np.nan,
                    "technology": tech,
                    "region_id": agg_p.index.astype(str),
                    "price_bin": bin_label,
                    "bin_left": bin_left_v,
                    "bin_right": bin_right_v,
                    "potential": total_pot.values,
                    "mean_specific_capex_power": mean_specific_capex_power.values,
                    "n_cells": agg_n["sum"].fillna(0).round().astype(np.int64).values,
                }
            )
            df = df[np.isfinite(df["potential"]) & (df["potential"] > 0)]
            if not df.empty:
                rows.append(df)

    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), eligible


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def power_eligibility(costs, capacities, bins, method):
    """Identify retained power before reservation, using the original bin edges."""
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


def aggregate_heat(ledger, eligible, region_ids, combined, specific_capex):
    """Attribute allocated heat to consumers, without duplicating power options."""
    ledger = ledger.copy()
    ledger["category"] = "heat"
    if combined and len(ledger):
        contested = eligible.values[ledger.row.to_numpy(), ledger.col.to_numpy()]
        ledger.loc[contested, "category"] = "contested"
    # A common heat capacity factor makes supplied-energy weighting equivalent
    # to weighting by the thermal capacity developed for each consumer.
    ledger["weighted_capex"] = (
        ledger.heat_mwh_per_year
        * specific_capex.values[ledger.row.to_numpy(), ledger.col.to_numpy()]
    )
    ledger["heat_distance"] = ledger.heat_mwh_per_year * ledger.network_distance_m
    grouped = ledger.groupby(["region_id", "category"]).agg(
        potential=("heat_mwh_per_year", "sum"),
        weighted_capex=("weighted_capex", "sum"),
        heat_distance=("heat_distance", "sum"),
        n_cells=("cell_id", "nunique"),
    )
    categories = ["heat", "contested"] if combined else ["heat"]
    index = pd.MultiIndex.from_product(
        [region_ids.astype(str), categories], names=["region_id", "category"]
    )
    grouped = grouped.reindex(index, fill_value=0).reset_index()
    grouped["mean_specific_capex_heat"] = (
        grouped.weighted_capex / grouped.potential.replace(0, np.nan)
    )
    grouped["mean_network_distance_m"] = (
        grouped.heat_distance / grouped.potential.replace(0, np.nan)
    )
    grouped = grouped.drop(columns=["weighted_capex", "heat_distance"])
    grouped["carrier"] = "heat"
    grouped["potential_unit"] = "MWh_th/year"
    grouped["technology"] = "geothermal_heat"
    grouped["price_bin"] = "all"
    grouped["bin_left"] = np.nan
    grouped["bin_right"] = np.nan
    grouped["mean_specific_capex_power"] = np.nan
    return grouped[(grouped.category != "contested") | (grouped.potential > 0)]


def main():
    """Write spatial alternatives and heat supplied to each consuming region."""
    heat_enabled = snakemake.params.carrier_out != "power"
    combined = snakemake.params.carrier_out == "heat_and_power"
    used_fraction = (
        read_raster(snakemake.input.heat_fraction[0]) if heat_enabled else None
    )
    heat_eligible = xr.zeros_like(used_fraction, dtype=bool) if heat_enabled else None
    method = snakemake.params.method_region_aggregation
    n_quantiles = snakemake.params.n_quantiles
    fixed_bins = [
        tuple(map(float, value.split("-", 1))) for value in snakemake.params.fixed_bins
    ]
    lcoe_cutoff = snakemake.params.lcoe_cutoff

    regions = _schemas.ShapeSchema.validate(gpd.read_parquet(snakemake.input.shapes))
    if "shape_class" in regions:
        regions = regions[regions.shape_class != "maritime"]
    regions = regions[["shape_id", "geometry"]].set_index("shape_id")
    regions.index = regions.index.astype(str)

    frames = []
    cheapest_cost = cheapest_potential = None
    for index, (tech, lcoe_path, pot_path, capex_path, power_path) in enumerate(
        zip(
            snakemake.params.cycles,
            snakemake.input.lcoe_tifs,
            snakemake.input.pot_tifs,
            snakemake.input.capex_tifs,
            snakemake.output.power,
            strict=True,
        )
    ):
        costs, capacities = read_raster(lcoe_path), read_raster(pot_path)
        df, eligible = aggregate_one_tech(
            tech,
            costs,
            capacities,
            read_raster(capex_path),
            regions,
            method,
            n_quantiles,
            fixed_bins,
            lcoe_cutoff,
            used_fraction,
            power_path=power_path,
            contested_path=snakemake.output.contested_power[index]
            if combined
            else None,
        )
        # Choose before regional summation; regional mean costs would select a
        # different portfolio. Ties follow the configured cycle order.
        remaining = (
            capacities * (1 - align_to_grid(used_fraction, capacities).clip(0, 1))
            if heat_enabled
            else capacities
        )
        candidate_cost = costs.where(eligible & (remaining > 0), np.inf)
        if cheapest_cost is None:
            cheapest_cost = xr.full_like(costs, np.inf)
            cheapest_potential = xr.zeros_like(capacities)
        better = candidate_cost < cheapest_cost
        cheapest_potential = xr.where(better, remaining, cheapest_potential)
        cheapest_cost = xr.where(better, candidate_cost, cheapest_cost)
        if heat_enabled:
            heat_eligible |= align_to_grid(eligible, used_fraction, fill=False)
        if not df.empty:
            frames.append(df)

    if heat_enabled:
        heat_lcoh = read_raster(snakemake.input.heat_lcoh[0])
        allocated = read_raster(snakemake.input.heat_allocated[0])
        check_grid(allocated, used_fraction)
        write_raster(
            allocated.where(~heat_eligible, 0).where(allocated.notnull()),
            snakemake.output.heat[0],
            "MWh_th/year",
        )
        write_raster(
            allocated.where(heat_eligible, 0).where(allocated.notnull()),
            snakemake.output.contested_heat[0],
            "MWh_th/year",
        )
        ledger = pd.read_parquet(snakemake.input.heat_allocations[0])
        heat_capex = read_raster(snakemake.input.heat_capex[0])
        check_grid(heat_capex, allocated)
        frames.append(
            aggregate_heat(ledger, heat_eligible, regions.index, combined, heat_capex)
        )

    if frames:
        data = pd.concat(frames, ignore_index=True)
        result = gpd.GeoDataFrame(
            data.join(regions[["geometry"]], on="region_id"),
            geometry="geometry",
            crs=regions.crs,
        )
        result = result[list(_schemas.GeothermalRegionsSchema.schema.columns)]
    else:
        result = _schemas.empty_geothermal_regions(regions.crs)
    result = _schemas.GeothermalRegionsSchema.schema.validate(result)
    result.to_parquet(snakemake.output.geoparquet, index=False)

    if snakemake.output.regional_fig:
        power = result.loc[(result.carrier == "power") & (result.category == "power")]
        maps = {
            tech: power.loc[power.technology == tech]
            .groupby("region_id")
            .potential.sum()
            .reindex(regions.index, fill_value=0)
            .to_numpy()
            for tech in snakemake.params.cycles
        }
        selected = gregor.aggregate.aggregate_raster_to_polygon(
            inherit_rio_meta(cheapest_potential, costs),
            regions.to_crs(costs.rio.crs),
            stats="sum",
        )
        maps["winner"] = selected["sum"].reindex(regions.index).fillna(0).to_numpy()
        _plots.make_resource_overview(
            maps,
            list(snakemake.params.cycles),
            snakemake.output.regional_fig,
            snakemake.input.shapes,
            snakemake.params.max_depth,
            potential=True,
            regions=regions,
            reference_grid=costs,
        )
    if heat_enabled:
        for data, output, potential in (
            (heat_lcoh, snakemake.output.heat_lcoh_fig, False),
            (allocated.where(allocated > 0), snakemake.output.heat_potential_fig, True),
        ):
            _plots.make_resource_overview(
                {"heat": data},
                ["heat"],
                output,
                snakemake.input.shapes,
                snakemake.params.max_depth,
                heat=True,
                potential=potential,
            )
    if snakemake.output.regional_heat_fig:
        # Heat categories partition supplied energy; include contested supply once.
        heat = result.loc[result.carrier == "heat"].groupby("region_id").potential.sum()
        _plots.make_resource_overview(
            {"heat": heat.reindex(regions.index, fill_value=0).to_numpy()},
            ["heat"],
            snakemake.output.regional_heat_fig,
            snakemake.input.shapes,
            snakemake.params.max_depth,
            potential=True,
            heat=True,
            regions=regions,
            reference_grid=allocated,
        )


if __name__ == "__main__":
    main()
