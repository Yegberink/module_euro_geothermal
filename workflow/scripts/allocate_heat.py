"""Allocate scarce geothermal heat jointly across district-heating networks.

Only feasible network–cell edges are stored. Each connected component is solved
twice: first for maximum served demand, then for minimum annual heat cost at
that supply level. This avoids network-order bias and double counting cells.
"""

import sys
import time

import _schemas
import geopandas as gpd
import numpy as np
import pandas as pd
from _heat import (
    DEMAND_COLUMN,
    METRIC_CRS,
    check_grid,
    load_networks,
    read_raster,
    write_raster,
)
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def assign_network_regions(networks, regions):
    """Assign network centroids, with an intersecting-region fallback for gaps."""
    networks = networks.to_crs(METRIC_CRS).reset_index(drop=True)
    regions = regions.to_crs(METRIC_CRS).sort_values("shape_id").reset_index(drop=True)
    centroids = networks[["geometry"]].copy()
    # FIXME: centroid attribution is inexpensive but does not split networks
    # across region boundaries. Consider demand-weighted polygon allocation.
    centroids.geometry = networks.geometry.centroid
    matches = gpd.sjoin(
        centroids, regions[["shape_id", "geometry"]], predicate="intersects", how="left"
    )
    # Sorted region IDs break exact-boundary and overlap ties deterministically.
    matches = matches.sort_values("shape_id").groupby(level=0).first()
    assigned = matches.shape_id.reindex(networks.index).copy()
    for index in assigned[assigned.isna()].index:
        candidates = regions.sindex.query(
            networks.geometry.iloc[index], predicate="intersects"
        )
        if not len(candidates):
            raise ValueError(
                f"Network {networks.area_id.iloc[index]} has no intersecting region."
            )
        choices = regions.iloc[candidates].copy()
        choices["distance"] = choices.geometry.distance(centroids.geometry.iloc[index])
        chosen = choices.sort_values(["distance", "shape_id"]).iloc[0]
        assigned.loc[index] = chosen.shape_id
        print(
            f"Centroid fallback: network {networks.area_id.iloc[index]} -> {chosen.shape_id} ({chosen.distance:.1f} m)"
        )
    return assigned.astype(str).to_numpy()


def candidate_edges(networks, lcoh, potential, buffer_km):
    """Query buffered polygons against an index of feasible source-cell centres."""
    check_grid(lcoh, potential)
    valid = (
        np.isfinite(lcoh.values)
        & np.isfinite(potential.values)
        & (potential.values > 0)
    )
    row, col = np.nonzero(valid)
    cells = gpd.GeoDataFrame(
        {"cell_id": row * potential.sizes["x"] + col},
        geometry=gpd.points_from_xy(potential.x.values[col], potential.y.values[row]),
        crs=potential.rio.crs,
    ).to_crs(METRIC_CRS)
    if cells.empty or networks.empty:
        return np.empty((0, 2), dtype=np.int64)
    polygons = networks.to_crs(METRIC_CRS).geometry
    # GEOS dwithin tests the buffer relationship without materializing buffers
    # of detailed network boundaries (which otherwise dominate time and RAM).
    network_index, cell_index = cells.sindex.query(
        polygons, predicate="dwithin", distance=buffer_km * 1000
    )
    edges = np.column_stack([network_index, cells.cell_id.to_numpy()[cell_index]])
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def solve_allocation(edges, demand, capacity, cost):
    """Return MWh/year per candidate edge using sparse lexicographic optimization.

    The first stage defines the achievable supply exactly; the second fixes
    that sum and chooses its cheapest feasible distribution. Scarcity is not
    resolved by arbitrary penalties or by the order of input networks.
    """
    edges = np.asarray(edges, dtype=np.int64)
    demand, capacity, cost = map(
        lambda x: np.asarray(x, dtype=float), (demand, capacity, cost)
    )
    if not len(edges):
        return np.zeros(0)
    if not (np.isfinite(demand) & (demand >= 0)).all():
        raise ValueError("Demand must be finite and nonnegative.")
    used_cells, cell_indices = np.unique(edges[:, 1], return_inverse=True)
    if not (
        np.isfinite(capacity[used_cells])
        & (capacity[used_cells] > 0)
        & np.isfinite(cost[used_cells])
    ).all():
        raise ValueError("Candidate cells require finite costs and positive capacity.")
    left, right = edges[:, 0], len(demand) + cell_indices
    graph = coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[left, right], np.r_[right, left])),
        shape=(len(demand) + len(used_cells),) * 2,
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    edge_labels = labels[left]
    order = np.argsort(edge_labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(edge_labels[order])) + 1)
    allocated = np.zeros(len(edges))
    for indices in groups:
        local = edges[indices]
        networks, ni = np.unique(local[:, 0], return_inverse=True)
        cells, ci = np.unique(local[:, 1], return_inverse=True)
        columns = np.arange(len(local))
        constraints = coo_matrix(
            (
                np.ones(2 * len(local)),
                (np.r_[ni, len(networks) + ci], np.r_[columns, columns]),
            ),
            shape=(len(networks) + len(cells), len(local)),
        ).tocsr()
        bounds = np.r_[demand[networks], capacity[cells]]
        options = {
            "primal_feasibility_tolerance": 1e-8,
            "dual_feasibility_tolerance": 1e-8,
        }
        maximum = linprog(
            -np.ones(len(local)),
            A_ub=constraints,
            b_ub=bounds,
            method="highs",
            options=options,
        )
        if not maximum.success:
            raise RuntimeError(f"Heat supply maximization failed: {maximum.message}")
        served = maximum.x.sum()
        if served <= 0:
            continue
        total = coo_matrix(
            (np.ones(len(local)), (np.zeros(len(local), dtype=int), columns)),
            shape=(1, len(local)),
        ).tocsr()
        cheapest = linprog(
            cost[local[:, 1]],
            A_ub=constraints,
            b_ub=bounds,
            A_eq=total,
            b_eq=[served],
            method="highs",
            options=options,
        )
        if not cheapest.success:
            raise RuntimeError(f"Heat cost minimization failed: {cheapest.message}")
        allocated[indices] = np.maximum(cheapest.x, 0)
    network_sum = np.bincount(edges[:, 0], weights=allocated, minlength=len(demand))
    cell_sum = np.bincount(edges[:, 1], weights=allocated, minlength=len(capacity))
    if np.any(network_sum > demand + 1e-5) or np.any(
        cell_sum[used_cells] > capacity[used_cells] + 1e-5
    ):
        raise RuntimeError("Heat allocation exceeded demand or cell capacity.")
    return allocated


def main():
    """Write network summaries, an auditable allocation ledger, and used rasters."""
    start = time.perf_counter()
    networks = load_networks(snakemake.input.networks)
    regions = _schemas.ShapeSchema.validate(gpd.read_parquet(snakemake.input.shapes))
    if "shape_class" in regions:
        regions = regions[regions.shape_class != "maritime"]
    networks["region_id"] = assign_network_regions(networks, regions)
    lcoh, potential = map(
        read_raster, [snakemake.input.lcoh, snakemake.input.potential]
    )
    edges = candidate_edges(networks, lcoh, potential, snakemake.params.buffer_km)
    print(
        f"Candidate edges: {len(edges)}; construction and attribution: {time.perf_counter() - start:.2f} s"
    )
    flow = solve_allocation(
        edges,
        networks[DEMAND_COLUMN].values,
        potential.values.ravel(),
        lcoh.values.ravel(),
    )
    supplied = np.bincount(edges[:, 0], weights=flow, minlength=len(networks))
    annual_cost = np.bincount(
        edges[:, 0],
        weights=flow * lcoh.values.ravel()[edges[:, 1]],
        minlength=len(networks),
    )
    networks["geothermal_supplied_mwh_per_year"] = supplied
    networks["geothermal_unmet_mwh_per_year"] = np.maximum(
        networks[DEMAND_COLUMN] - supplied, 0
    )
    networks["geothermal_lcoh_eur_per_mwh"] = np.divide(
        annual_cost, supplied, out=np.full(len(networks), np.nan), where=supplied > 0
    )
    selected = flow > 0
    ni, ci = edges[selected].T
    row, col = np.divmod(ci, potential.sizes["x"])
    ledger = pd.DataFrame(
        {
            "area_id": networks.area_id.to_numpy()[ni],
            "region_id": networks.region_id.to_numpy()[ni],
            "cell_id": ci,
            "row": row,
            "col": col,
            "x": potential.x.values[col],
            "y": potential.y.values[row],
            "heat_mwh_per_year": flow[selected],
            "cell_fraction": flow[selected] / potential.values.ravel()[ci],
            "lcoh_eur_per_mwh": lcoh.values.ravel()[ci],
        }
    )
    used = np.bincount(edges[:, 1], weights=flow, minlength=potential.size).reshape(
        potential.shape
    )
    valid = (
        np.isfinite(potential.values)
        & (potential.values > 0)
        & np.isfinite(lcoh.values)
    )
    allocated = potential.copy(data=np.where(valid, used, np.nan))
    fraction = potential.copy(
        data=np.divide(
            used, potential.values, out=np.full_like(used, np.nan), where=valid
        )
    )
    # Preserve floating-point precision for conservation across thousands of networks.
    write_raster(allocated, snakemake.output.allocated, "MWh_th/year")
    write_raster(fraction, snakemake.output.fraction, "1")
    for name in ["networks", "allocations"]:
        _schemas.ensure_parent(snakemake.output[name])
    networks.to_parquet(snakemake.output.networks, index=False)
    ledger.to_parquet(snakemake.output.allocations, index=False)
    print(
        f"Allocation completed in {time.perf_counter() - start:.2f} s; supplied {supplied.sum():.3f} MWh/year"
    )


if __name__ == "__main__":
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
