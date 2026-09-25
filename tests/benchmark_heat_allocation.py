"""Benchmark sparse allocation with real polygons and synthetic 10 km supply.

Run with the module runtime's Python. This does not estimate real geothermal
potential: constant cell supply and deterministic synthetic costs isolate the
performance of candidate construction and joint allocation.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/scripts"))

from _heat import DEMAND_COLUMN, METRIC_CRS, load_networks  # noqa: E402
from allocate_heat import candidate_edges, solve_allocation  # noqa: E402


def main():
    """Report candidate count, timing, memory and conservation as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "networks", nargs="?", default="resources/user/BENLDE/areas.parquet"
    )
    args = parser.parse_args()
    networks = load_networks(args.networks).to_crs(METRIC_CRS)
    lo_x, lo_y, hi_x, hi_y = networks.total_bounds
    spacing = 10000
    x = np.arange(
        np.floor(lo_x / spacing) * spacing - spacing / 2, hi_x + spacing, spacing
    )
    y = np.arange(
        np.ceil(hi_y / spacing) * spacing + spacing / 2, lo_y - spacing, -spacing
    )
    row, col = np.indices((len(y), len(x)))
    costs = xr.DataArray(
        15 + (col % 23) + row * 0.01, dims=("y", "x"), coords={"x": x, "y": y}
    )
    costs = costs.rio.write_crs(METRIC_CRS).rio.write_transform(
        Affine(spacing, 0, x[0] - spacing / 2, 0, -spacing, y[0] + spacing / 2)
    )
    capacity = costs.copy(data=np.full(costs.shape, 100000.0))
    start = time.perf_counter()
    edges = candidate_edges(networks, costs, capacity, 10)
    construction = time.perf_counter() - start
    start = time.perf_counter()
    flow = solve_allocation(
        edges, networks[DEMAND_COLUMN], capacity.values.ravel(), costs.values.ravel()
    )
    solving = time.perf_counter() - start
    supplied = np.bincount(edges[:, 0], weights=flow, minlength=len(networks))
    used = np.bincount(edges[:, 1], weights=flow, minlength=capacity.size)
    assert np.all(supplied <= networks[DEMAND_COLUMN].values + 1e-5)
    assert np.all(used <= capacity.values.ravel() + 1e-5)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    print(
        json.dumps(
            {
                "networks": len(networks),
                "grid_cells": costs.size,
                "candidate_edges": len(edges),
                "candidate_array_mib": edges.nbytes / 1024**2,
                "dense_network_cell_entries_avoided": len(networks) * costs.size,
                "candidate_seconds": construction,
                "solve_seconds": solving,
                "process_peak_rss_mib": peak_mib,
                "supplied_mwh_per_year": float(flow.sum()),
                "unmet_mwh_per_year": float(networks[DEMAND_COLUMN].sum() - flow.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
