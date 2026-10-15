"""Summarise study-wide geothermal potential for the configured model."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

MW_PER_TW = 1_000_000.0
PORTFOLIO_LABEL = "lowest_lcoe_technology"
RASTER_NAME = re.compile(
    r"^(?P<technology>[^_]+)_2d_(?:(?P<depth_category>.+)_)?"
    r"(?P<kind>lcoe|potential)\.tif$"
)


def _raster_key(path):
    """Return ``(depth category, technology, kind)`` for a 2-D raster."""
    path = Path(path)
    match = RASTER_NAME.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognised 2-D geothermal raster name: {path}")
    if path.parent.name != "documents":
        raise ValueError(f"Expected a raster in a documents directory: {path}")
    return (
        match.group("depth_category") or "lcoe_minimization",
        match.group("technology").lower(),
        match.group("kind"),
    )


def _read_raster(path):
    with rasterio.open(path) as src:
        values = src.read(1, masked=True).filled(np.nan).astype(np.float64)
        grid = (src.shape, src.transform, src.crs)
    values[~np.isfinite(values)] = np.nan
    return values, grid


def lowest_lcoe_potential_tw(lcoe_by_technology, potential_by_technology):
    """Sum potential from the lowest-LCOE available technology in every cell."""
    technologies = list(lcoe_by_technology)
    if not technologies:
        return 0.0
    if set(technologies) != set(potential_by_technology):
        raise ValueError("LCOE and potential technologies do not match")

    lcoe = np.stack([lcoe_by_technology[name] for name in technologies])
    potential = np.stack([potential_by_technology[name] for name in technologies])
    valid = np.isfinite(lcoe) & np.isfinite(potential)
    winner = np.argmin(np.where(valid, lcoe, np.inf), axis=0)
    selected = np.take_along_axis(potential, winner[None, ...], axis=0)[0]
    selected[~valid.any(axis=0)] = np.nan
    return float(np.nansum(np.asarray(selected, dtype=np.float64)) / MW_PER_TW)


def potential_summary_table(lcoe_paths, potential_paths):
    """Build technology and least-LCOE portfolio totals for the configured model."""
    paths = {}
    for path in [*lcoe_paths, *potential_paths]:
        key = _raster_key(path)
        if key in paths:
            raise ValueError(f"Duplicate raster for {key}: {path}")
        paths[key] = path

    rows = []
    option_keys = sorted(
        {
            (depth_category, technology)
            for depth_category, technology, kind in paths
            if kind == "lcoe"
        }
    )
    if not option_keys:
        raise ValueError("No LCOE rasters found")

    totals_by_technology = {}
    portfolio_total = 0.0
    for depth_category in sorted({key[0] for key in option_keys}):
        lcoe_by_technology = {}
        potential_by_technology = {}
        reference_grid = None
        for _, technology in [key for key in option_keys if key[0] == depth_category]:
            lcoe_key = (depth_category, technology, "lcoe")
            potential_key = (depth_category, technology, "potential")
            if potential_key not in paths:
                raise ValueError(f"Missing matching potential raster for {lcoe_key}")
            lcoe, lcoe_grid = _read_raster(paths[lcoe_key])
            potential, potential_grid = _read_raster(paths[potential_key])
            if lcoe_grid != potential_grid:
                raise ValueError(f"LCOE and potential grids differ for {lcoe_key[:-1]}")
            if reference_grid is not None and lcoe_grid != reference_grid:
                raise ValueError(f"Technology grids differ in {depth_category!r}")
            reference_grid = lcoe_grid
            lcoe_by_technology[technology] = lcoe
            potential_by_technology[technology] = potential
            totals_by_technology[technology] = totals_by_technology.get(
                technology, 0.0
            ) + float(np.nansum(np.asarray(potential, dtype=np.float64)) / MW_PER_TW)

        portfolio_total += lowest_lcoe_potential_tw(
            lcoe_by_technology, potential_by_technology
        )

    rows.extend(
        {"technology": technology, "potential_tw": potential_tw}
        for technology, potential_tw in sorted(totals_by_technology.items())
    )
    rows.append({"technology": PORTFOLIO_LABEL, "potential_tw": portfolio_total})

    return pd.DataFrame(rows)


def main():
    """Write study-wide technology and least-cost portfolio potential totals."""
    summary = potential_summary_table(
        snakemake.input["lcoe_tifs"], snakemake.input["potential_tifs"]
    )
    output = snakemake.output["summary"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    main()
