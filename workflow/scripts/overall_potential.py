"""Summarise study-wide geothermal potential across sensitivity scenarios."""

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
    """Return ``(scenario, depth category, technology, kind)`` for a 2-D raster."""
    path = Path(path)
    match = RASTER_NAME.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognised 2-D geothermal raster name: {path}")
    if path.parent.name != "documents":
        raise ValueError(f"Expected a raster in a scenario documents directory: {path}")
    return (
        path.parent.parent.name,
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


def total_potential_tw(values):
    """Return the finite potential sum, converting MW to TW."""
    return float(np.nansum(np.asarray(values, dtype=np.float64)) / MW_PER_TW)


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
    return total_potential_tw(selected)


def scenario_potential_table(lcoe_paths, potential_paths, scenario_table, baseline):
    """Build long-form technology and least-LCOE portfolio totals per scenario."""
    paths = {}
    for path in [*lcoe_paths, *potential_paths]:
        key = _raster_key(path)
        if key in paths:
            raise ValueError(f"Duplicate raster for {key}: {path}")
        paths[key] = path

    scenario_metadata = scenario_table.set_index("scenario", drop=False)
    rows = []
    for scenario in scenario_table["scenario"].astype(str):
        option_keys = sorted(
            {
                (depth_category, technology)
                for scen, depth_category, technology, kind in paths
                if scen == scenario and kind == "lcoe"
            }
        )
        if not option_keys:
            raise ValueError(f"No LCOE rasters found for scenario {scenario!r}")

        totals_by_technology = {}
        portfolio_total = 0.0
        for depth_category in sorted({key[0] for key in option_keys}):
            lcoe_by_technology = {}
            potential_by_technology = {}
            reference_grid = None
            for _, technology in [
                key for key in option_keys if key[0] == depth_category
            ]:
                lcoe_key = (scenario, depth_category, technology, "lcoe")
                potential_key = (scenario, depth_category, technology, "potential")
                if potential_key not in paths:
                    raise ValueError(
                        f"Missing matching potential raster for {lcoe_key}"
                    )
                lcoe, lcoe_grid = _read_raster(paths[lcoe_key])
                potential, potential_grid = _read_raster(paths[potential_key])
                if lcoe_grid != potential_grid:
                    raise ValueError(
                        f"LCOE and potential grids differ for {lcoe_key[:-1]}"
                    )
                if reference_grid is not None and lcoe_grid != reference_grid:
                    raise ValueError(
                        f"Technology grids differ in {scenario!r}, {depth_category!r}"
                    )
                reference_grid = lcoe_grid
                lcoe_by_technology[technology] = lcoe
                potential_by_technology[technology] = potential
                totals_by_technology[technology] = totals_by_technology.get(
                    technology, 0.0
                ) + total_potential_tw(potential)

            portfolio_total += lowest_lcoe_potential_tw(
                lcoe_by_technology, potential_by_technology
            )

        metadata = scenario_metadata.loc[scenario]
        common = {
            "scenario": scenario,
            "is_baseline": scenario == baseline,
            "sensitivity_parameter": metadata.get("sensitivity_parameter"),
            "sensitivity_base_value": metadata.get("sensitivity_base_value"),
            "sensitivity_scenario_value": metadata.get("sensitivity_scenario_value"),
        }
        rows.extend(
            {**common, "technology": technology, "potential_tw": potential_tw}
            for technology, potential_tw in sorted(totals_by_technology.items())
        )
        rows.append(
            {**common, "technology": PORTFOLIO_LABEL, "potential_tw": portfolio_total}
        )

    return pd.DataFrame(rows)


def potential_summary_table(by_scenario, baseline):
    """Build publication-ready baseline and sensitivity-envelope values."""
    rows = []
    for technology, values in by_scenario.groupby("technology", sort=True):
        values = values.reset_index(drop=True)
        baseline_values = values.loc[values["scenario"] == baseline, "potential_tw"]
        if len(baseline_values) != 1:
            raise ValueError(
                f"Expected one baseline value for {technology!r}, found {len(baseline_values)}"
            )
        minimum = values.loc[values["potential_tw"].idxmin()]
        maximum = values.loc[values["potential_tw"].idxmax()]
        rows.append(
            {
                "technology": technology,
                "baseline_potential_tw": float(baseline_values.iloc[0]),
                "sensitivity_min_potential_tw": float(minimum["potential_tw"]),
                "sensitivity_max_potential_tw": float(maximum["potential_tw"]),
                "sensitivity_min_scenario": minimum["scenario"],
                "sensitivity_max_scenario": maximum["scenario"],
                "n_sensitivity_scenarios": int((values["scenario"] != baseline).sum()),
            }
        )
    return pd.DataFrame(rows)


def main():
    """Write detailed and publication-ready overall-potential tables."""
    scenario_table = pd.read_csv(snakemake.input["scenario_table"], sep="\t")
    scenario_table["scenario"] = scenario_table["scenario"].astype(str)
    baseline = str(snakemake.params["baseline"])
    by_scenario = scenario_potential_table(
        snakemake.input["lcoe_tifs"],
        snakemake.input["potential_tifs"],
        scenario_table,
        baseline,
    )
    summary = potential_summary_table(by_scenario, baseline)

    for output in snakemake.output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    by_scenario.to_csv(snakemake.output["by_scenario"], sep="\t", index=False)
    summary.to_csv(snakemake.output["summary"], sep="\t", index=False)


if __name__ == "__main__":
    main()
