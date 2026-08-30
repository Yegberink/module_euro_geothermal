"""Calculate per-scenario sensitivity tables and metrics."""

# The small procedural helpers are intentionally public within this Snakemake script.
# ruff: noqa: D103

import re
import sys
from pathlib import Path

import _schemas
import numpy as np
import pandas as pd
import rasterio

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def make_cycle_re(cycles):
    return re.compile(
        r"/(?P<cycle>"
        + "|".join(map(re.escape, cycles))
        + r")_2d_(?P<kind>lcoe|potential)\.tif$",
        re.IGNORECASE,
    )


def list_to_cycle_map(paths, cycle_re):
    output = {}
    for path in paths:
        match = cycle_re.search(str(path).replace("\\", "/"))
        if match is None:
            raise ValueError(f"Unrecognized sensitivity raster path: {path}")
        output[match.group("cycle").lower()] = str(path)
    return output


def ensure_north_up(array, transform):
    """Give all raster arrays the same row/column orientation."""
    if transform.e > 0:
        array = np.flipud(array)
    if transform.a < 0:
        array = np.fliplr(array)
    return array


def read_tif(path):
    with rasterio.open(path) as source:
        array = source.read(1).astype(np.float32)
        nodata = source.nodata
        transform = source.transform
    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)
    array = np.where(np.isfinite(array), array, np.nan).astype(np.float32)
    return ensure_north_up(array, transform)


def stack_by_tech(paths_by_tech, cycles):
    return np.stack([read_tif(paths_by_tech[cycle]) for cycle in cycles], axis=0)


def pct_change_map(base, scenario, eps=1e-6):
    output = np.full_like(base, np.nan, dtype=np.float32)
    valid = np.isfinite(base) & np.isfinite(scenario) & (np.abs(base) > eps)
    output[valid] = 100 * (scenario[valid] - base[valid]) / base[valid]
    return output


def save_pixel_table(
    out_path,
    scenario,
    baseline,
    cycles,
    base_lcoe_stack,
    scenario_lcoe_stack,
    base_potential_stack,
    scenario_potential_stack,
):
    rows = []
    for index, technology in enumerate(cycles):
        lcoe_base = base_lcoe_stack[index]
        lcoe_scenario = scenario_lcoe_stack[index]
        potential_base = base_potential_stack[index]
        potential_scenario = scenario_potential_stack[index]
        percent_change = pct_change_map(lcoe_base, lcoe_scenario)
        valid = (
            np.isfinite(lcoe_base)
            | np.isfinite(lcoe_scenario)
            | np.isfinite(potential_base)
            | np.isfinite(potential_scenario)
        )
        row, col = np.where(valid)
        rows.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "row": row.astype(np.int32),
                    "col": col.astype(np.int32),
                    "tech": technology,
                    "lcoe_base": lcoe_base[row, col].astype(np.float32),
                    "lcoe_scen": lcoe_scenario[row, col].astype(np.float32),
                    "potential_base": potential_base[row, col].astype(np.float32),
                    "potential_scen": potential_scenario[row, col].astype(np.float32),
                    "pct_change_lcoe": percent_change[row, col].astype(np.float32),
                }
            )
        )
    table = (
        pd.concat(rows, ignore_index=True) if rows else _schemas.empty_boxplot_table()
    )
    _schemas.write_boxplot_table(table, out_path)


def argmin_with_nan(stack):
    all_nan = np.all(~np.isfinite(stack), axis=0)
    filled = np.where(np.isfinite(stack), stack, np.inf)
    winner = np.where(all_nan, -1, np.argmin(filled, axis=0)).astype(np.int16)
    best = np.where(all_nan, np.nan, np.min(filled, axis=0)).astype(np.float32)
    if filled.shape[0] < 2:
        return winner, best, np.full(best.shape, np.nan, dtype=np.float32)
    runner_up = np.where(all_nan, np.nan, np.sort(filled, axis=0)[1]).astype(np.float32)
    return winner, best, (runner_up - best).astype(np.float32)


def technology_shares(winner, cycles, tech_code):
    valid = winner >= 0
    count = int(valid.sum())
    if count == 0:
        return {technology: np.nan for technology in cycles}
    return {
        technology: float((winner == tech_code[technology]).sum() / count)
        for technology in cycles
    }


def transition_matrix(base_winner, scenario_winner, cycles):
    valid = (base_winner >= 0) & (scenario_winner >= 0)
    matrix = np.zeros((len(cycles), len(cycles)), dtype=np.int64)
    for source in range(len(cycles)):
        for target in range(len(cycles)):
            matrix[source, target] = int(
                ((base_winner == source) & (scenario_winner == target) & valid).sum()
            )
    return matrix, int(valid.sum())


def finite_stat(array, operation):
    values = np.asarray(array)
    values = values[np.isfinite(values)]
    return float(operation(values)) if values.size else np.nan


def _write_metrics(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=["metric", "value"])
    table.to_csv(path, sep="\t", index=False)
    _schemas.SensitivityMetricsSchema.validate(pd.read_csv(path, sep="\t", dtype=str))


def main():
    cycles = [str(c).strip().lower() for c in snakemake.params["cycles"]]
    baseline = str(snakemake.params["baseline"])
    scenario = str(snakemake.wildcards.scenario)
    sens_tech = str(snakemake.params.get("sens_tech", "orc")).strip().lower()
    tech_code = {technology: index for index, technology in enumerate(cycles)}
    cycle_re = make_cycle_re(cycles)

    base_lcoe = stack_by_tech(
        list_to_cycle_map(snakemake.input.base_lcoe, cycle_re), cycles
    )
    scenario_lcoe = stack_by_tech(
        list_to_cycle_map(snakemake.input.scen_lcoe, cycle_re), cycles
    )
    base_potential = stack_by_tech(
        list_to_cycle_map(snakemake.input.base_pot, cycle_re), cycles
    )
    scenario_potential = stack_by_tech(
        list_to_cycle_map(snakemake.input.scen_pot, cycle_re), cycles
    )
    save_pixel_table(
        snakemake.output.boxplot_data,
        scenario,
        baseline,
        cycles,
        base_lcoe,
        scenario_lcoe,
        base_potential,
        scenario_potential,
    )

    sensitivity_change = pct_change_map(
        base_lcoe[tech_code[sens_tech]], scenario_lcoe[tech_code[sens_tech]]
    )
    finite_change = sensitivity_change[np.isfinite(sensitivity_change)]
    mean_abs_change_sens = (
        float(np.mean(np.abs(finite_change))) if finite_change.size else np.nan
    )
    max_abs_change_sens = (
        float(np.max(np.abs(finite_change))) if finite_change.size else np.nan
    )

    common_rows = [
        ("scenario", scenario),
        ("baseline", baseline),
        ("cycles_included", ",".join(cycles)),
        ("sens_tech", sens_tech),
    ]
    if len(cycles) == 1:
        minimum_base = finite_stat(base_lcoe[0], np.min)
        minimum_scenario = finite_stat(scenario_lcoe[0], np.min)
        mean_base = finite_stat(base_lcoe[0], np.mean)
        mean_scenario = finite_stat(scenario_lcoe[0], np.mean)
        maximum_base = finite_stat(base_lcoe[0], np.max)
        maximum_scenario = finite_stat(scenario_lcoe[0], np.max)
        rows = common_rows + [
            ("mean_abs_pct_change_sens", mean_abs_change_sens),
            ("max_abs_pct_change_sens", max_abs_change_sens),
            ("min_lcoe_baseline", minimum_base),
            ("min_lcoe_scenario", minimum_scenario),
            ("mean_lcoe_baseline", mean_base),
            ("mean_lcoe_scenario", mean_scenario),
            ("max_lcoe_baseline", maximum_base),
            ("max_lcoe_scenario", maximum_scenario),
            (f"min_lcoe_baseline_{cycles[0]}", minimum_base),
            (f"min_lcoe_scenario_{cycles[0]}", minimum_scenario),
            (f"mean_lcoe_baseline_{cycles[0]}", mean_base),
            (f"mean_lcoe_scenario_{cycles[0]}", mean_scenario),
            (f"max_lcoe_baseline_{cycles[0]}", maximum_base),
            (f"max_lcoe_scenario_{cycles[0]}", maximum_scenario),
            ("switch_rate_pixels", np.nan),
            ("switch_rate_potential_weighted", np.nan),
            ("mean_abs_pct_change_best", np.nan),
            ("mean_abs_pct_change_best_margin_weighted", np.nan),
            ("mean_abs_pct_change_best_simple", np.nan),
            ("max_abs_pct_change_best", np.nan),
            ("transition_total_pixels", np.nan),
            (f"share_baseline_{cycles[0]}", 1.0),
            (f"share_scenario_{cycles[0]}", 1.0),
        ]
        _write_metrics(snakemake.output.metrics, rows)
        return

    base_winner, best_base, base_margin = argmin_with_nan(base_lcoe)
    scenario_winner, best_scenario, _ = argmin_with_nan(scenario_lcoe)
    valid_both = (base_winner >= 0) & (scenario_winner >= 0)
    switched = np.where(
        valid_both, (base_winner != scenario_winner).astype(np.float32), np.nan
    )
    best_base_potential = np.full_like(best_base, np.nan, dtype=np.float32)
    for index in range(len(cycles)):
        best_base_potential = np.where(
            base_winner == index, base_potential[index], best_base_potential
        )
    switch_rate = finite_stat(switched, np.mean)
    weighted = (
        np.isfinite(switched)
        & np.isfinite(best_base_potential)
        & (best_base_potential > 0)
    )
    switch_rate_weighted = (
        float(
            np.sum(switched[weighted] * best_base_potential[weighted])
            / np.sum(best_base_potential[weighted])
        )
        if weighted.any()
        else np.nan
    )

    best_change = pct_change_map(best_base, best_scenario)
    finite_best_change = best_change[np.isfinite(best_change)]
    mean_abs_best_simple = (
        float(np.mean(np.abs(finite_best_change)))
        if finite_best_change.size
        else np.nan
    )
    max_abs_best = (
        float(np.max(np.abs(finite_best_change))) if finite_best_change.size else np.nan
    )
    margin_weight = 1 / (base_margin + 1e-6)
    valid_margin = (
        np.isfinite(best_change) & np.isfinite(margin_weight) & (base_winner >= 0)
    )
    mean_abs_best = (
        float(np.mean(np.abs(best_change[valid_margin])))
        if valid_margin.any()
        else np.nan
    )
    mean_abs_best_margin = (
        float(
            np.sum(np.abs(best_change[valid_margin]) * margin_weight[valid_margin])
            / np.sum(margin_weight[valid_margin])
        )
        if valid_margin.any()
        else np.nan
    )
    base_shares = technology_shares(base_winner, cycles, tech_code)
    scenario_shares = technology_shares(scenario_winner, cycles, tech_code)
    transitions, comparable_pixels = transition_matrix(
        base_winner, scenario_winner, cycles
    )

    rows = common_rows + [
        ("switch_rate_pixels", switch_rate),
        ("switch_rate_potential_weighted", switch_rate_weighted),
        ("mean_abs_pct_change_best", mean_abs_best),
        ("mean_abs_pct_change_best_margin_weighted", mean_abs_best_margin),
        ("mean_abs_pct_change_best_simple", mean_abs_best_simple),
        ("max_abs_pct_change_best", max_abs_best),
        ("mean_abs_pct_change_sens", mean_abs_change_sens),
        ("max_abs_pct_change_sens", max_abs_change_sens),
        ("min_lcoe_baseline", finite_stat(best_base, np.min)),
        ("min_lcoe_scenario", finite_stat(best_scenario, np.min)),
        ("mean_lcoe_baseline", finite_stat(best_base, np.mean)),
        ("mean_lcoe_scenario", finite_stat(best_scenario, np.mean)),
        ("max_lcoe_baseline", finite_stat(best_base, np.max)),
        ("max_lcoe_scenario", finite_stat(best_scenario, np.max)),
    ]
    for technology in cycles:
        index = tech_code[technology]
        rows.extend(
            [
                (
                    f"min_lcoe_baseline_{technology}",
                    finite_stat(base_lcoe[index], np.min),
                ),
                (
                    f"min_lcoe_scenario_{technology}",
                    finite_stat(scenario_lcoe[index], np.min),
                ),
                (
                    f"mean_lcoe_baseline_{technology}",
                    finite_stat(base_lcoe[index], np.mean),
                ),
                (
                    f"mean_lcoe_scenario_{technology}",
                    finite_stat(scenario_lcoe[index], np.mean),
                ),
                (
                    f"max_lcoe_baseline_{technology}",
                    finite_stat(base_lcoe[index], np.max),
                ),
                (
                    f"max_lcoe_scenario_{technology}",
                    finite_stat(scenario_lcoe[index], np.max),
                ),
                (f"share_baseline_{technology}", base_shares[technology]),
                (f"share_scenario_{technology}", scenario_shares[technology]),
            ]
        )
    rows.append(("transition_total_pixels", comparable_pixels))
    for source, source_technology in enumerate(cycles):
        for target, target_technology in enumerate(cycles):
            rows.append(
                (
                    f"transition_{source_technology}_to_{target_technology}",
                    transitions[source, target],
                )
            )
    _write_metrics(snakemake.output.metrics, rows)


if __name__ == "__main__":
    main()
