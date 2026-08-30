"""Prepare study-wide data for the retained sensitivity figures."""

# The small procedural helpers are intentionally public within this Snakemake script.
# ruff: noqa: D103

import sys
import textwrap

import _plots
import _schemas
import numpy as np
import pandas as pd

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


PARAM_LABELS = {
    "EGS_cutoff": "EGS cutoff (m)",
    "temp_turbine_out_ORC": "ORC turbine outlet temp",
    "temp_turbine_out_Flash": "Flash turbine outlet temp",
    "temp_turbine_out_Kalina": "Kalina turbine outlet temp",
    "stimulation_cost_per_doublet": "Stimulation cost per doublet",
    "well_spacing": "Well spacing",
    "pump_cost": "Pump cost",
    "lifetime": "Lifetime",
    "capacity_factor": "Capacity factor",
    "fixed_OM_rate": "Fixed O&M rate",
    "parasitic_fraction": "Parasitic fraction",
    "n_pairs": "Max doublets per pixel",
    "orc_money_per_kw": "ORC CAPEX",
    "flash_money_per_kw": "Flash CAPEX",
    "kalina_money_per_kw": "Kalina CAPEX",
    "drilling_cost_curve": "Drilling cost curve",
    "include_egs": "Include EGS",
    "SED_NAN_POLICY": "Sediment NaN policy",
    "max_depth": "Max depth",
    "flowrate_egs": "EGS flow rate",
    "flowrate_conventional": "Conventional flow rate",
}


def pretty_param_name(param: str, width: int = 34) -> str:
    label = PARAM_LABELS.get(param, param.replace("_", " "))
    return "\n".join(textwrap.wrap(label, width=width))


def load_scenario_table(path):
    df = pd.read_csv(path, sep="\t", comment="#").replace({np.nan: None})
    df["scenario"] = df["scenario"].astype(str)
    return df.set_index("scenario")


def read_metric_value(metrics_tsv, metric_name):
    df = pd.read_csv(metrics_tsv, sep="\t")
    return float(df.loc[df["metric"] == metric_name, "value"].iloc[0])


def get_scenario_name_from_metrics(metrics_tsv):
    df = pd.read_csv(metrics_tsv, sep="\t")
    return str(df.loc[df["metric"] == "scenario", "value"].iloc[0])


def infer_scenario_from_path(path):
    return str(path).replace("\\", "/").split("/scenarios/")[1].split("/")[0]


def load_boxplot_table(path):
    path = str(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_baseline_winner_df(df, baseline_name="base"):
    df = df.sort_values(["row", "col", "tech"]).copy()
    idx = (
        df.loc[np.isfinite(df["lcoe_base"])]
        .groupby(["row", "col"], sort=True)["lcoe_base"]
        .idxmin()
    )
    out = (
        df.loc[idx, ["row", "col", "tech", "lcoe_base", "potential_base"]]
        .rename(
            columns={
                "tech": "tech_base_winner",
                "lcoe_base": "winner_lcoe_base",
                "potential_base": "winner_potential_base",
            }
        )
        .reset_index(drop=True)
    )
    out["baseline"] = baseline_name
    return out


def build_scenario_winner_df(df, scenario_name, baseline_name="base"):
    df = df.sort_values(["row", "col", "tech"]).copy()
    idx = (
        df.loc[np.isfinite(df["lcoe_scen"])]
        .groupby(["row", "col"], sort=True)["lcoe_scen"]
        .idxmin()
    )
    out = (
        df.loc[idx, ["row", "col", "tech", "lcoe_scen", "potential_scen"]]
        .rename(
            columns={
                "tech": "tech_scen_winner",
                "lcoe_scen": "winner_lcoe_scen",
                "potential_scen": "winner_potential_scen",
            }
        )
        .reset_index(drop=True)
    )
    out["scenario"] = scenario_name
    out["baseline"] = baseline_name
    return out


def make_parameter_box_label(param, base_value, scenario_value):
    return f"{pretty_param_name(param, width=18)}\n({base_value} -> {scenario_value})"


def summarise_potential_tw(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return f"Total: {np.nansum(values) / 1e6:.1f} TW" if values.size else ""


def mean_absolute_change(base_values, scenario_values):
    base = np.asarray(base_values, dtype=float)
    scenario = np.asarray(scenario_values, dtype=float)
    base = base[np.isfinite(base)]
    scenario = scenario[np.isfinite(scenario)]
    if not base.size or not scenario.size:
        return np.nan
    return abs(float(np.nanmean(scenario)) - float(np.nanmean(base)))


def total_potential_change_pct(base_values, scenario_values):
    base = np.asarray(base_values, dtype=float)
    scenario = np.asarray(scenario_values, dtype=float)
    base_total = float(np.nansum(base[np.isfinite(base)]))
    scenario_total = float(np.nansum(scenario[np.isfinite(scenario)]))
    if np.isclose(base_total, 0):
        return abs(scenario_total - base_total)
    return abs((scenario_total - base_total) / base_total) * 100


def select_representative_scenarios_per_parameter(df, parameter_order):
    selected = []
    for parameter in parameter_order:
        group = df.loc[df["parameter"] == parameter].copy()
        if group.empty:
            continue
        group = group.sort_values(
            ["metric_value", "sort_magnitude", "scenario"],
            ascending=[False, False, True],
        )
        selected.append(
            group.iloc[0][
                [
                    "parameter",
                    "scenario",
                    "label",
                    "sort_magnitude",
                    "metric_value",
                    "transition_total_pixels",
                ]
            ]
        )
    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


def sort_representative_scenarios(df):
    if df.empty:
        return df
    return df.sort_values(
        ["metric_value", "sort_magnitude", "parameter", "scenario"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def include_required_parameters(selected, available, required):
    output = list(selected)
    for parameter in required:
        if parameter in available.index and parameter not in output:
            output.append(parameter)
    return output


def _write_blank_outputs():
    for name in (
        "boxplot_lcoe_top6_winner",
        "boxplot_potential_top6_winner",
        "map_switch_transition_top4",
    ):
        _plots.write_blank_png(snakemake.output[name])


def main():
    baseline = snakemake.params["baseline"]
    metric_name = snakemake.params.get("metric_name", "mean_abs_pct_change_sens")
    top_n = int(snakemake.params.get("top_n", 6))
    map_top_n = 4
    include_all = bool(snakemake.params.get("include_all", False))
    required_params = ["well_spacing", "stimulation_cost_per_doublet"]
    scenario_table = load_scenario_table(snakemake.input["scenario_table"])

    scenario_to_metric = {
        get_scenario_name_from_metrics(path) or infer_scenario_from_path(path): path
        for path in snakemake.input["metrics"]
    }
    rows = []
    for scenario, metrics_path in scenario_to_metric.items():
        if scenario == baseline or scenario not in scenario_table.index:
            continue
        scenario_row = scenario_table.loc[scenario]
        parameter = scenario_row.get("sensitivity_parameter")
        if parameter is None or pd.isna(parameter):
            continue
        pixels = read_metric_value(metrics_path, "switch_rate_pixels")
        magnitude = scenario_row.get("sensitivity_sort_magnitude")
        rows.append(
            {
                "scenario": scenario,
                "parameter": str(parameter),
                "label": make_parameter_box_label(
                    parameter,
                    scenario_row.get("sensitivity_base_value"),
                    scenario_row.get("sensitivity_scenario_value"),
                ),
                "metric_value": read_metric_value(metrics_path, metric_name),
                "transition_total_pixels": float(pixels)
                if pd.notna(pixels)
                else np.nan,
                "sort_magnitude": float(magnitude) if pd.notna(magnitude) else 1,
            }
        )
    ranking_data = pd.DataFrame(rows)
    if ranking_data.empty:
        _write_blank_outputs()
        return

    ranked_parameters = (
        ranking_data.groupby("parameter")["metric_value"]
        .max()
        .sort_values(ascending=False)
    )
    selected_parameters = (
        ranked_parameters.index.tolist()
        if include_all
        else ranked_parameters.head(top_n).index.tolist()
    )
    selected_parameters = include_required_parameters(
        selected_parameters, ranked_parameters, required_params
    )

    scenario_to_boxplot = {
        infer_scenario_from_path(path): path for path in snakemake.input["boxplot_data"]
    }
    if baseline not in scenario_to_boxplot:
        _write_blank_outputs()
        return
    baseline_table = _schemas.SensitivityBoxplotSchema.validate(
        load_boxplot_table(scenario_to_boxplot[baseline])
    )
    if baseline_table.empty:
        _write_blank_outputs()
        return
    baseline_winners = build_baseline_winner_df(baseline_table, baseline)
    baseline_lcoe = baseline_winners["winner_lcoe_base"].to_numpy()
    baseline_potential = baseline_winners["winner_potential_base"].to_numpy()

    scenario_specs = []
    for _, row in ranking_data.iterrows():
        scenario = row["scenario"]
        if scenario not in scenario_to_boxplot:
            continue
        table = _schemas.SensitivityBoxplotSchema.validate(
            load_boxplot_table(scenario_to_boxplot[scenario])
        )
        if table.empty:
            continue
        winners = build_scenario_winner_df(table, scenario, baseline)
        scenario_lcoe = winners["winner_lcoe_scen"].to_numpy()
        scenario_potential = winners["winner_potential_scen"].to_numpy()
        scenario_specs.append(
            {
                "kind": "scenario",
                "parameter": row["parameter"],
                "scenario": scenario,
                "label": row["label"],
                "lcoe_values": scenario_lcoe,
                "potential_values": scenario_potential,
                "lcoe_metric_value": mean_absolute_change(baseline_lcoe, scenario_lcoe),
                "potential_metric_value": mean_absolute_change(
                    baseline_potential, scenario_potential
                ),
                "potential_total_metric_value": total_potential_change_pct(
                    baseline_potential, scenario_potential
                ),
                "sort_magnitude": float(row["sort_magnitude"]),
                "transition_total_pixels": float(row["transition_total_pixels"]),
            }
        )
    if not scenario_specs:
        _write_blank_outputs()
        return

    def representative_specs(metric_key):
        rank_df = pd.DataFrame(
            [
                {
                    "parameter": spec["parameter"],
                    "scenario": spec["scenario"],
                    "label": spec["label"],
                    "sort_magnitude": spec["sort_magnitude"],
                    "metric_value": spec[metric_key],
                    "transition_total_pixels": spec["transition_total_pixels"],
                }
                for spec in scenario_specs
            ]
        )
        parameter_order = (
            rank_df.groupby("parameter")["metric_value"]
            .max()
            .sort_values(ascending=False)
        )
        parameter_order = (
            parameter_order.index.tolist()
            if include_all
            else parameter_order.head(top_n).index.tolist()
        )
        return sort_representative_scenarios(
            select_representative_scenarios_per_parameter(rank_df, parameter_order)
        )

    representative_lcoe = representative_specs("lcoe_metric_value")
    representative_potential = representative_specs("potential_metric_value")
    spec_by_scenario = {spec["scenario"]: spec for spec in scenario_specs}
    baseline_spec = {
        "kind": "baseline",
        "scenario": baseline,
        "label": "Baseline",
        "lcoe_values": baseline_lcoe,
        "potential_values": baseline_potential,
    }
    lcoe_specs = [baseline_spec] + [
        spec_by_scenario[row["scenario"]] for _, row in representative_lcoe.iterrows()
    ]
    potential_specs = [baseline_spec] + [
        spec_by_scenario[row["scenario"]]
        for _, row in representative_potential.iterrows()
    ]

    _plots.make_sensitivity_boxplot(
        specs=lcoe_specs,
        value_key="lcoe_values",
        ylabel="LCOE (€/MWh)",
        out_path=snakemake.output["boxplot_lcoe_top6_winner"],
        summary_position="none",
        show_mean_dot=True,
    )
    _plots.make_sensitivity_boxplot(
        specs=potential_specs,
        value_key="potential_values",
        ylabel="Potential (GW per 10x10km grid cell)",
        out_path=snakemake.output["boxplot_potential_top6_winner"],
        summary_fn=summarise_potential_tw,
        summary_position="top",
        show_mean_dot=True,
    )

    switch_ranking = ranking_data.copy()
    switch_ranking["metric_value"] = switch_ranking["transition_total_pixels"]
    switch_df = select_representative_scenarios_per_parameter(
        switch_ranking, selected_parameters
    )
    switch_df["transition_total_pixels"] = switch_df["metric_value"]
    switch_df = switch_df.drop(columns=["metric_value"])
    _plots.make_switch_transition_map(
        switch_df=switch_df,
        scenario_to_boxplot=scenario_to_boxplot,
        baseline_winner_df=baseline_winners,
        baseline=baseline,
        top_n=map_top_n,
        out_path=snakemake.output["map_switch_transition_top4"],
        shapes_path=snakemake.input["shapes"],
        reference_raster_path=snakemake.input["reference_lcoe"],
        required_params=required_params,
    )


if __name__ == "__main__":
    main()
