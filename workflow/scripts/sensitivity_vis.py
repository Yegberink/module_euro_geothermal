import os
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


# =========================
# Scientific plot styling
# =========================
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
})

okabe_ito = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
plt.rcParams["axes.prop_cycle"] = cycler(color=okabe_ito)

PARAM_LABELS = {
    "EGS_cutoff": "EGS cutoff (m)",
    "temp_turbine_out_ORC": "ORC turbine outlet temp",
    "temp_turbine_out_Flash": "Flash turbine outlet temp",
    "temp_turbine_out_Kalina": "Kalina turbine outlet temp",
    "stimulation_cost_per_doublet": "Stimulation cost per doublet",
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
}


# =========================
# Utilities
# =========================
def pretty_param_name(param: str, width: int = 34) -> str:
    label = PARAM_LABELS.get(param, param.replace("_", " "))
    return "\n".join(textwrap.wrap(label, width=width))


def load_scenario_table(path):
    df = pd.read_csv(path, sep="\t", comment="#").replace({np.nan: None})
    df["scenario"] = df["scenario"].astype(str)
    return df.set_index("scenario")


def read_metric_value(metrics_tsv, metric_name):
    df = pd.read_csv(metrics_tsv, sep="\t")
    row = df.loc[df["metric"] == metric_name, "value"]
    val = row.iloc[0]
    return float(val)


def get_scenario_name_from_metrics(metrics_tsv):
    df = pd.read_csv(metrics_tsv, sep="\t")
    row = df.loc[df["metric"] == "scenario", "value"]
    return str(row.iloc[0])


def infer_scenario_from_path(p):
    p2 = str(p).replace("\\", "/")
    parts = p2.split("/scenarios/")
    return parts[1].split("/")[0]


def load_boxplot_table(path):
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def build_baseline_winner_df(df, baseline_name="base"):
    """
    Build one fixed baseline winner table, one row per pixel:
      - cheapest baseline tech by lcoe_base
    """
    req = {"row", "col", "tech", "lcoe_base", "potential_base"}
    df = df.sort_values(["row", "col", "tech"]).copy()

    base_idx = (
        df.loc[np.isfinite(df["lcoe_base"])]
        .groupby(["row", "col"], sort=True)["lcoe_base"]
        .idxmin()
    )

    out = (
        df.loc[base_idx, ["row", "col", "tech", "lcoe_base", "potential_base"]]
        .rename(columns={
            "tech": "tech_base_winner",
            "lcoe_base": "winner_lcoe_base",
            "potential_base": "winner_potential_base",
        })
        .reset_index(drop=True)
    )

    out["baseline"] = baseline_name
    return out


def build_scenario_winner_df(df, scenario_name, baseline_name="base"):
    """
    Build one scenario winner table, one row per pixel:
      - cheapest scenario tech by lcoe_scen
    """
    req = {"row", "col", "tech", "lcoe_scen", "potential_scen"}
    df = df.sort_values(["row", "col", "tech"]).copy()

    scen_idx = (
        df.loc[np.isfinite(df["lcoe_scen"])]
        .groupby(["row", "col"], sort=True)["lcoe_scen"]
        .idxmin()
    )

    out = (
        df.loc[scen_idx, ["row", "col", "tech", "lcoe_scen", "potential_scen"]]
        .rename(columns={
            "tech": "tech_scen_winner",
            "lcoe_scen": "winner_lcoe_scen",
            "potential_scen": "winner_potential_scen",
        })
        .reset_index(drop=True)
    )

    out["scenario"] = scenario_name
    out["baseline"] = baseline_name
    return out


def make_parameter_box_label(param, base_value, scenario_value):
    change = f"{base_value} -> {scenario_value}"
    return f"{pretty_param_name(param, width=18)}\n({change})"


def select_representative_scenarios_per_parameter(df, parameter_order):
    """
    Select at most one scenario per parameter for box plotting.

    Selection rule:
      - highest metric_value
      - then largest change magnitude
      - then scenario name for deterministic tie-breaking
    """
    selected_rows = []

    for param in parameter_order:
        g = df.loc[df["parameter"] == param].copy()
        if g.empty:
            continue

        g = g.sort_values(
            ["metric_value", "sort_magnitude", "scenario"],
            ascending=[False, False, True],
        )

        selected_rows.append(g.iloc[0][[
            "parameter",
            "scenario",
            "label",
            "sort_magnitude",
            "metric_value",
            "transition_total_pixels",
        ]])

    if not selected_rows:
        return pd.DataFrame(columns=[
            "parameter",
            "scenario",
            "label",
            "sort_magnitude",
            "metric_value",
            "transition_total_pixels",
        ])

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def make_single_baseline_plus_parameter_boxplot(specs, value_key, ylabel, title, out_path):
    """
    Plot one baseline box plus one box per selected parameter.

    specs: list of dicts with keys:
      - label
      - values
      - kind in {"baseline", "scenario"}
    """
    positions = []
    data = []
    labels = []
    kinds = []

    pos = 1
    for spec in specs:
        vals = np.asarray(spec[value_key], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        positions.append(pos)
        data.append(vals)
        labels.append(spec["label"])
        kinds.append(spec["kind"])
        pos += 1

    fig_w = max(9, 1.7 * len(data))
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.7,
        showfliers=False,
    )

    kind_colors = {
        "baseline": "#BDBDBD",
        "scenario": "#1F78B4",
    }

    for patch, kind in zip(bp["boxes"], kinds):
        patch.set_facecolor(kind_colors[kind])
        patch.set_alpha(0.80)
        patch.set_linewidth(0.8)

    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_top_switch_rate_barplot(df, top_n, out_path):
    """
    Plot a horizontal bar chart for the scenarios with the largest pixel switch rate.

    Expects one representative scenario per parameter with columns:
      - parameter
      - label
      - transition_total_pixels
    """
    plot_df = df.copy()
    plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()].copy()
    plot_df = plot_df.loc[np.isfinite(plot_df["transition_total_pixels"].to_numpy(dtype=float))].copy()
    plot_df = plot_df.sort_values(
        ["transition_total_pixels", "sort_magnitude", "parameter"],
        ascending=[False, False, True],
    ).head(top_n)

    labels = plot_df["label"].tolist()
    values = plot_df["transition_total_pixels"].to_numpy(dtype=float) * 100.0

    fig_h = max(5.0, 0.9 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(plot_df))
    bars = ax.barh(
        y,
        values,
        color="#D55E00",
        edgecolor="#7F2704",
        alpha=0.85,
        height=0.7,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Pixels with changed winning technology (%)")
    ax.set_title(f"Top {len(plot_df)} parameter scenarios by technology switch rate")
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.35)

    xmax = float(np.nanmax(values)) if len(values) else 0.0
    if not np.isfinite(xmax):
        xmax = 0.0
    ax.set_xlim(0.0, xmax * 1.15 if xmax > 0 else 1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(xmax * 0.015, 0.3),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    scenario_table = snakemake.input["scenario_table"]
    metrics_files = list(snakemake.input["metrics"])
    boxplot_files = list(snakemake.input["boxplot_data"])

    baseline = snakemake.params["baseline"]
    metric_name = snakemake.params.get("metric_name", "mean_abs_pct_change_sens")
    top_n = int(snakemake.params.get("top_n", 6))
    include_all = bool(snakemake.params.get("include_all", False))

    scen_df = load_scenario_table(scenario_table)
    scenario_to_metric = {}
    for p in metrics_files:
        scen_name = get_scenario_name_from_metrics(p) or infer_scenario_from_path(p)
        scenario_to_metric[scen_name] = p

    rows = []
    for scen_name, metrics_path in scenario_to_metric.items():
        if scen_name == baseline:
            continue
        if scen_name not in scen_df.index:
            continue

        row = scen_df.loc[scen_name]
        param = row.get("sensitivity_parameter")
        if param is None or pd.isna(param):
            continue

        sens_val = read_metric_value(metrics_path, metric_name)
        pixels_val = read_metric_value(metrics_path, "switch_rate_pixels")
        base_value = row.get("sensitivity_base_value")
        scenario_value = row.get("sensitivity_scenario_value")
        sort_magnitude = row.get("sensitivity_sort_magnitude")

        rows.append(
            dict(
                scenario=scen_name,
                parameter=str(param),
                label=make_parameter_box_label(param, base_value, scenario_value),
                metric_value=float(sens_val),
                transition_total_pixels=float(pixels_val) if pd.notna(pixels_val) else np.nan,
                sort_magnitude=(
                    float(sort_magnitude) if pd.notna(sort_magnitude) else 1.0
                ),
            )
        )

    df = pd.DataFrame(rows)

    rank_params = (
        df.groupby("parameter")["metric_value"]
        .max()
        .sort_values(ascending=False)
    )

    if not include_all:
        selected_params = rank_params.head(top_n).index.tolist()
    else:
        selected_params = rank_params.index.tolist()

    # -----------------------------------
    # Boxplot selection: baseline once +
    # one representative scenario/parameter
    # -----------------------------------
    representative_df = select_representative_scenarios_per_parameter(
        df=df,
        parameter_order=selected_params,
    )

    scenario_to_boxplot = {}
    for p in boxplot_files:
        scen_name = infer_scenario_from_path(p)
        scenario_to_boxplot[scen_name] = p

    df_box_base = load_boxplot_table(scenario_to_boxplot[baseline])
    baseline_winner_df = build_baseline_winner_df(df_box_base, baseline_name=baseline)

    box_specs = [
        {
            "kind": "baseline",
            "parameter": None,
            "scenario": baseline,
            "pct_change": 0.0,
            "label": "Baseline",
            "lcoe_values": baseline_winner_df["winner_lcoe_base"].to_numpy(),
            "potential_values": baseline_winner_df["winner_potential_base"].to_numpy(),
        }
    ]

    for _, row in representative_df.iterrows():
        param = row["parameter"]
        scen_name = row["scenario"]

        df_box = load_boxplot_table(scenario_to_boxplot[scen_name])
        scen_winner_df = build_scenario_winner_df(
            df_box,
            scenario_name=scen_name,
            baseline_name=baseline,
        )

        box_specs.append(
            {
                "kind": "scenario",
                "parameter": param,
                "scenario": scen_name,
                "label": row["label"],
                "lcoe_values": scen_winner_df["winner_lcoe_scen"].to_numpy(),
                "potential_values": scen_winner_df["winner_potential_scen"].to_numpy(),
            }
        )

    if include_all:
        title_suffix = f"Baseline + one representative scenario per parameter (n={len(representative_df)})"
    else:
        title_suffix = f"Baseline + one representative scenario for top {len(representative_df)} parameters"

    make_single_baseline_plus_parameter_boxplot(
        specs=box_specs,
        value_key="lcoe_values",
        ylabel="LCOE",
        title=f"LCOE distributions for cheapest technology per pixel\n{title_suffix}",
        out_path=snakemake.output["boxplot_lcoe_top6_winner"],
    )

    make_single_baseline_plus_parameter_boxplot(
        specs=box_specs,
        value_key="potential_values",
        ylabel="Potential",
        title=f"Potential distributions for cheapest technology per pixel\n{title_suffix}",
        out_path=snakemake.output["boxplot_potential_top6_winner"],
    )

    df_switch = df.copy()
    df_switch["metric_value"] = df_switch["transition_total_pixels"]
    switch_df = select_representative_scenarios_per_parameter(
        df=df_switch,
        parameter_order=selected_params,
    )
    switch_df["transition_total_pixels"] = switch_df["metric_value"]
    switch_df = switch_df.drop(columns=["metric_value"])

    make_top_switch_rate_barplot(
        df=switch_df,
        top_n=top_n,
        out_path=snakemake.output["barplot_switch_rate_top6"],
    )


if __name__ == "__main__":
    main()
