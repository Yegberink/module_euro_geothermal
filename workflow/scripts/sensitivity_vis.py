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
    "fixed_OM_rate": "Fixed O&M rate",
    "parasitic_fraction": "Parasitic fraction",
    "n_pairs": "Max doublets per pixel",
    "orc_money_per_kw": "ORC CAPEX",
    "flash_money_per_kw": "Flash CAPEX",
    "kalina_money_per_kw": "Kalina CAPEX",
    "well_cost_sens": "Well cost",
    "include_egs": "Include EGS",
}


# =========================
# Utilities
# =========================
def pretty_param_name(param: str, width: int = 34) -> str:
    label = PARAM_LABELS.get(param, param.replace("_", " "))
    return "\n".join(textwrap.wrap(label, width=width))


def is_number(x):
    if isinstance(x, (bool, np.bool_)):
        return False
    if x is None:
        return False
    if isinstance(x, str) and x.strip().lower() in {"true", "false"}:
        return False
    try:
        float(x)
        return True
    except Exception:
        return False


def load_scenario_table(path):
    df = pd.read_csv(path, sep="\t", comment="#").replace({np.nan: None})
    if "scenario" not in df.columns:
        raise ValueError(f"{path} must have a 'scenario' column.")
    df["scenario"] = df["scenario"].astype(str)
    return df.set_index("scenario")


def read_metric_value(metrics_tsv, metric_name):
    df = pd.read_csv(metrics_tsv, sep="\t")
    row = df.loc[df["metric"] == metric_name, "value"]
    if row.empty:
        raise ValueError(f"{metrics_tsv} does not contain metric={metric_name!r}")
    val = row.iloc[0]
    try:
        return float(val)
    except Exception:
        return val


def get_scenario_name_from_metrics(metrics_tsv):
    df = pd.read_csv(metrics_tsv, sep="\t")
    row = df.loc[df["metric"] == "scenario", "value"]
    if row.empty:
        return None
    return str(row.iloc[0])


def infer_scenario_from_path(p):
    p2 = str(p).replace("\\", "/")
    parts = p2.split("/scenarios/")
    if len(parts) < 2:
        return None
    return parts[1].split("/")[0]


def load_boxplot_table(path):
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported boxplot table format: {path}")


def build_baseline_winner_df(df, baseline_name="base"):
    """
    Build one fixed baseline winner table, one row per pixel:
      - cheapest baseline tech by lcoe_base
    """
    req = {"row", "col", "tech", "lcoe_base", "potential_base"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(
            f"Baseline winner table cannot be built. Missing columns: {sorted(missing)}"
        )

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
    missing = req - set(df.columns)
    if missing:
        raise ValueError(
            f"Scenario winner table cannot be built. Missing columns: {sorted(missing)}"
        )

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


def format_pct_change(x):
    x = float(x)
    if np.isclose(x, round(x), atol=1e-8):
        return f"{x:+.0f}%"
    return f"{x:+.1f}%"


def make_parameter_box_label(param, pct_change):
    return f"{pretty_param_name(param, width=18)}\n({format_pct_change(pct_change)})"


def select_representative_scenarios_per_parameter(df, parameter_order):
    """
    Select at most one scenario per parameter for box plotting.

    Selection rule:
      - highest metric_value
      - then largest absolute pct_change
      - then scenario name for deterministic tie-breaking
    """
    selected_rows = []

    for param in parameter_order:
        g = df.loc[df["parameter"] == param].copy()
        if g.empty:
            continue

        g["abs_pct_change"] = g["pct_change"].abs()

        g = g.sort_values(
            ["metric_value", "abs_pct_change", "scenario"],
            ascending=[False, False, True],
        )

        selected_rows.append(g.iloc[0][[
            "parameter",
            "scenario",
            "pct_change",
            "pct_change_snapped",
            "metric_value",
            "transition_total_pixels",
        ]])

    if not selected_rows:
        return pd.DataFrame(columns=[
            "parameter",
            "scenario",
            "pct_change",
            "pct_change_snapped",
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

    if not data:
        raise ValueError(f"No valid data for {out_path}")

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
    exclude_params = set(snakemake.params.get("exclude_params", ["include_egs"]))

    scen_df = load_scenario_table(scenario_table)
    if baseline not in scen_df.index:
        raise ValueError(f"baseline={baseline!r} not found in {scenario_table}")

    base_row = scen_df.loc[baseline]

    scenario_to_metric = {}
    for p in metrics_files:
        scen_name = get_scenario_name_from_metrics(p) or infer_scenario_from_path(p)
        if scen_name is None:
            raise ValueError(f"Could not determine scenario name for {p}")
        scenario_to_metric[scen_name] = p

    numeric_cols = [
        c for c in scen_df.columns
        if c not in exclude_params and is_number(base_row.get(c, None))
    ]

    rows = []
    for scen_name, metrics_path in scenario_to_metric.items():
        if scen_name == baseline:
            continue
        if scen_name not in scen_df.index:
            continue

        row = scen_df.loc[scen_name]

        diffs = []
        for c in numeric_cols:
            try:
                bf = float(base_row[c])
                vf = float(row[c])
            except Exception:
                continue
            if not np.isclose(vf, bf, rtol=0, atol=0):
                diffs.append(c)

        if len(diffs) == 0:
            continue

        sens_val = read_metric_value(metrics_path, metric_name)
        pixels_val = read_metric_value(metrics_path, "switch_rate_pixels")

        for param in diffs:
            bf = float(base_row[param])
            vf = float(row[param])
            pct_change = (vf / bf - 1.0) * 100.0

            rows.append(
                dict(
                    scenario=scen_name,
                    parameter=param,
                    pct_change=pct_change,
                    metric_value=float(sens_val),
                    transition_total_pixels=float(pixels_val) if pd.notna(pixels_val) else np.nan,
                )
            )

    if not rows:
        raise ValueError("No numeric parameter scenarios found.")

    df = pd.DataFrame(rows)
    df["pct_change_snapped"] = df["pct_change"].round(6)

    df_agg = (
        df.groupby(["parameter", "pct_change_snapped"], as_index=False)
        .agg(metric_value=("metric_value", "mean"))
    )

    params = df_agg["parameter"].unique()
    zero_rows = pd.DataFrame({
        "parameter": params,
        "pct_change_snapped": 0.0,
        "metric_value": 0.0,
    })
    df_agg = pd.concat([df_agg, zero_rows], ignore_index=True)

    rank_params = (
        df_agg.groupby("parameter")["metric_value"]
        .max()
        .sort_values(ascending=False)
    )

    if not include_all:
        selected_params = rank_params.head(top_n).index.tolist()
        keep_metric = set(selected_params)
        df_plot = df_agg[df_agg["parameter"].isin(keep_metric)].copy()
    else:
        selected_params = rank_params.index.tolist()
        keep_metric = set(selected_params)
        df_plot = df_agg.copy()

    wide = df_agg.pivot_table(
        index="parameter",
        columns="pct_change_snapped",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()

    os.makedirs(os.path.dirname(snakemake.output["lineplot_table"]), exist_ok=True)
    wide.to_csv(snakemake.output["lineplot_table"], sep="\t", index=False)

    long_path = str(snakemake.output["lineplot_table"]).replace(".tsv", "_long.tsv")
    df_agg.sort_values(["parameter", "pct_change_snapped"]).to_csv(
        long_path, sep="\t", index=False
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.2)
    ax.minorticks_on()

    for param, g in df_plot.groupby("parameter"):
        g2 = (
            g[["pct_change_snapped", "metric_value"]]
            .drop_duplicates(subset=["pct_change_snapped"])
            .set_index("pct_change_snapped")
            .sort_index()
        )

        g2.loc[0.0] = 0.0
        g2 = g2.sort_index()

        ax.plot(
            g2.index.to_numpy(),
            g2["metric_value"].to_numpy(),
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=pretty_param_name(param),
        )

    ax.axvline(0, linewidth=1.0, alpha=0.8)
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Parameter change (%) vs baseline")
    ax.set_ylabel(metric_name)

    title_bits = [f"Sensitivity: {metric_name}"]
    if not include_all:
        title_bits.append(f"(top {top_n})")
    ax.set_title(" ".join(title_bits))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_alpha(0.9)

    os.makedirs(os.path.dirname(snakemake.output["lineplot_fig"]), exist_ok=True)
    fig.tight_layout()
    fig.savefig(snakemake.output["lineplot_fig"])
    plt.close(fig)

    # -----------------------------------
    # Boxplot selection: baseline once +
    # one representative scenario/parameter
    # -----------------------------------
    representative_df = select_representative_scenarios_per_parameter(
        df=df,
        parameter_order=selected_params,
    )

    if representative_df.empty:
        raise ValueError("No representative scenarios selected for boxplot generation.")

    scenario_to_boxplot = {}
    for p in boxplot_files:
        scen_name = infer_scenario_from_path(p)
        if scen_name is None:
            raise ValueError(f"Could not determine scenario name from boxplot path: {p}")
        scenario_to_boxplot[scen_name] = p

    if baseline not in scenario_to_boxplot:
        raise ValueError(
            f"Baseline scenario {baseline!r} not found among boxplot files. "
            f"Available: {sorted(scenario_to_boxplot)}"
        )

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
        pct_change = float(row["pct_change"])

        if scen_name not in scenario_to_boxplot:
            raise ValueError(
                f"Selected scenario {scen_name!r} for parameter {param!r} "
                f"does not have a matching boxplot file."
            )

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
                "pct_change": pct_change,
                "label": make_parameter_box_label(param, pct_change),
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


if __name__ == "__main__":
    main()