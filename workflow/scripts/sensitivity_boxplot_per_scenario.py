import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import _schemas

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

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
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.fontsize": 10,
    "legend.frameon": False,
})

LCOE_LABEL = r"LCOE (€/MWh$_e$)"
POTENTIAL_LABEL = "Theoretical potential(GW)"
MW_PER_GW = 1000.0


TECH_ORDER = ["orc", "flash", "kalina"]
TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
SCENARIO_LABELS = {
    "EGS_cutoff": "EGS cutoff",
    "stimulation_cost_per_doublet": "Stimulation cost",
    "pump_cost": "Pump cost",
    "lifetime": "Lifetime",
    "capacity_factor": "Capacity factor",
    "fixed_OM_rate": "Fixed O&M rate",
    "parasitic_fraction": "Parasitic fraction",
    "orc_money_per_kw": "ORC CAPEX",
    "flash_money_per_kw": "Flash CAPEX",
    "kalina_money_per_kw": "Kalina CAPEX",
    "drilling_cost_curve": "Drilling cost curve",
    "include_egs": "Include EGS",
    "SED_NAN_POLICY": "Sediment NaN policy",
    "max_depth": "Max depth",
    "well_spacing": "Well spacing",
    "flowrate_egs": "EGS flow rate",
    "flowrate_conventional": "Conventional flow rate",
    "temp_turbine_out_ORC": "ORC outlet temp",
    "temp_turbine_out_Flash": "Flash outlet temp",
    "temp_turbine_out_Kalina": "Kalina outlet temp",
    "min_dt": "Minimum dT",
}

STATE_COLORS = {
    "Baseline": "#9E9E9E",
    "Scenario": "#2166AC",
}

BASELINE_DUAL_COLORS = {
    "LCOE": "#6F6F6F",
    "Potential": "#2166AC",
}


def load_boxplot_table(path):
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_scenario_table(path):
    df = pd.read_csv(path, sep="\t", comment="#").replace({np.nan: None})
    df["scenario"] = df["scenario"].astype(str)
    return df.set_index("scenario")


def pretty_scenario_label(scen_df, scenario_name, baseline_name):
    if scenario_name == baseline_name:
        return "Baseline"
    row = scen_df.loc[scenario_name]
    param = row.get("sensitivity_parameter")
    base_value = row.get("sensitivity_base_value")
    scenario_value = row.get("sensitivity_scenario_value")
    if param is None or pd.isna(param):
        return scenario_name
    param_label = SCENARIO_LABELS.get(param, str(param).replace("_", " "))
    return f"{param_label} ({base_value} -> {scenario_value})"


def style_axes(ax, add_grid=True):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    if add_grid:
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def add_top_padding(ax, frac=0.18):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if not np.isfinite(yrange) or yrange <= 0:
        yrange = max(abs(ymax), 1.0)
    ax.set_ylim(ymin, ymax + frac * yrange)


def draw_boxplot(
    ax,
    data,
    positions,
    colors,
    widths=0.65,
    zorder=3,
):
    """
    Draw a styled boxplot and return the matplotlib boxplot dictionary.
    """
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(color="#4D4D4D", linewidth=0.9),
        capprops=dict(color="#4D4D4D", linewidth=0.9),
        boxprops=dict(edgecolor="#4D4D4D", linewidth=0.9),
        zorder=zorder,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    return bp


def make_metric_comparison_boxplot(
    df,
    value_base,
    value_scen,
    ylabel,
    scenario_label,
    baseline_label,
    outfig,
    value_scale=1.0,
):
    """
    For non-baseline scenarios:
    create a single-metric figure comparing Baseline vs Scenario for each technology.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)

    data = []
    positions = []
    colors = []
    xticks = []
    xticklabels = []

    pos = 1.0
    tech_centers = []

    for tech in TECH_ORDER:
        dft = df[df["tech"].str.lower() == tech].copy()
        if dft.empty:
            pos += 2.4
            continue

        vals_base = dft[value_base].to_numpy() / value_scale
        vals_base = vals_base[np.isfinite(vals_base)]

        vals_scen = dft[value_scen].to_numpy() / value_scale
        vals_scen = vals_scen[np.isfinite(vals_scen)]

        this_positions = []

        if vals_base.size > 0:
            data.append(vals_base)
            positions.append(pos)
            colors.append(STATE_COLORS["Baseline"])
            xticks.append(pos)
            xticklabels.append("Base")
            this_positions.append(pos)
            pos += 0.8

        if vals_scen.size > 0:
            data.append(vals_scen)
            positions.append(pos)
            colors.append(STATE_COLORS["Scenario"])
            xticks.append(pos)
            xticklabels.append("Scenario")
            this_positions.append(pos)
            pos += 0.8

        if this_positions:
            tech_centers.append((np.mean(this_positions), TECH_LABELS[tech]))

        pos += 0.8

    if not data:
        plt.close(fig)
        _schemas.write_blank_png(outfig)
        return

    draw_boxplot(ax, data=data, positions=positions, colors=colors)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)

    # Technology labels under each pair
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    for center, tech_label in tech_centers:
        ax.text(
            center,
            ymin - 0.10 * yrange,
            tech_label,
            ha="center",
            va="top",
            fontsize=11,
            clip_on=False,
        )

    style_axes(ax, add_grid=False)
    add_top_padding(ax)

    legend_handles = [
        Patch(facecolor=STATE_COLORS["Baseline"], edgecolor="#4D4D4D", label=baseline_label),
        Patch(facecolor=STATE_COLORS["Scenario"], edgecolor="#4D4D4D", label=f"Scenario: {scenario_label}"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.savefig(outfig, bbox_inches="tight")
    plt.close(fig)


def make_baseline_dual_axis_boxplot(
    df,
    scenario_label,
    is_baseline_case,
    outfig,
):
    """
    For the baseline scenario:
    create one figure with twin y-axes:
      - LCOE boxplots on left axis
      - Potential boxplots on right axis
    One LCOE and one Potential boxplot per technology.
    """
    fig, ax1 = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax2 = ax1.twinx()

    data_lcoe = []
    pos_lcoe = []
    colors_lcoe = []

    data_potential = []
    pos_potential = []
    colors_potential = []

    xtick_positions = []
    xtick_labels = []

    base_pos = 1.0
    for tech in TECH_ORDER:
        dft = df[df["tech"].str.lower() == tech].copy()
        if dft.empty:
            base_pos += 2.0
            continue

        lcoe_col = "lcoe_base" if is_baseline_case else "lcoe_scen"
        potential_col = "potential_base" if is_baseline_case else "potential_scen"

        vals_lcoe = dft[lcoe_col].to_numpy()
        vals_lcoe = vals_lcoe[np.isfinite(vals_lcoe)]

        vals_potential = dft[potential_col].to_numpy() / MW_PER_GW
        vals_potential = vals_potential[np.isfinite(vals_potential)]

        center = base_pos
        x_lcoe = center - 0.22
        x_potential = center + 0.22

        if vals_lcoe.size > 0:
            data_lcoe.append(vals_lcoe)
            pos_lcoe.append(x_lcoe)
            colors_lcoe.append(BASELINE_DUAL_COLORS["LCOE"])

        if vals_potential.size > 0:
            data_potential.append(vals_potential)
            pos_potential.append(x_potential)
            colors_potential.append(BASELINE_DUAL_COLORS["Potential"])

        xtick_positions.append(center)
        xtick_labels.append(TECH_LABELS[tech])

        base_pos += 1.6

    if data_lcoe:
        draw_boxplot(
            ax1,
            data=data_lcoe,
            positions=pos_lcoe,
            colors=colors_lcoe,
            widths=0.36,
            zorder=3,
        )

    if data_potential:
        draw_boxplot(
            ax2,
            data=data_potential,
            positions=pos_potential,
            colors=colors_potential,
            widths=0.36,
            zorder=2,
        )

    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_ylabel(LCOE_LABEL)
    ax2.set_ylabel(POTENTIAL_LABEL)

    style_axes(ax1, add_grid=False)
    ax2.spines["top"].set_visible(True)
    add_top_padding(ax1)
    add_top_padding(ax2)

    legend_handles = [
        Patch(facecolor=BASELINE_DUAL_COLORS["LCOE"], edgecolor="#4D4D4D", label="LCOE"),
        Patch(facecolor=BASELINE_DUAL_COLORS["Potential"], edgecolor="#4D4D4D", label="Potential"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left")

    fig.savefig(outfig, bbox_inches="tight")
    plt.close(fig)


def main():
    scen_df = load_scenario_table(snakemake.input["scenario_table"])
    inpath = snakemake.input["boxplot_data"]
    df = _schemas.SensitivityBoxplotSchema.validate(load_boxplot_table(inpath))
    if df.empty:
        _schemas.write_blank_png(snakemake.output["base_fig"])
        _schemas.write_blank_png(snakemake.output["lcoe_fig"])
        _schemas.write_blank_png(snakemake.output["potential_fig"])
        return

    df = df.copy()
    df["tech"] = df["tech"].astype(str)

    scenario = str(df["scenario"].iloc[0])
    baseline = str(df["baseline"].iloc[0])
    is_baseline_case = scenario == baseline
    baseline_label = pretty_scenario_label(scen_df, baseline, baseline)
    scenario_label = pretty_scenario_label(scen_df, scenario, baseline)
    make_baseline_dual_axis_boxplot(
        df=df,
        scenario_label=scenario_label,
        is_baseline_case=is_baseline_case,
        outfig=snakemake.output["base_fig"],
    )

    make_metric_comparison_boxplot(
        df=df,
        value_base="lcoe_base",
        value_scen="lcoe_scen",
        ylabel=LCOE_LABEL,
        scenario_label=scenario_label,
        baseline_label=baseline_label,
        outfig=snakemake.output["lcoe_fig"],
        value_scale=1.0,
    )

    make_metric_comparison_boxplot(
        df=df,
        value_base="potential_base",
        value_scen="potential_scen",
        ylabel=POTENTIAL_LABEL,
        scenario_label=scenario_label,
        baseline_label=baseline_label,
        outfig=snakemake.output["potential_fig"],
        value_scale=MW_PER_GW,
    )


if __name__ == "__main__":
    main()
