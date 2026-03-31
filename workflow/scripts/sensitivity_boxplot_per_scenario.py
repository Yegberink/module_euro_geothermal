import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
})


TECH_ORDER = ["orc", "flash", "kalina"]
TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}

STATE_COLORS = {
    "Baseline": "#BDBDBD",
    "Scenario": "#1F78B4",
}


def load_boxplot_table(path):
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def make_grouped_boxplot(ax, df, value_base, value_scen, ylabel, title):
    data = []
    positions = []
    labels = []
    pos = 1

    for tech in TECH_ORDER:
        dft = df[df["tech"].str.lower() == tech].copy()
        if dft.empty:
            pos += 3
            continue

        vals_base = dft[value_base].to_numpy()
        vals_base = vals_base[np.isfinite(vals_base)]

        vals_scen = dft[value_scen].to_numpy()
        vals_scen = vals_scen[np.isfinite(vals_scen)]

        if vals_base.size > 0:
            data.append(vals_base)
            positions.append(pos)
            labels.append((TECH_LABELS[tech], "Baseline"))
            pos += 1

        if vals_scen.size > 0:
            data.append(vals_scen)
            positions.append(pos)
            labels.append((TECH_LABELS[tech], "Scenario"))
            pos += 1

        pos += 1  # gap between technologies

    if not data:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
    )

    for patch, (_, state) in zip(bp["boxes"], labels):
        patch.set_facecolor(STATE_COLORS[state])
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)

    for element in ["whiskers", "caps", "medians"]:
        for artist in bp[element]:
            artist.set_linewidth(0.8)

    ticklabels = [f"{tech}\n{state}" for tech, state in labels]
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    inpath = snakemake.input["boxplot_data"]
    outfig = snakemake.output["fig"]

    df = load_boxplot_table(inpath)

    req = {
        "scenario", "baseline", "tech",
        "lcoe_base", "lcoe_scen",
        "potential_base", "potential_scen",
    }
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    scenario = str(df["scenario"].iloc[0]) if "scenario" in df.columns else "scenario"
    baseline = str(df["baseline"].iloc[0]) if "baseline" in df.columns else "baseline"

    fig, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)

    make_grouped_boxplot(
        axes[0],
        df=df,
        value_base="lcoe_base",
        value_scen="lcoe_scen",
        ylabel="LCOE",
        title="LCOE distribution by technology",
    )

    make_grouped_boxplot(
        axes[1],
        df=df,
        value_base="potential_base",
        value_scen="potential_scen",
        ylabel="Potential",
        title="Potential distribution by technology",
    )

    fig.suptitle(
        f"Sensitivity boxplots — scenario '{scenario}' vs baseline '{baseline}'",
        fontsize=14,
    )

    os.makedirs(os.path.dirname(outfig), exist_ok=True)
    fig.savefig(outfig)
    plt.close(fig)


if __name__ == "__main__":
    main()