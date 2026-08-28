"""Summarize ORC potential by reservoir-temperature bin and scenario."""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from matplotlib.patches import Rectangle
from rasterio.features import geometry_mask

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 10,
    }
)


PARAMETER_LABELS = {
    "max_depth": "Maximum depth",
    "EGS_cutoff": "EGS cutoff",
    "drilling_cost_curve": "Drilling cost curve",
}


def scenario_label(row: pd.Series, baseline: str) -> str:
    """Build a concise display label for one scenario-table row."""
    if row["scenario"] == baseline:
        return "Baseline"
    parameter = row.get("sensitivity_parameter")
    value = row.get("sensitivity_scenario_value")
    label = PARAMETER_LABELS.get(parameter, str(parameter).replace("_", " "))
    if parameter in {"max_depth", "EGS_cutoff"} and pd.notna(value):
        value = f"{float(value) / 1000:g} km"
    return f"{label}: {value}"


def land_mask(data, shapes_path: str) -> np.ndarray:
    """Return a raster mask covering included land polygons."""
    shapes = gpd.read_parquet(shapes_path)
    if "shape_class" in shapes.columns:
        shapes = shapes.loc[shapes["shape_class"] == "land"].copy()
    if data.rio.crs is not None and shapes.crs != data.rio.crs:
        shapes = shapes.to_crs(data.rio.crs)
    return geometry_mask(
        shapes.geometry,
        out_shape=(data.sizes["y"], data.sizes["x"]),
        transform=data.rio.transform(),
        invert=True,
    )


scenario_names = list(snakemake.params.scenarios)
bin_width = float(snakemake.params.bin_width)
scenario_table = pd.read_csv(snakemake.input.scenario_table, sep="\t", comment="#")
scenario_table = scenario_table.set_index("scenario").loc[scenario_names].reset_index()
baseline = scenario_names[0]

values_by_scenario = []
all_temperatures = []
for scenario, temperature_path, potential_path in zip(
    scenario_names, snakemake.input.temperatures, snakemake.input.potentials
):
    temperature = (
        rioxarray.open_rasterio(temperature_path).squeeze(drop=True).transpose("y", "x")
    )
    potential = (
        rioxarray.open_rasterio(potential_path).squeeze(drop=True).transpose("y", "x")
    )
    included = land_mask(temperature, snakemake.input.shapes)
    temp_values = np.asarray(temperature.values, dtype=np.float64)
    pot_values = np.asarray(potential.values, dtype=np.float64)
    valid = (
        included & np.isfinite(temp_values) & np.isfinite(pot_values) & (pot_values > 0)
    )
    temp_values = temp_values[valid]
    pot_values = pot_values[valid]
    values_by_scenario.append((scenario, temp_values, pot_values))
    if temp_values.size:
        all_temperatures.append(temp_values)

if not all_temperatures:
    raise ValueError("No finite ORC temperature and potential values were found")

temperature_values = np.concatenate(all_temperatures)
first_edge = np.floor(np.nanmin(temperature_values) / bin_width) * bin_width
last_edge = np.ceil(np.nanmax(temperature_values) / bin_width) * bin_width
edges = np.arange(first_edge, last_edge + bin_width, bin_width)
if edges.size < 2:
    edges = np.array([first_edge, first_edge + bin_width])

shares = np.zeros((len(scenario_names), len(edges) - 1), dtype=np.float64)
records = []
for row, (scenario, temperatures, potentials) in enumerate(values_by_scenario):
    totals, _ = np.histogram(temperatures, bins=edges, weights=potentials)
    total_potential = totals.sum()
    if total_potential > 0:
        shares[row] = totals / total_potential
    dominant = int(np.argmax(shares[row]))
    for col, share in enumerate(shares[row]):
        records.append(
            {
                "scenario": scenario,
                "temperature_bin_min_c": edges[col],
                "temperature_bin_max_c": edges[col + 1],
                "potential_share": share,
                "is_largest_share": col == dominant,
            }
        )

Path(snakemake.output.data).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame.from_records(records).to_csv(snakemake.output.data, sep="\t", index=False)

labels = [scenario_label(row, baseline) for _, row in scenario_table.iterrows()]
fig_height = max(5.2, 0.55 * len(labels) + 1.8)
fig, ax = plt.subplots(figsize=(13.333, fig_height), constrained_layout=True)
image = ax.imshow(
    shares, aspect="auto", cmap="viridis", vmin=0, vmax=max(shares.max(), 0.01)
)

for row in range(shares.shape[0]):
    dominant = int(np.argmax(shares[row]))
    ax.add_patch(
        Rectangle(
            (dominant - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor="white",
            linewidth=2.0,
        )
    )
    ax.text(
        dominant,
        row,
        f"{shares[row, dominant]:.0%}",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )

centers = 0.5 * (edges[:-1] + edges[1:])
ax.set_xticks(np.arange(len(centers)))
ax.set_xticklabels(
    [f"{edge:g}-{edge + bin_width:g}" for edge in edges[:-1]], rotation=45, ha="right"
)
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Temperature bin at minimum LCOE (°C)")
ax.set_ylabel("Scenario")
ax.set_title("Distribution of ORC potential across 10 °C temperature bins")
colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Share of total ORC potential")
colorbar.ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")

fig.savefig(snakemake.output.fig, dpi=200)
plt.close(fig)
