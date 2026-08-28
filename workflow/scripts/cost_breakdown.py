"""Plot geographic distributions of LCOE component shares."""

from pathlib import Path
import sys

import _schemas
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

plt.switch_backend("Agg")


COMPONENTS = (
    ("drilling", "Well drilling"),
    ("stimulation", "EGS stimulation"),
    ("pump", "Pumps"),
    ("power_plant", "Power plant"),
    ("fixed_om", "Fixed O&M"),
)
TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
TECH_COLORS = {"orc": "#0072B2", "flash": "#E69F00", "kalina": "#009E73"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)


def _cycle_from_cube(path):
    return Path(path).name.replace("_lcoe_cube.nc", "").lower()


def _geographic_mask(shapes_path, reference, crs):
    shapes = gpd.read_parquet(shapes_path)
    if "shape_class" in shapes.columns:
        shapes = shapes.loc[shapes["shape_class"] == "land"].copy()
    shapes = shapes.loc[shapes.geometry.notna() & ~shapes.geometry.is_empty].copy()

    raster = reference.transpose("y", "x").rio.set_spatial_dims(x_dim="x", y_dim="y")
    if crs:
        raster = raster.rio.write_crs(crs, inplace=False)
    if raster.rio.crs is not None and shapes.crs != raster.rio.crs:
        shapes = shapes.to_crs(raster.rio.crs)

    return geometry_mask(
        shapes.geometry,
        out_shape=(raster.sizes["y"], raster.sizes["x"]),
        transform=raster.rio.transform(),
        invert=True,
    )


def component_shares_for_cube(cube_path, shapes_path):
    """Return cell-level LCOE component shares at each cell's optimal depth."""
    with xr.open_dataset(cube_path) as ds:
        required = ["lcoe", *(f"lcoe_{name}" for name, _ in COMPONENTS)]
        missing = [name for name in required if name not in ds]
        if missing:
            raise ValueError(
                f"{cube_path} lacks cost variables {missing}; regenerate the LCOE cube."
            )

        lcoe = ds["lcoe"]
        argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim="depth_m")
        valid_depth = np.isfinite(lcoe.min(dim="depth_m", skipna=True))
        total = lcoe.isel(depth_m=argmin).where(valid_depth).load()
        inside = _geographic_mask(
            shapes_path,
            total,
            ds.attrs.get("crs") or total.attrs.get("crs"),
        )

        shares = {}
        valid_total = inside & np.isfinite(total) & (total > 0)
        for component, _ in COMPONENTS:
            contribution = ds[f"lcoe_{component}"].isel(depth_m=argmin).where(valid_depth).load()
            shares[component] = (100.0 * contribution / total).where(valid_total)

    return xr.Dataset(shares).expand_dims(technology=[_cycle_from_cube(cube_path)])


def summarize_technology_shares(technology_shares):
    """Return the mean and full geographic range per technology and component."""
    summaries = {}
    for dataset in technology_shares:
        technology = str(dataset["technology"].values[0])
        summaries[technology] = {}
        for component, _ in COMPONENTS:
            if component not in dataset:
                continue
            values = np.asarray(dataset[component].values, dtype=float).ravel()
            values = values[np.isfinite(values)]
            if values.size:
                summaries[technology][component] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
    return summaries


def plot_cost_breakdown(ax, technology_shares, *, compact=False):
    """Draw technology-specific cost-component boxplots on an existing axis."""
    summaries = summarize_technology_shares(technology_shares)
    technologies = [technology for technology in TECH_LABELS if technology in summaries]
    technologies.extend(
        technology
        for technology in summaries
        if technology not in technologies
    )
    if not technologies:
        ax.axis("off")
        return False

    positions = np.arange(1, len(COMPONENTS) + 1)
    offsets = np.linspace(-0.22, 0.22, len(technologies)) if len(technologies) > 1 else [0.0]
    rng = np.random.default_rng(42)

    for technology, offset in zip(technologies, offsets):
        color = TECH_COLORS.get(technology, "#4D4D4D")
        for position, (component, _) in zip(positions, COMPONENTS):
            stats = summaries[technology].get(component)
            if stats is None:
                continue
            x = position + offset
            boxplot = ax.boxplot(
                [stats["values"]],
                positions=[x],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                zorder=2,
            )
            boxplot["boxes"][0].set_facecolor(color)
            boxplot["boxes"][0].set_alpha(0.58)
            boxplot["boxes"][0].set_edgecolor("#707070")
            boxplot["boxes"][0].set_linewidth(1.0)
            for element in ("whiskers", "caps", "medians"):
                for item in boxplot[element]:
                    item.set_color("#707070" if element != "medians" else "#202020")
                    item.set_linewidth(1.0)

            sample_size = min(700, stats["values"].size)
            sample = rng.choice(stats["values"], size=sample_size, replace=False)
            jitter = rng.uniform(-0.07, 0.07, size=sample_size)
            ax.scatter(
                x + jitter,
                sample,
                s=3,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=1,
            )
            ax.scatter(
                x,
                stats["mean"],
                s=15,
                marker="o",
                facecolor="white",
                edgecolor=color,
                linewidth=1.2 if compact else 1.5,
                zorder=4,
            )

    ax.set_xticks(positions, [label for _, label in COMPONENTS])
    if compact:
        ax.set_xticklabels(
            [label for _, label in COMPONENTS],
            rotation=24,
            ha="right",
            fontsize=9.5,
        )
    ax.set_xlim(0.5, len(COMPONENTS) + 0.5)
    ax.set_ylabel("Contribution to LCOE (%)")
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
    ax.legend(
        handles=[
            Patch(
                facecolor=TECH_COLORS.get(technology, "#4D4D4D"),
                edgecolor="#707070",
                alpha=0.58,
                label=TECH_LABELS.get(technology, technology.upper()),
            )
            for technology in technologies
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#202020",
                markeredgewidth=1.2,
                markersize=5.5 if compact else 7,
                label="Mean",
            )
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#BBBBBB",
        title="Technology",
        fontsize=9 if compact else 13,
        title_fontsize=9 if compact else 13,
    )
    if compact:
        ax.set_ylabel("Contribution to LCOE (%)", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
    return True


def make_figure(technology_shares, output_path):
    """Plot geographic boxplots and means by technology and component."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.27, 5.8), constrained_layout=True)
    if not plot_cost_breakdown(ax, technology_shares):
        plt.close(fig)
        _schemas.write_blank_png(output_path, figsize=(8.6, 5.6))
        return
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Write technology-specific geographic cost-component boxplots."""
    shares = [
        component_shares_for_cube(cube, snakemake.input["shapes"])
        for cube in snakemake.input["cubes"]
    ]
    make_figure(shares, snakemake.output["fig"])


if __name__ == "__main__":
    main()
