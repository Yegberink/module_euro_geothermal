"""All figure rendering used by the geothermal workflow.

Computation scripts prepare arrays and tables, then call the functions in this
module. Keeping the rendering here makes the set of workflow figures explicit
and prevents plotting dependencies from leaking into calculation code.
"""

from pathlib import Path

import _schemas
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns

TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
TECH_COLORS = {"orc": "#0072B2", "flash": "#E69F00", "kalina": "#009E73"}
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
COMPONENTS = (
    ("drilling", "Well drilling"),
    ("stimulation", "EGS stimulation"),
    ("pump", "Pumps"),
    ("power_plant", "Power plant"),
    ("fixed_om", "Fixed O&M"),
)

LCOE_CMAP = "YlOrBr_r"
LCOE_COLORBAR_CUTOFF = 600.0
LCOE_NORM_GAMMA = 0.4
MAP_BACKGROUND_COLOR = "#DCEEF4"
UNAVAILABLE_COLOR = "#D9D9D9"
LCOE_UNAVAILABLE_COLOR = "#BDBDBD"
COUNTRY_EDGE_COLOR = "#4D5960"
UNCHANGED_COLOR = "#FFFFFF"

BASE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}

SENSITIVITY_STYLE = {
    **BASE_STYLE,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
}


def _ensure_parent(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_blank_png(path, figsize=(6, 4)) -> None:
    """Write a valid empty placeholder when no plottable data are available."""
    path = _ensure_parent(path)
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _map_extent(da: xr.DataArray) -> list[float]:
    raster = da.transpose("y", "x")
    x = raster["x"].values
    y = raster["y"].values
    return [float(x.min()), float(x.max()), float(y.min()), float(y.max())]


def _map_crs(da: xr.DataArray):
    try:
        return da.rio.crs
    except (AttributeError, ValueError):
        return None


def _country_layer(shapes_path, reference_da: xr.DataArray) -> gpd.GeoDataFrame:
    countries = gpd.read_parquet(shapes_path)
    if "shape_class" in countries.columns:
        countries = countries.loc[countries["shape_class"] == "land"].copy()
    countries = countries.loc[
        countries.geometry.notna() & ~countries.geometry.is_empty
    ].copy()
    target_crs = _map_crs(reference_da)
    if target_crs is not None and countries.crs != target_crs:
        countries = countries.to_crs(target_crs)
    return countries


def _plot_values(
    da: xr.DataArray, countries: gpd.GeoDataFrame, value_scale: float = 1.0
) -> np.ndarray:
    raster = da.transpose("y", "x")
    inside = geometry_mask(
        countries.geometry,
        out_shape=(raster.sizes["y"], raster.sizes["x"]),
        transform=raster.rio.transform(),
        invert=True,
    )
    values = np.asarray(raster.values, dtype=np.float32) / value_scale
    return np.where(inside, values, np.nan)


def _decorate_map(
    ax,
    countries: gpd.GeoDataFrame,
    extent: list[float],
    unavailable_color: str = UNAVAILABLE_COLOR,
):
    ax.set_facecolor(MAP_BACKGROUND_COLOR)
    countries.plot(ax=ax, facecolor=unavailable_color, edgecolor="none", zorder=1)
    countries.boundary.plot(ax=ax, color=COUNTRY_EDGE_COLOR, linewidth=0.35, zorder=3)
    ax.annotate(
        "N",
        xy=(0.94, 0.91),
        xytext=(0.94, 0.78),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#202020", lw=1.2),
        zorder=5,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def _clean_map_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#666666")


def _finite_limits(arrays) -> tuple[float, float]:
    finite = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    if not finite:
        return np.nan, np.nan
    return (
        float(min(np.nanmin(a) for a in finite)),
        float(max(np.nanmax(a) for a in finite)),
    )


def _winner_map(lcoe_arrays: dict[str, np.ndarray], cycles: list[str]) -> np.ndarray:
    stack = np.stack([lcoe_arrays[cycle] for cycle in cycles], axis=0)
    finite = np.isfinite(stack)
    winner = np.argmin(np.where(finite, stack, np.inf), axis=0).astype(np.int16)
    return np.where(np.any(finite, axis=0), winner, -1)


def make_lcoe_overview(
    lcoe_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path,
    shapes_path,
    max_depth: float,
):
    """Plot cycle-specific LCOE maps and the least-cost technology."""
    if not cycles:
        write_blank_png(out_path)
        return

    reference = lcoe_maps[cycles[0]]
    countries = _country_layer(shapes_path, reference)
    if countries.empty:
        write_blank_png(out_path)
        return

    panels = cycles + ["winner"]
    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols))
    extent = _map_extent(reference)
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    panel_w = 4.5
    panel_h = panel_w * data_aspect
    gap = 0.15
    margin_l = margin_r = margin_t = 0.1
    margin_b = 0.9
    fig_w = margin_l + ncols * panel_w + (ncols - 1) * gap + margin_r
    fig_h = margin_t + nrows * panel_h + (nrows - 1) * gap + margin_b

    with plt.rc_context(BASE_STYLE):
        fig = plt.figure(figsize=(fig_w, fig_h))

        def make_ax(row, col):
            x0 = (margin_l + col * (panel_w + gap)) / fig_w
            y0 = (margin_b + (nrows - 1 - row) * (panel_h + gap)) / fig_h
            return fig.add_axes([x0, y0, panel_w / fig_w, panel_h / fig_h])

        axes = [make_ax(*divmod(i, ncols)) for i in range(len(panels))]
        lcoe_cmap = plt.get_cmap(LCOE_CMAP).copy()
        lcoe_cmap.set_bad((0, 0, 0, 0))
        winner_cmap = ListedColormap([TECH_COLORS[c] for c in cycles])
        winner_cmap.set_bad((0, 0, 0, 0))

        plot_arrays = {
            cycle: _plot_values(lcoe_maps[cycle], countries) for cycle in cycles
        }
        lcoe_vmin, data_vmax = _finite_limits(plot_arrays.values())
        if not np.isfinite(lcoe_vmin) or not np.isfinite(data_vmax):
            plt.close(fig)
            write_blank_png(out_path)
            return
        lcoe_vmax = min(data_vmax, LCOE_COLORBAR_CUTOFF)
        lcoe_norm = PowerNorm(gamma=LCOE_NORM_GAMMA, vmin=70, vmax=LCOE_COLORBAR_CUTOFF)
        colorbar_extend = "max" if data_vmax > lcoe_vmax else "neither"
        winner = _winner_map(plot_arrays, cycles)

        for idx, panel in enumerate(panels):
            ax = axes[idx]
            _decorate_map(
                ax, countries, extent, unavailable_color=LCOE_UNAVAILABLE_COLOR
            )
            if panel == "winner":
                ax.imshow(
                    np.where(winner >= 0, winner, np.nan),
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    cmap=winner_cmap,
                    vmin=-0.5,
                    vmax=len(cycles) - 0.5,
                    aspect="auto",
                    zorder=2,
                )
                title = "Cheapest technology"
            else:
                ax.imshow(
                    plot_arrays[panel],
                    origin="lower",
                    extent=extent,
                    interpolation="nearest",
                    cmap=lcoe_cmap,
                    norm=lcoe_norm,
                    aspect="auto",
                    zorder=2,
                )
                title = TECH_LABELS.get(panel, panel.upper())
            ax.text(
                0.02,
                0.98,
                f"{PANEL_LETTERS[idx]}) {title}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
            _clean_map_axes(ax)

        cbar_x0 = (margin_l + 0.1 * fig_w) / fig_w
        cbar_y0 = 0.38 / fig_h
        cbar_w = 1 - cbar_x0 - (margin_r + 0.1 * fig_w) / fig_w
        cax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, 0.04 / fig_h * fig_w])
        sm = plt.cm.ScalarMappable(norm=lcoe_norm, cmap=lcoe_cmap)
        cbar = fig.colorbar(
            sm, cax=cax, orientation="horizontal", extend=colorbar_extend
        )
        cbar.set_label("Levelized Cost of Electricity (€/MWh$_e$)")
        cbar.ax.tick_params(length=0, labelsize=11)

        handles = [
            Patch(
                facecolor=TECH_COLORS[c],
                edgecolor="none",
                label=TECH_LABELS.get(c, c.upper()),
            )
            for c in cycles
        ]
        depth_label = (
            f"{max_depth / 1000:g} km" if max_depth >= 1000 else f"{max_depth:g} m"
        )
        handles.append(
            Patch(
                facecolor=LCOE_UNAVAILABLE_COLOR,
                edgecolor=COUNTRY_EDGE_COLOR,
                label=f"Not possible below {depth_label}",
            )
        )
        axes[-1].legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="#AAAAAA",
            fontsize=10,
        )
        fig.savefig(_ensure_parent(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)


def _cumulative_potential_curve(
    sort_da: xr.DataArray, potential_da: xr.DataArray, *, descending: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sort_da.values, dtype=np.float64).ravel()
    potential = np.asarray(potential_da.values, dtype=np.float64).ravel() / 1000.0
    valid = np.isfinite(x) & np.isfinite(potential) & (potential > 0)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(x[valid], kind="mergesort")
    if descending:
        order = order[::-1]
    return x[valid][order], np.cumsum(potential[valid][order])


def make_lcoe_potential_cumulative_four_panel(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    temperature_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path,
    *,
    max_scatter_points_per_tech: int = 700,
):
    """Plot LCOE/potential distributions and cumulative-potential curves."""
    lcoe_data = []
    potential_data = []
    labels = []
    colors = []
    scatter_rows = []
    rng = np.random.default_rng(42)

    for cycle in cycles:
        label = TECH_LABELS.get(cycle, cycle.upper())
        color = TECH_COLORS.get(cycle, "#333333")
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=float).ravel()
        lcoe = lcoe[np.isfinite(lcoe)]
        potential = np.asarray(potential_maps[cycle].values, dtype=float).ravel() / 1000
        potential = potential[np.isfinite(potential) & (potential > 0)]
        if not lcoe.size or not potential.size:
            continue
        lcoe_data.append(lcoe)
        potential_data.append(potential)
        labels.append(label)
        colors.append(color)
        for metric, values in (("LCOE", lcoe), ("Potential", potential)):
            sample_size = min(max_scatter_points_per_tech, values.size)
            sample = rng.choice(values.size, size=sample_size, replace=False)
            scatter_rows.extend(
                {"technology": label, "metric": metric, "value": value}
                for value in values[sample]
            )

    if not labels:
        write_blank_png(out_path, figsize=(8.27, 8.27))
        return

    scatter_df = pd.DataFrame(scatter_rows)
    palette = dict(zip(labels, colors))
    with plt.rc_context(BASE_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.27))
        ax_lcoe_box, ax_potential_box = axes[0]
        ax_lcoe_curve, ax_temp_curve = axes[1]

        def add_panel_letter(ax, label):
            ax.text(
                0.03,
                0.95,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2),
                zorder=6,
            )

        box_specs = [
            (ax_lcoe_box, "LCOE", lcoe_data, "LCOE (€/MWh$_e$)", "a)"),
            (
                ax_potential_box,
                "Potential",
                potential_data,
                "Theoretical potential (GW per grid cell)",
                "b)",
            ),
        ]
        for ax, metric, data, ylabel, panel_label in box_specs:
            bp = ax.boxplot(
                data,
                positions=np.arange(len(labels)),
                widths=0.48,
                patch_artist=True,
                showfliers=False,
            )
            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.58)
                box.set_edgecolor("#707070")
            for element in ("whiskers", "caps", "medians"):
                for item in bp[element]:
                    item.set_color("#707070" if element != "medians" else "#202020")
            sns.stripplot(
                data=scatter_df.loc[scatter_df["metric"] == metric],
                x="technology",
                y="value",
                order=labels,
                palette=palette,
                jitter=0.22,
                size=1.5,
                alpha=0.20,
                linewidth=0,
                ax=ax,
            )
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.grid(False)
            add_panel_letter(ax, panel_label)

        curve_specs = [
            (
                ax_lcoe_curve,
                lcoe_maps,
                "LCOE threshold (€/MWh$_e$)",
                False,
                "c)",
                "lower right",
                True,
            ),
            (
                ax_temp_curve,
                temperature_maps,
                "Temperature threshold (°C)",
                True,
                "d)",
                "upper right",
                False,
            ),
        ]
        for ax, x_maps, xlabel, descending, panel, legend_loc, annotate in curve_specs:
            for cycle in cycles:
                x, y = _cumulative_potential_curve(
                    x_maps[cycle], potential_maps[cycle], descending=descending
                )
                if not x.size:
                    continue
                ax.plot(
                    x,
                    y,
                    color=TECH_COLORS.get(cycle, "#333333"),
                    linewidth=2.2,
                    label=TECH_LABELS.get(cycle, cycle.upper()),
                )
                if annotate:
                    idx = min(int(np.searchsorted(y, 0.8 * y[-1])), len(y) - 1)
                    ax.scatter(
                        [x[idx]],
                        [y[idx]],
                        color=TECH_COLORS.get(cycle, "#333333"),
                        edgecolor="black",
                        linewidth=0.45,
                        s=18,
                        zorder=4,
                    )
                    ax.annotate(
                        f"{x[idx]:.0f} €/MWh$_e$",
                        (x[idx], y[idx]),
                        xytext=(0, -8),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=11,
                        bbox=dict(
                            boxstyle="round,pad=0.18",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.85,
                        ),
                        zorder=5,
                    )
            add_panel_letter(ax, panel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Cumulative potential (GW)")
            ax.legend(
                loc=legend_loc,
                frameon=True,
                facecolor="white",
                framealpha=0.88,
                edgecolor="#BBBBBB",
                fontsize=11,
            )
            ax.grid(False)

        for ax in axes.ravel():
            ax.set_box_aspect(1)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)
            ax.tick_params(axis="both", labelsize=11)
            for spine in ax.spines.values():
                spine.set_visible(True)
        fig.tight_layout(pad=0.25, w_pad=0.45, h_pad=0.45)
        fig.savefig(_ensure_parent(out_path), dpi=250, bbox_inches="tight")
        plt.close(fig)


def _summarize_component_shares(technology_shares):
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
                    "values": values,
                }
    return summaries


def _draw_cost_breakdown(ax, technology_shares):
    summaries = _summarize_component_shares(technology_shares)
    technologies = [technology for technology in TECH_LABELS if technology in summaries]
    technologies.extend(t for t in summaries if t not in technologies)
    if not technologies:
        ax.axis("off")
        return

    positions = np.arange(1, len(COMPONENTS) + 1)
    offsets = (
        np.linspace(-0.22, 0.22, len(technologies)) if len(technologies) > 1 else [0.0]
    )
    rng = np.random.default_rng(42)
    for technology, offset in zip(technologies, offsets):
        color = TECH_COLORS.get(technology, "#4D4D4D")
        for position, (component, _) in zip(positions, COMPONENTS):
            stats = summaries[technology].get(component)
            if stats is None:
                continue
            x = position + offset
            bp = ax.boxplot(
                [stats["values"]],
                positions=[x],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                zorder=2,
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.58)
            bp["boxes"][0].set_edgecolor("#707070")
            for element in ("whiskers", "caps", "medians"):
                for item in bp[element]:
                    item.set_color("#707070" if element != "medians" else "#202020")
            sample_size = min(700, stats["values"].size)
            sample = rng.choice(stats["values"], size=sample_size, replace=False)
            ax.scatter(
                x + rng.uniform(-0.07, 0.07, size=sample_size),
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
                linewidth=1.2,
                zorder=4,
            )

    labels = [label for _, label in COMPONENTS]
    ax.set_xticks(positions, labels)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=9.5)
    ax.set_xlim(0.5, len(COMPONENTS) + 0.5)
    ax.set_ylabel("Contribution to LCOE (%)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
    handles = [
        Patch(
            facecolor=TECH_COLORS.get(technology, "#4D4D4D"),
            edgecolor="#707070",
            alpha=0.58,
            label=TECH_LABELS.get(technology, technology.upper()),
        )
        for technology in technologies
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#202020",
            markeredgewidth=1.2,
            markersize=5.5,
            label="Mean",
        )
    )
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#BBBBBB",
        title="Technology",
        fontsize=9,
        title_fontsize=9,
    )


def make_depth_cost_breakdown_four_panel(
    depth_maps: dict[str, xr.DataArray], technology_shares, shapes_path, out_path
):
    """Combine optimal-depth maps and geographic LCOE cost shares."""
    technologies = [t for t in TECH_LABELS if t in depth_maps]
    if not technologies:
        write_blank_png(out_path, figsize=(8.27, 8.27))
        return
    reference = depth_maps[technologies[0]]
    countries = _country_layer(shapes_path, reference)
    arrays = [
        _plot_values(depth_maps[t], countries, value_scale=1000.0) for t in technologies
    ]
    vmin, vmax = _finite_limits(arrays)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        write_blank_png(out_path, figsize=(8.27, 8.27))
        return
    extent = _map_extent(reference)

    with plt.rc_context(BASE_STYLE):
        cmap = plt.get_cmap(LCOE_CMAP).copy()
        cmap.set_bad((0, 0, 0, 0))
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.27))
        map_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
        image = None
        for index, (ax, technology, values) in enumerate(
            zip(map_axes, technologies, arrays)
        ):
            _decorate_map(ax, countries, extent)
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                zorder=2,
            )
            ax.text(
                0.02,
                0.98,
                f"{PANEL_LETTERS[index]}) {TECH_LABELS.get(technology, technology.upper())}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )
            _clean_map_axes(ax)
            ax.set_box_aspect(1)
        for ax in map_axes[len(technologies) :]:
            ax.axis("off")

        cost_ax = axes[1, 1]
        _draw_cost_breakdown(cost_ax, technology_shares)
        cost_ax.set_box_aspect(1)
        cost_ax.text(
            0.02,
            0.98,
            "d)",
            transform=cost_ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2),
            zorder=6,
        )
        fig.subplots_adjust(
            left=0.055, right=0.985, top=0.98, bottom=0.12, wspace=0.16, hspace=0.16
        )
        if image is not None:
            cax = fig.add_axes([0.09, 0.045, 0.38, 0.022])
            colorbar = fig.colorbar(
                cm.ScalarMappable(norm=image.norm, cmap=image.cmap),
                cax=cax,
                orientation="horizontal",
            )
            colorbar.set_label("Depth at minimum LCOE (km)", fontsize=12)
            colorbar.ax.tick_params(length=0, labelsize=11)
        fig.savefig(_ensure_parent(out_path), dpi=200)
        plt.close(fig)


def make_regional_potential_overview(
    geoparquet_path,
    shapes_path,
    out_path,
    cycles: list[str],
    depth_category: str,
    bin_id: str,
):
    """Plot regional potential for one configured depth/category bin."""
    gdf = gpd.read_parquet(geoparquet_path)
    gdf = gdf.loc[
        (gdf["depth_category"].astype(str) == str(depth_category))
        & (gdf["bin_id"].astype(str) == str(bin_id))
    ].copy()
    regions = _schemas.ShapeSchema.validate(gpd.read_parquet(shapes_path))
    if "shape_class" in regions.columns:
        regions = regions.loc[regions["shape_class"] != "maritime"].copy()
    regions["shape_id"] = regions["shape_id"].astype(str)
    regions = regions[["shape_id", regions.geometry.name]].dissolve(
        by="shape_id", as_index=False
    )
    if gdf.crs is not None and regions.crs != gdf.crs:
        regions = regions.to_crs(gdf.crs)
    panel_specs = [
        (
            technology,
            regions.merge(
                gdf.loc[gdf["technology"] == technology, ["region_id", "potential"]],
                left_on="shape_id",
                right_on="region_id",
                how="left",
            ),
        )
        for technology in cycles
        if not gdf.loc[gdf["technology"] == technology].empty
    ]
    if not panel_specs:
        write_blank_png(out_path)
        return
    values = [
        panel["potential"].to_numpy(dtype=float) / 1000.0 for _, panel in panel_specs
    ]
    vmin, vmax = _finite_limits(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        write_blank_png(out_path)
        return

    bounds = panel_specs[0][1].total_bounds
    xmin, ymin, xmax, ymax = bounds
    data_aspect = (ymax - ymin) / (xmax - xmin)
    n_panels = len(panel_specs)
    fig_w, fig_h = 9.5, 5.3
    gap, margin_l, margin_r, margin_t, margin_b = 0.08, 0.14, 0.14, 0.42, 0.78
    available_w = fig_w - margin_l - margin_r - (n_panels - 1) * gap
    available_h = fig_h - margin_t - margin_b
    panel_w = available_w / n_panels
    panel_h = min(available_h, panel_w * data_aspect)

    with plt.rc_context(BASE_STYLE):
        fig = plt.figure(figsize=(fig_w, fig_h))
        axes = []
        for index in range(n_panels):
            x0 = (margin_l + index * (panel_w + gap)) / fig_w
            y0 = (margin_b + 0.5 * (available_h - panel_h)) / fig_h
            axes.append(fig.add_axes([x0, y0, panel_w / fig_w, panel_h / fig_h]))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(UNAVAILABLE_COLOR)
        for idx, (ax, (technology, panel)) in enumerate(zip(axes, panel_specs)):
            panel["_plot_value"] = panel["potential"].astype(float) / 1000.0
            ax.set_facecolor(UNAVAILABLE_COLOR)
            panel.plot(
                column="_plot_value",
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                missing_kwds={"color": UNAVAILABLE_COLOR},
            )
            panel.geometry.boundary.plot(ax=ax, linewidth=0.35, color="#666666")
            ax.text(
                0.02,
                0.98,
                f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(technology, technology.upper())}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
            )
            _clean_map_axes(ax)
        cax = fig.add_axes([0.14, 0.08, 0.72, 0.04])
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(vmin, vmax)
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Potential (GW)")
        cbar.ax.tick_params(length=0, labelsize=9)
        fig.suptitle(
            f"Potential per region — {depth_category}, {bin_id}", y=0.96, fontsize=12
        )
        fig.savefig(_ensure_parent(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_sensitivity_boxplot(
    specs,
    value_key,
    ylabel,
    out_path,
    *,
    summary_fn=None,
    summary_position="top",
    show_mean_dot=False,
):
    """Plot one baseline box plus one box per selected sensitivity parameter."""
    positions, data, labels, kinds, summaries, means = [], [], [], [], [], []
    for spec in specs:
        values = np.asarray(spec[value_key], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        positions.append(len(positions) + 1)
        data.append(values)
        labels.append(spec["label"])
        kinds.append(spec["kind"])
        summaries.append(summary_fn(values) if summary_fn else None)
        means.append(float(np.nanmean(values)))
    if not data:
        write_blank_png(out_path)
        return

    with plt.rc_context(SENSITIVITY_STYLE):
        fig, ax = plt.subplots(figsize=(max(12, 2.25 * len(data)), 8.5))
        bp = ax.boxplot(
            data, positions=positions, patch_artist=True, widths=0.7, showfliers=False
        )
        kind_colors = {"baseline": "#BDBDBD", "scenario": "#1F78B4"}
        for patch, kind in zip(bp["boxes"], kinds):
            patch.set_facecolor(kind_colors[kind])
            patch.set_alpha(0.80)
            patch.set_linewidth(0.8)
        for element in ("whiskers", "caps", "medians"):
            for item in bp[element]:
                item.set_linewidth(0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(False)
        if show_mean_dot:
            ax.scatter(
                positions,
                means,
                s=64,
                marker="o",
                facecolor="white",
                edgecolor="#202020",
                linewidth=1.2,
                zorder=4,
            )
        handles = [
            Patch(
                facecolor=kind_colors["baseline"],
                edgecolor="black",
                alpha=0.8,
                label="Baseline",
            ),
            Patch(
                facecolor=kind_colors["scenario"],
                edgecolor="black",
                alpha=0.8,
                label="Scenario",
            ),
        ]
        if show_mean_dot:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor="#202020",
                    markeredgewidth=1.2,
                    markersize=8,
                    label="Mean",
                )
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="#BBBBBB",
            fontsize=16,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin, ymax + (0.08 if summary_position == "top" else 0.04) * yrange)
        text_transform = mtransforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )
        if summary_position in {"top", "bottom"}:
            for x, summary in zip(positions, summaries):
                if summary:
                    ax.text(
                        x,
                        1.03 if summary_position == "top" else -0.13,
                        summary,
                        transform=text_transform,
                        ha="center",
                        va="bottom" if summary_position == "top" else "top",
                        fontsize=18,
                        clip_on=False,
                    )
        fig.tight_layout()
        if summary_position == "bottom":
            fig.subplots_adjust(bottom=0.22)
        fig.savefig(_ensure_parent(out_path))
        plt.close(fig)


def _load_boxplot_table(path):
    path = str(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _scenario_winners(df, scenario_name, baseline_name):
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


def _transition_label(tech_from, tech_to):
    return (
        f"{TECH_LABELS.get(tech_from, tech_from.upper())} to "
        f"{TECH_LABELS.get(tech_to, tech_to.upper())}"
    )


def _transition_colors(transitions):
    preferred = {
        "ORC to Kalina": "#0072B2",
        "Kalina to ORC": "#D55E00",
        "ORC to Flash": "#009E73",
        "Flash to ORC": "#CC79A7",
        "Kalina to Flash": "#E69F00",
        "Flash to Kalina": "#56B4E9",
    }
    fallback = ["#000000", "#F0E442", "#8C564B", "#9467BD", "#17BECF", "#7F7F7F"]
    colors = {}
    fallback_index = 0
    for label in transitions:
        if label in preferred:
            colors[label] = preferred[label]
        else:
            colors[label] = fallback[fallback_index % len(fallback)]
            fallback_index += 1
    return colors


def make_switch_transition_map(
    switch_df,
    scenario_to_boxplot,
    baseline_winner_df,
    baseline,
    top_n,
    out_path,
    shapes_path,
    reference_raster_path,
    required_params=None,
):
    """Plot spatial technology switches for selected sensitivity scenarios."""
    plot_df = switch_df.loc[
        np.isfinite(switch_df["transition_total_pixels"].to_numpy(dtype=float))
    ].copy()
    plot_df = plot_df.sort_values(
        ["transition_total_pixels", "sort_magnitude", "parameter"],
        ascending=[False, False, True],
    )
    required_params = required_params or []
    plot_df = (
        pd.concat(
            [
                plot_df.head(top_n),
                plot_df.loc[plot_df["parameter"].isin(required_params)],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["parameter"], keep="first")
        .sort_values(
            ["transition_total_pixels", "sort_magnitude", "parameter"],
            ascending=[False, False, True],
        )
        .head(top_n)
    )
    if plot_df.empty or not reference_raster_path:
        write_blank_png(out_path)
        return

    if not isinstance(reference_raster_path, (str, Path)):
        reference_paths = list(reference_raster_path)
        if not reference_paths:
            write_blank_png(out_path)
            return
        reference_raster_path = reference_paths[0]

    with rasterio.open(reference_raster_path) as src:
        height, width = src.height, src.width
        bounds = src.bounds
        extent = [
            min(bounds.left, bounds.right),
            max(bounds.left, bounds.right),
            min(bounds.bottom, bounds.top),
            max(bounds.bottom, bounds.top),
        ]
        raster_crs = src.crs
    countries = gpd.read_parquet(shapes_path)
    if "shape_class" in countries.columns:
        countries = countries.loc[countries["shape_class"] == "land"].copy()
    if raster_crs is not None and countries.crs != raster_crs:
        countries = countries.to_crs(raster_crs)

    panel_specs = []
    transitions_seen = []
    base_cols = ["row", "col", "tech_base_winner"]
    for _, row in plot_df.iterrows():
        scenario = row["scenario"]
        if scenario not in scenario_to_boxplot:
            continue
        box = _schemas.SensitivityBoxplotSchema.validate(
            _load_boxplot_table(scenario_to_boxplot[scenario])
        )
        if box.empty:
            continue
        scenario_winners = _scenario_winners(box, scenario, baseline)
        joined = baseline_winner_df[base_cols].merge(
            scenario_winners[["row", "col", "tech_scen_winner"]],
            on=["row", "col"],
            how="inner",
        )
        comparable = joined.loc[
            joined["tech_base_winner"].notna() & joined["tech_scen_winner"].notna()
        ].copy()
        changed = comparable.loc[
            comparable["tech_base_winner"] != comparable["tech_scen_winner"]
        ].copy()
        unchanged = comparable.loc[
            comparable["tech_base_winner"] == comparable["tech_scen_winner"]
        ].copy()
        labels = [
            _transition_label(t_from, t_to)
            for t_from, t_to in zip(
                changed["tech_base_winner"], changed["tech_scen_winner"]
            )
        ]
        if not labels:
            continue
        changed["_transition"] = labels
        transitions_seen.extend(labels)
        panel_specs.append(
            {
                "label": row["label"],
                "changed": changed,
                "unchanged": unchanged,
                "switch_rate": float(row["transition_total_pixels"]) * 100,
            }
        )
    if not panel_specs or not transitions_seen:
        write_blank_png(out_path)
        return

    transitions = sorted(set(transitions_seen))
    colors = _transition_colors(transitions)
    transition_to_code = {label: index + 2 for index, label in enumerate(transitions)}
    cmap = ListedColormap(
        [(1, 1, 1, 0), UNCHANGED_COLOR, *[colors[label] for label in transitions]]
    )
    norm = BoundaryNorm(np.arange(-0.5, len(transitions) + 2.5), cmap.N)
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig_w, fig_h = 8.9, 8.35
    margin_l = margin_r = 0.18
    margin_t, margin_b, gap_x, gap_y = 0.10, 0.78, 0.08, 0.12
    available_w = fig_w - margin_l - margin_r - gap_x
    available_h = fig_h - margin_t - margin_b - gap_y
    panel_w = min(available_w / 2, (available_h / 2) / data_aspect)
    panel_h = panel_w * data_aspect
    start_x = margin_l + 0.5 * (available_w - 2 * panel_w - gap_x)
    start_y = margin_b + 0.5 * (available_h - 2 * panel_h - gap_y)

    with plt.rc_context(SENSITIVITY_STYLE):
        fig = plt.figure(figsize=(fig_w, fig_h))
        axes = []
        for row_index in range(2):
            for col_index in range(2):
                x0 = (start_x + col_index * (panel_w + gap_x)) / fig_w
                y0 = (start_y + (1 - row_index) * (panel_h + gap_y)) / fig_h
                axes.append(fig.add_axes([x0, y0, panel_w / fig_w, panel_h / fig_h]))
        for idx, ax in enumerate(axes):
            ax.set_facecolor(MAP_BACKGROUND_COLOR)
            countries.plot(
                ax=ax, facecolor=UNAVAILABLE_COLOR, edgecolor="none", zorder=1
            )
            countries.boundary.plot(
                ax=ax, color=COUNTRY_EDGE_COLOR, linewidth=0.25, zorder=4
            )
            _clean_map_axes(ax)
            if idx >= len(panel_specs):
                ax.axis("off")
                continue
            spec = panel_specs[idx]
            image = np.zeros((height, width), dtype=np.int16)
            for _, pixel in spec["unchanged"].iterrows():
                row, col = int(pixel["row"]), int(pixel["col"])
                if 0 <= row < height and 0 <= col < width:
                    image[row, col] = 1
            for _, pixel in spec["changed"].iterrows():
                row, col = int(pixel["row"]), int(pixel["col"])
                if 0 <= row < height and 0 <= col < width:
                    image[row, col] = transition_to_code[pixel["_transition"]]
            ax.imshow(
                image,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                extent=extent,
                origin="upper",
                zorder=2,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.text(
                0.02,
                0.985,
                f"{PANEL_LETTERS[idx]}) {spec['label']}\n{spec['switch_rate']:.1f}% switched",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                linespacing=1.08,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
                zorder=5,
            )
        handles = [
            Patch(
                facecolor=UNAVAILABLE_COLOR,
                edgecolor=COUNTRY_EDGE_COLOR,
                label="Geothermal not possible",
            ),
            Patch(facecolor=UNCHANGED_COLOR, edgecolor="#666666", label="Not changed"),
            *[
                Patch(facecolor=colors[label], edgecolor="none", label=label)
                for label in transitions
            ],
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=11,
        )
        fig.savefig(_ensure_parent(out_path), dpi=300)
        plt.close(fig)
