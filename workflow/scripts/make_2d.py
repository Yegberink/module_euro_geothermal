"""Reduce geothermal cubes to rasters and create diagnostic figures."""

import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import _schemas
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import seaborn as sns
import xarray as xr
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


plt.rcParams.update(
    {
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
)

TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
TECH_COLORS = {"orc": "#0072B2", "flash": "#E69F00", "kalina": "#009E73"}
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
LCOE_OVERVIEW_CMAP = "YlOrBr_r"
LCOE_OVERVIEW_COLORBAR_CUTOFF = 600.0
LCOE_OVERVIEW_NORM_GAMMA = 0.4
MAP_CMAP = "YlOrBr"
MAP_BACKGROUND_COLOR = "#DCEEF4"
UNAVAILABLE_COLOR = "#D9D9D9"
LCOE_UNAVAILABLE_COLOR = "#BDBDBD"
COUNTRY_EDGE_COLOR = "#4D5960"

# --------------------------
# Data classes
# --------------------------


@dataclass(frozen=True)
class DepthCategory:
    """One half-open interval used to aggregate depth-resolved results."""

    zmin: float
    zmax: float

    @property
    def label(self) -> str:
        """Return the interval label used in output filenames."""
        return f"{self.zmin / 1000:g}-{self.zmax / 1000:g}km"


# --------------------------
# Helpers
# --------------------------


def _parse_depth_categories(raw) -> list[DepthCategory]:
    raw = raw or []
    out: list[DepthCategory] = []
    for pair in raw:
        out.append(DepthCategory(float(pair[0]), float(pair[1])))
    return out


def _ensure_raster_metadata(
    da: xr.DataArray, *, x_name: str = "x", y_name: str = "y", crs: str | None = None
) -> xr.DataArray:
    # Ensure dims are (y, x) for rasters
    if da.dims != (y_name, x_name):
        da = da.transpose(y_name, x_name)

    da = da.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=False)

    crs_guess = (
        crs
        or da.attrs.get("crs")
        or da.attrs.get("spatial_ref")
        or da.attrs.get("epsg")
    )
    if crs_guess is not None:
        if isinstance(crs_guess, (int, np.integer)):
            crs_guess = f"EPSG:{int(crs_guess)}"
        da = da.rio.write_crs(crs_guess, inplace=False)

    return da


def write_geotiff(
    da: xr.DataArray,
    out_path: Path,
    *,
    nodata: float = np.nan,
    crs: str | None = None,
    x_name: str = "x",
    y_name: str = "y",
):
    """Write a georeferenced data array as a compressed GeoTIFF."""
    if da.sizes.get(y_name, 1) == 0 or da.sizes.get(x_name, 1) == 0:
        da = _schemas.empty_raster_dataarray(da, x_name=x_name, y_name=y_name)
    da2 = _ensure_raster_metadata(da, x_name=x_name, y_name=y_name, crs=crs)

    if np.isnan(nodata):
        da2 = da2.rio.write_nodata(np.nan, inplace=False)
    else:
        da2 = da2.rio.write_nodata(nodata, inplace=False)

    da2.rio.to_raster(str(out_path), compress="deflate")


def _map_extent(da: xr.DataArray, x_name: str = "x", y_name: str = "y") -> list[float]:
    if da.dims != (y_name, x_name):
        da = da.transpose(y_name, x_name)
    x = da[x_name].values
    y = da[y_name].values
    return [x.min(), x.max(), y.min(), y.max()]


def _clean_map_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#666666")


def _map_crs(da: xr.DataArray):
    """Return a raster CRS without requiring it to be present."""
    try:
        return da.rio.crs
    except (AttributeError, ValueError):
        return None


def _country_layer(shapes_path: Path, reference_da: xr.DataArray) -> gpd.GeoDataFrame:
    countries = gpd.read_parquet(shapes_path)
    if "shape_class" in countries.columns:
        countries = countries.loc[countries["shape_class"] == "land"].copy()
    target_crs = _map_crs(reference_da)
    if target_crs is not None and countries.crs != target_crs:
        countries = countries.to_crs(target_crs)
    return countries


def _plot_values(
    da: xr.DataArray, countries: gpd.GeoDataFrame, value_scale: float = 1.0
) -> np.ndarray:
    """Return map values masked to the included land polygons."""
    raster = da.transpose("y", "x")
    inside_countries = geometry_mask(
        countries.geometry,
        out_shape=(raster.sizes["y"], raster.sizes["x"]),
        transform=raster.rio.transform(),
        invert=True,
    )
    values = np.asarray(raster.values, dtype=np.float32) / value_scale
    return np.where(inside_countries, values, np.nan)


def _decorate_map(
    ax,
    countries: gpd.GeoDataFrame,
    extent: list[float],
    crs,
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


def _winner_map(lcoe_maps: dict[str, xr.DataArray], cycles: list[str]) -> np.ndarray:
    stack = np.stack([lcoe_maps[cycle].values for cycle in cycles], axis=0).astype(
        np.float32
    )
    finite = np.isfinite(stack)
    stack_filled = np.where(finite, stack, np.inf)
    winner = np.argmin(stack_filled, axis=0).astype(np.int16)
    all_nan = ~np.any(finite, axis=0)
    return np.where(all_nan, -1, winner)


def _finite_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite_arrays = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    if not finite_arrays:
        return (np.nan, np.nan)
    return (
        float(min(np.nanmin(a) for a in finite_arrays)),
        float(max(np.nanmax(a) for a in finite_arrays)),
    )


def _cumulative_potential_curve(
    sort_da: xr.DataArray,
    potential_da: xr.DataArray,
    potential_scale: float = 1000.0,
    x_scale: float = 1.0,
    descending: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sort_da.values, dtype=np.float64).ravel() / x_scale
    potential = (
        np.asarray(potential_da.values, dtype=np.float64).ravel() / potential_scale
    )
    mask = np.isfinite(x) & np.isfinite(potential) & (potential > 0)
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x = x[mask]
    potential = potential[mask]
    order = np.argsort(x, kind="mergesort")
    if descending:
        order = order[::-1]
    x_sorted = x[order]
    cumulative_potential = np.cumsum(potential[order])
    return x_sorted, cumulative_potential


def _finite_mean_by_depth(da3d: xr.DataArray, value_scale: float = 1.0) -> np.ndarray:
    return (
        da3d.astype(np.float64)
        .mean(dim=("y", "x"), skipna=True)
        .values.astype(np.float64)
        / value_scale
    )


def make_three_panel_map_figure(
    maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    colorbar_label: str,
    cmap: str,
    value_scale: float = 1.0,
    countries: gpd.GeoDataFrame,
):
    """Plot one consistently styled map panel per conversion cycle."""
    n_panels = len(cycles)
    ncols = n_panels

    reference_da = maps[cycles[0]]
    extent = _map_extent(reference_da)
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    # Widescreen PowerPoint canvas (16:9), with three equal map panels.
    fig_w = 13.333
    fig_h = 7.5
    panel_w = 4.25
    panel_h = min(panel_w * data_aspect, 6.35)
    gap = 0.15
    margin_l = (fig_w - ncols * panel_w - (ncols - 1) * gap) / 2
    margin_b = 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))

    def make_ax(col):
        x0 = (margin_l + col * (panel_w + gap)) / fig_w
        y0 = (margin_b + (fig_h - margin_b - panel_h) / 2) / fig_h
        w = panel_w / fig_w
        h = panel_h / fig_h
        return fig.add_axes([x0, y0, w, h])

    axes = [make_ax(i) for i in range(n_panels)]
    scaled_arrays = [
        _plot_values(maps[cycle], countries, value_scale) for cycle in cycles
    ]
    vmin, vmax = _finite_limits(scaled_arrays)
    map_cmap = plt.get_cmap(cmap).copy()
    map_cmap.set_bad((0, 0, 0, 0))

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        _decorate_map(ax, countries, extent, _map_crs(reference_da))
        im = ax.imshow(
            scaled_arrays[idx],
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=map_cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="black",
        )
        _clean_map_axes(ax)
        ax.set_aspect("equal")

    cbar_y0 = max(0.06, axes[0].get_position().y0 - 0.08)
    cax = fig.add_axes([0.14, cbar_y0, 0.72, 0.045])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(length=0, labelsize=16)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_cumulative_potential_figure(
    x_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    xlabel: str,
    x_scale: float = 1.0,
    descending: bool = False,
    ylabel: str = "Cumulative potential below cost (GW)",
):
    """Plot cumulative potential against cost, depth, or temperature."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    is_depth_curve = "Depth" in xlabel
    is_temperature_curve = "Temperature" in xlabel

    for cycle in cycles:
        x, y = _cumulative_potential_curve(
            x_maps[cycle], potential_maps[cycle], x_scale=x_scale, descending=descending
        )
        if x.size == 0:
            continue

        ax.plot(
            x,
            y,
            color=TECH_COLORS.get(cycle, "#333333"),
            linewidth=2.4,
            label=TECH_LABELS.get(cycle, cycle.upper()),
        )

        if is_temperature_curve:
            continue

        target_y = 0.8 * y[-1]
        idx = int(np.searchsorted(y, target_y, side="left"))
        idx = min(idx, len(y) - 1)

        ax.scatter(
            [x[idx]],
            [y[idx]],
            color=TECH_COLORS.get(cycle, "#333333"),
            edgecolor="black",
            linewidth=0.5,
            s=22,
            zorder=4,
        )
        label_value = (
            f"{x[idx]:.1f} km" if is_depth_curve else f"{x[idx]:.0f} €/MWh$_e$"
        )
        if is_temperature_curve:
            label_value = f"{x[idx]:.0f} °C"
        ax.annotate(
            label_value,
            (x[idx], y[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            color="black",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if descending and not is_temperature_curve:
        ax.invert_xaxis()
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_average_potential_by_depth_figure(
    potential_profiles: dict[str, np.ndarray],
    depths_m: np.ndarray,
    cycles: list[str],
    out_path: Path,
):
    """Plot average grid-cell potential at each evaluated depth."""
    fig, ax = plt.subplots(figsize=(6, 4))

    depths_km = np.asarray(depths_m, dtype=np.float64) / 1000.0
    for cycle in cycles:
        y = np.asarray(potential_profiles[cycle], dtype=np.float64)
        mask = np.isfinite(depths_km) & np.isfinite(y)
        if not np.any(mask):
            continue

        ax.plot(
            depths_km[mask],
            y[mask],
            color=TECH_COLORS.get(cycle, "#333333"),
            linewidth=2.0,
            label=TECH_LABELS.get(cycle, cycle.upper()),
        )

    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Average potential per grid cell (GW)")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_depth_profile_figure(
    lcoe_map: xr.DataArray,
    potential_map: xr.DataArray,
    depth_map: xr.DataArray,
    cycle: str,
    out_path: Path,
):
    """Plot potential and LCOE distributions by selected depth."""
    depth = np.asarray(depth_map.values, dtype=np.float64).ravel() / 1000.0
    lcoe = np.asarray(lcoe_map.values, dtype=np.float64).ravel()
    potential = np.asarray(potential_map.values, dtype=np.float64).ravel() / 1000.0
    valid = (
        np.isfinite(depth)
        & np.isfinite(potential)
        & np.isfinite(lcoe)
        & (potential > 0)
    )

    if not np.any(valid):
        _schemas.write_blank_png(out_path)
        return

    depth = depth[valid]
    potential = potential[valid]
    lcoe = lcoe[valid]
    depth_levels = np.unique(depth[np.isfinite(depth)])
    mean_potential = np.array(
        [np.nanmean(potential[np.isclose(depth, level)]) for level in depth_levels],
        dtype=np.float64,
    )
    mean_lcoe = np.array(
        [np.nanmean(lcoe[np.isclose(depth, level)]) for level in depth_levels],
        dtype=np.float64,
    )
    line_valid = (
        np.isfinite(depth_levels) & np.isfinite(mean_potential) & np.isfinite(mean_lcoe)
    )
    if not np.any(line_valid):
        _schemas.write_blank_png(out_path)
        return

    depth_levels = depth_levels[line_valid]
    mean_potential = mean_potential[line_valid]
    mean_lcoe = mean_lcoe[line_valid]
    color = TECH_COLORS.get(cycle, "#333333")
    label = TECH_LABELS.get(cycle, cycle.upper())

    fig, ax_potential = plt.subplots(figsize=(6.6, 5.2))
    ax_lcoe = ax_potential.twiny()

    potential_line = ax_potential.plot(
        mean_potential, depth_levels, color=color, linewidth=2.4, label="Mean potential"
    )[0]
    lcoe_line = ax_lcoe.plot(
        mean_lcoe, depth_levels, color="#202020", linewidth=2.4, label="Mean LCOE"
    )[0]

    ax_potential.invert_yaxis()
    ax_potential.set_ylabel("Depth at minimum LCOE (km)")
    ax_potential.set_xlabel("Mean potential (GW per 10x10km grid cell)", color=color)
    ax_lcoe.set_xlabel("Mean LCOE (€/MWh$_e$)", color="#202020")
    ax_potential.tick_params(axis="x", colors=color)
    ax_lcoe.tick_params(axis="x", colors="#202020")
    ax_potential.set_title(f"{label} minimized-depth cells")
    ax_potential.grid(False)

    for ax in (ax_potential, ax_lcoe):
        for spine in ax.spines.values():
            spine.set_visible(True)

    ax_potential.legend(
        handles=[potential_line, lcoe_line],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.86, bottom=0.22)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_potential_boxplot_figure(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
    """Plot LCOE and potential boxplots for all conversion cycles."""
    lcoe_data = []
    potential_data = []
    labels = []

    for cycle in cycles:
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        potential = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        lcoe_mask = np.isfinite(lcoe)
        potential_mask = np.isfinite(potential) & (potential > 0)
        if not np.any(lcoe_mask) or not np.any(potential_mask):
            continue
        lcoe_data.append(lcoe[lcoe_mask])
        potential_data.append(potential[potential_mask])
        labels.append(TECH_LABELS.get(cycle, cycle.upper()))

    if not lcoe_data:
        _schemas.write_blank_png(out_path)
        return

    fig, ax_lcoe = plt.subplots(figsize=(10.5, 6.2))
    ax_potential = ax_lcoe.twinx()

    centers = np.arange(1, len(labels) + 1, dtype=float)
    offset = 0.18
    lcoe_positions = centers - offset
    potential_positions = centers + offset

    lcoe_bp = ax_lcoe.boxplot(
        lcoe_data,
        positions=lcoe_positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )
    potential_bp = ax_potential.boxplot(
        potential_data,
        positions=potential_positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )

    def style_boxplot(bp, facecolor):
        for box in bp["boxes"]:
            box.set_facecolor(facecolor)
            box.set_edgecolor("#707070")
            box.set_linewidth(1.0)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("#707070" if element != "medians" else "#202020")
                item.set_linewidth(1.0)

    style_boxplot(lcoe_bp, "#8C8C8C")
    style_boxplot(potential_bp, "#3F7FBF")

    ax_lcoe.set_xticks(centers)
    ax_lcoe.set_xticklabels(labels, fontsize=15)
    ax_lcoe.set_ylabel("LCOE (€/MWh$_e$)", fontsize=18)
    ax_potential.set_ylabel("Theoretical potential (GW)", fontsize=18)
    ax_lcoe.tick_params(axis="y", labelsize=16)
    ax_potential.tick_params(axis="y", labelsize=16)
    ax_lcoe.tick_params(axis="x", labelsize=15)
    ax_lcoe.grid(False)
    ax_potential.grid(False)

    for ax in (ax_lcoe, ax_potential):
        for spine in ax.spines.values():
            spine.set_visible(True)

    legend_handles = [
        Patch(facecolor="#8C8C8C", edgecolor="#707070", label="LCOE"),
        Patch(facecolor="#3F7FBF", edgecolor="#707070", label="Potential"),
    ]
    ax_lcoe.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        fontsize=15,
        ncol=2,
        borderaxespad=0.0,
    )

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_potential_split_boxplot_figure(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    max_scatter_points_per_tech: int = 900,
):
    """Plot separate LCOE, potential, and joint-distribution panels."""
    lcoe_data = []
    potential_data = []
    labels = []
    colors = []
    scatter_rows = []
    rng = np.random.default_rng(42)

    for cycle in cycles:
        label = TECH_LABELS.get(cycle, cycle.upper())

        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = lcoe[np.isfinite(lcoe)]

        potential = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        potential = potential[np.isfinite(potential) & (potential > 0)]

        if lcoe.size == 0 or potential.size == 0:
            continue

        lcoe_data.append(lcoe)
        potential_data.append(potential)
        labels.append(label)
        colors.append(TECH_COLORS.get(cycle, "#333333"))

        for metric, values in (("LCOE", lcoe), ("Potential", potential)):
            sample_size = min(max_scatter_points_per_tech, values.size)
            sample_idx = rng.choice(values.size, size=sample_size, replace=False)
            scatter_rows.extend(
                {"technology": label, "metric": metric, "value": value}
                for value in values[sample_idx]
            )

    if not labels:
        _schemas.write_blank_png(out_path)
        return

    scatter_df = pd.DataFrame(scatter_rows)
    palette = {label: color for label, color in zip(labels, colors)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=False)
    metric_specs = [
        ("LCOE", lcoe_data, "LCOE (€/MWh$_e$)", "a) LCOE"),
        (
            "Potential",
            potential_data,
            "Theoretical potential (GW per grid cell)",
            "b) Theoretical potential",
        ),
    ]

    for ax, (metric, data, ylabel, title) in zip(axes, metric_specs):
        metric_scatter = scatter_df.loc[scatter_df["metric"] == metric]

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
            box.set_linewidth(1.0)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("#707070" if element != "medians" else "#202020")
                item.set_linewidth(1.0)

        sns.stripplot(
            data=metric_scatter,
            x="technology",
            y="value",
            order=labels,
            palette=palette,
            jitter=0.22,
            size=1.8,
            alpha=0.22,
            linewidth=0,
            ax=ax,
        )

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.tight_layout(w_pad=2.6)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_potential_cumulative_four_panel_figure(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    temperature_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    max_scatter_points_per_tech: int = 700,
):
    """Plot four complementary LCOE and potential summaries."""
    lcoe_data = []
    potential_data = []
    labels = []
    colors = []
    scatter_rows = []
    rng = np.random.default_rng(42)

    for cycle in cycles:
        label = TECH_LABELS.get(cycle, cycle.upper())
        color = TECH_COLORS.get(cycle, "#333333")

        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = lcoe[np.isfinite(lcoe)]

        potential = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        potential = potential[np.isfinite(potential) & (potential > 0)]

        if lcoe.size == 0 or potential.size == 0:
            continue

        lcoe_data.append(lcoe)
        potential_data.append(potential)
        labels.append(label)
        colors.append(color)

        for metric, values in (("LCOE", lcoe), ("Potential", potential)):
            sample_size = min(max_scatter_points_per_tech, values.size)
            sample_idx = rng.choice(values.size, size=sample_size, replace=False)
            scatter_rows.extend(
                {"technology": label, "metric": metric, "value": value}
                for value in values[sample_idx]
            )

    if not labels:
        _schemas.write_blank_png(out_path)
        return

    scatter_df = pd.DataFrame(scatter_rows)
    palette = {label: color for label, color in zip(labels, colors)}

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.27))
    ax_lcoe_box, ax_potential_box = axes[0]
    ax_lcoe_curve, ax_temp_curve = axes[1]

    def add_panel_letter(ax, label: str):
        ax.text(
            0.03,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
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
        metric_scatter = scatter_df.loc[scatter_df["metric"] == metric]
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
            box.set_linewidth(1.0)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]:
                item.set_color("#707070" if element != "medians" else "#202020")
                item.set_linewidth(1.0)

        sns.stripplot(
            data=metric_scatter,
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
        add_panel_letter(ax, panel_label)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(False)

    curve_specs = [
        (
            ax_lcoe_curve,
            lcoe_maps,
            "LCOE threshold (€/MWh$_e$)",
            "Cumulative potential (GW)",
            False,
            "c)",
            "lower right",
            True,
        ),
        (
            ax_temp_curve,
            temperature_maps,
            "Temperature threshold (°C)",
            "Cumulative potential (GW)",
            True,
            "d)",
            "upper right",
            False,
        ),
    ]

    for (
        ax,
        x_maps,
        xlabel,
        ylabel,
        descending,
        panel_label,
        legend_loc,
        annotate_80pct,
    ) in curve_specs:
        for cycle in cycles:
            x, y = _cumulative_potential_curve(
                x_maps[cycle], potential_maps[cycle], descending=descending
            )
            if x.size == 0:
                continue

            ax.plot(
                x,
                y,
                color=TECH_COLORS.get(cycle, "#333333"),
                linewidth=2.2,
                label=TECH_LABELS.get(cycle, cycle.upper()),
            )

            if annotate_80pct:
                target_y = 0.8 * y[-1]
                idx = int(np.searchsorted(y, target_y, side="left"))
                idx = min(idx, len(y) - 1)
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
                    color="black",
                    fontsize=11,
                    ha="center",
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85,
                    ),
                    zorder=5,
                )

        add_panel_letter(ax, panel_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_vs_potential_scatter_figure(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
    """Plot minimized LCOE against grid-cell potential."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for cycle in cycles:
        x = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        y = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        if not np.any(mask):
            continue

        ax.scatter(
            x[mask],
            y[mask],
            s=6,
            alpha=0.35,
            linewidths=0,
            color=TECH_COLORS.get(cycle, "#333333"),
            label=TECH_LABELS.get(cycle, cycle.upper()),
        )

    ax.set_xlabel("Minimized LCOE (€/MWh$_e$)")
    ax.set_ylabel("Minimized potential (GW per 10x10km grid cell)")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_lcoe_potential_heatmap_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    n_temp_bins: int = 25,
    n_lcoe_bins: int = 25,
):
    """Plot potential binned jointly by temperature and LCOE."""
    all_temp = []
    all_lcoe = []
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)
        if np.any(mask):
            all_temp.append(temp[mask])
            all_lcoe.append(lcoe[mask])

    if not all_temp or not all_lcoe:
        _schemas.write_blank_png(out_path)
        return

    temp_all = np.concatenate(all_temp)
    lcoe_all = np.concatenate(all_lcoe)

    temp_min = float(np.floor(np.nanmin(temp_all) / 5.0) * 5.0)
    temp_max = float(np.ceil(np.nanmax(temp_all) / 5.0) * 5.0)
    lcoe_min = float(np.nanmin(lcoe_all))
    lcoe_max = float(np.nanmax(lcoe_all))

    if not np.isfinite(temp_min) or not np.isfinite(temp_max) or temp_min == temp_max:
        _schemas.write_blank_png(out_path)
        return
    if not np.isfinite(lcoe_min) or not np.isfinite(lcoe_max) or lcoe_min == lcoe_max:
        _schemas.write_blank_png(out_path)
        return

    temp_edges = np.linspace(temp_min, temp_max, n_temp_bins + 1)
    lcoe_edges = np.linspace(lcoe_min, lcoe_max, n_lcoe_bins + 1)

    heatmaps = []
    vmax = 0.0
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)

        if np.any(mask):
            heatmap, _, _ = np.histogram2d(
                temp[mask], lcoe[mask], bins=[temp_edges, lcoe_edges], weights=pot[mask]
            )
        else:
            heatmap = np.zeros(
                (len(temp_edges) - 1, len(lcoe_edges) - 1), dtype=np.float64
            )

        heatmaps.append(heatmap)
        vmax = max(vmax, float(np.nanmax(heatmap)))

    if vmax <= 0:
        _schemas.write_blank_png(out_path)
        return

    fig, axes = plt.subplots(
        1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(MAP_CMAP).copy()
    cmap.set_bad("#D9D9D9")

    mesh = None
    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        heatmap = np.where(heatmaps[idx] > 0, heatmaps[idx].T, np.nan)
        mesh = ax.pcolormesh(
            temp_edges,
            lcoe_edges,
            heatmap,
            cmap=cmap,
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
        )
        ax.set_xlabel("Temperature at minimum LCOE (°C)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(False)

    axes[0].set_ylabel("Minimized LCOE (€/MWh$_e$)")

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.9, pad=0.04)
    cbar.set_label("Total potential per bin (GW)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_vs_lcoe_scatter_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
    """Plot reservoir temperature against minimized LCOE."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for cycle in cycles:
        x = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        y = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue

        ax.scatter(
            x[mask],
            y[mask],
            s=6,
            alpha=0.35,
            linewidths=0,
            color=TECH_COLORS.get(cycle, "#333333"),
            label=TECH_LABELS.get(cycle, cycle.upper()),
        )

    ax.set_xlabel("Temperature at minimum LCOE (°C)")
    ax.set_ylabel("Minimized LCOE (€/MWh$_e$)")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_lcoe_potential_scatter_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
    """Plot temperature and LCOE with potential represented by color."""
    potential_arrays = []
    for cycle in cycles:
        pot = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        pot = pot[np.isfinite(pot) & (pot > 0)]
        if pot.size:
            potential_arrays.append(pot)

    if not potential_arrays:
        _schemas.write_blank_png(out_path)
        return

    vmax = float(max(np.nanmax(p) for p in potential_arrays))
    if not np.isfinite(vmax) or vmax <= 0:
        _schemas.write_blank_png(out_path)
        return

    fig, axes = plt.subplots(
        1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(MAP_CMAP).copy()
    cmap.set_bad("#D9D9D9")

    scatter = None
    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        x = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        y = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        c = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c) & (c > 0)
        if not np.any(mask):
            continue

        scatter = ax.scatter(
            x[mask],
            y[mask],
            c=c[mask],
            s=7,
            alpha=0.45,
            linewidths=0,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
        )
        ax.set_xlabel("Temperature at minimum LCOE (°C)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(False)

    if scatter is None:
        plt.close(fig)
        _schemas.write_blank_png(out_path)
        return

    axes[0].set_ylabel("Minimized LCOE (€/MWh$_e$)")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.9, pad=0.04)
    cbar.set_label("Potential per grid cell (GW)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_lcoe_isopotential_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    *,
    n_temp_bins: int = 30,
    n_lcoe_bins: int = 30,
):
    """Plot cumulative-potential contours over temperature and LCOE."""
    all_temp = []
    all_lcoe = []
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)
        if np.any(mask):
            all_temp.append(temp[mask])
            all_lcoe.append(lcoe[mask])

    if not all_temp or not all_lcoe:
        _schemas.write_blank_png(out_path)
        return

    temp_all = np.concatenate(all_temp)
    lcoe_all = np.concatenate(all_lcoe)

    temp_min = float(np.floor(np.nanmin(temp_all) / 5.0) * 5.0)
    temp_max = float(np.ceil(np.nanmax(temp_all) / 5.0) * 5.0)
    lcoe_min = float(np.nanmin(lcoe_all))
    lcoe_max = float(np.nanmax(lcoe_all))

    if not np.isfinite(temp_min) or not np.isfinite(temp_max) or temp_min == temp_max:
        _schemas.write_blank_png(out_path)
        return
    if not np.isfinite(lcoe_min) or not np.isfinite(lcoe_max) or lcoe_min == lcoe_max:
        _schemas.write_blank_png(out_path)
        return

    temp_edges = np.linspace(temp_min, temp_max, n_temp_bins + 1)
    lcoe_edges = np.linspace(lcoe_min, lcoe_max, n_lcoe_bins + 1)
    temp_centers = 0.5 * (temp_edges[:-1] + temp_edges[1:])
    lcoe_centers = 0.5 * (lcoe_edges[:-1] + lcoe_edges[1:])

    surfaces = []
    vmax = 0.0
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = (
            np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        )
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)

        if np.any(mask):
            hist, _, _ = np.histogram2d(
                temp[mask], lcoe[mask], bins=[temp_edges, lcoe_edges], weights=pot[mask]
            )
            surface = hist[::-1, :].cumsum(axis=0)[::-1, :].cumsum(axis=1)
        else:
            surface = np.zeros((len(temp_centers), len(lcoe_centers)), dtype=np.float64)

        surfaces.append(surface)
        vmax = max(vmax, float(np.nanmax(surface)))

    if vmax <= 0:
        _schemas.write_blank_png(out_path)
        return

    n_levels = 7
    levels = np.linspace(0.0, vmax, n_levels + 1)
    if np.allclose(levels[0], levels[-1]):
        _schemas.write_blank_png(out_path)
        return

    fig, axes = plt.subplots(
        1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(MAP_CMAP).copy()
    cmap.set_bad("#D9D9D9")

    contourf = None
    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        surface = np.where(surfaces[idx] > 0, surfaces[idx].T, np.nan)
        contourf = ax.contourf(
            temp_centers, lcoe_centers, surface, levels=levels, cmap=cmap, extend="max"
        )
        contour = ax.contour(
            temp_centers,
            lcoe_centers,
            surface,
            levels=levels[1:-1],
            colors="white",
            linewidths=0.8,
            alpha=0.85,
        )
        ax.clabel(contour, inline=True, fontsize=10, fmt="%.0f")
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
        )
        ax.set_xlabel("Temperature threshold (°C)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(False)

    axes[0].set_ylabel("LCOE threshold (€/MWh$_e$)")

    cbar = fig.colorbar(contourf, ax=axes, shrink=0.9, pad=0.04)
    cbar.set_label("Cumulative potential above temperature and below LCOE (GW)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_overview_figure(
    lcoe_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
    countries: gpd.GeoDataFrame,
    max_depth: float,
):
    """Plot cycle-specific LCOE maps and the least-cost technology."""
    panels = cycles + ["winner"]
    n_panels = len(panels)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    reference_da = lcoe_maps[cycles[0]]
    extent = _map_extent(reference_da)
    # Compute the data aspect ratio once and apply it uniformly
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])  # height/width

    panel_w = 4.5
    panel_h = panel_w * data_aspect
    gap = 0.15  # gap between panels in inches
    margin_l = 0.1
    margin_r = 0.1
    margin_t = 0.1
    margin_b = 0.9  # room for colorbar + label

    fig_w = margin_l + ncols * panel_w + (ncols - 1) * gap + margin_r
    fig_h = margin_t + nrows * panel_h + (nrows - 1) * gap + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Place axes manually in figure-fraction coordinates
    def make_ax(row, col):
        x0 = (margin_l + col * (panel_w + gap)) / fig_w
        y0 = (margin_b + (nrows - 1 - row) * (panel_h + gap)) / fig_h
        w = panel_w / fig_w
        h = panel_h / fig_h
        return fig.add_axes([x0, y0, w, h])

    axes = []
    for i in range(n_panels):
        row, col = divmod(i, ncols)
        axes.append(make_ax(row, col))

    lcoe_cmap = plt.get_cmap(LCOE_OVERVIEW_CMAP).copy()
    lcoe_cmap.set_bad((0, 0, 0, 0))

    winner_cmap = ListedColormap([TECH_COLORS[c] for c in cycles])
    winner_cmap.set_bad((0, 0, 0, 0))

    plot_arrays = {cycle: _plot_values(lcoe_maps[cycle], countries) for cycle in cycles}
    lcoe_vmin, lcoe_data_vmax = _finite_limits(plot_arrays.values())
    lcoe_vmax = (
        min(lcoe_data_vmax, LCOE_OVERVIEW_COLORBAR_CUTOFF)
        if LCOE_OVERVIEW_COLORBAR_CUTOFF is not None
        else lcoe_data_vmax
    )
    if lcoe_vmin >= lcoe_vmax:
        lcoe_vmin = 0.0
    lcoe_norm = lcoe_norm = PowerNorm(
        gamma=LCOE_OVERVIEW_NORM_GAMMA, vmin=70, vmax=LCOE_OVERVIEW_COLORBAR_CUTOFF
    )
    lcoe_colorbar_extend = "max" if lcoe_data_vmax > lcoe_vmax else "neither"
    plot_maps = {
        cycle: xr.DataArray(values, dims=("y", "x"))
        for cycle, values in plot_arrays.items()
    }
    winner = _winner_map(plot_maps, cycles)
    winner_masked = np.where(winner >= 0, winner, np.nan)

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        _decorate_map(
            ax,
            countries,
            extent,
            _map_crs(reference_da),
            unavailable_color=LCOE_UNAVAILABLE_COLOR,
        )

        if panel == "winner":
            ax.imshow(
                winner_masked,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=winner_cmap,
                vmin=-0.5,
                vmax=len(cycles) - 0.5,
                aspect="auto",  # axes box is already the right shape
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
                aspect="auto",  # axes box is already the right shape
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
            color="black",
        )
        _clean_map_axes(ax)

    # Colorbar: placed manually in figure coordinates below all panels
    cbar_x0 = (margin_l + 0.1 * fig_w) / fig_w
    cbar_y0 = 0.38 / fig_h
    cbar_w = 1 - cbar_x0 - (margin_r + 0.1 * fig_w) / fig_w
    cbar_h_frac = 0.04 / fig_h * fig_w  # keep it thin

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, cbar_h_frac])
    sm = plt.cm.ScalarMappable(norm=lcoe_norm, cmap=lcoe_cmap)
    cbar = fig.colorbar(
        sm, cax=cax, orientation="horizontal", extend=lcoe_colorbar_extend
    )
    cbar.set_label("Levelized Cost of Electricity (€/MWh$_e$)")
    cbar.ax.tick_params(length=0, labelsize=11)

    legend_handles = [
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
    legend_handles.append(
        Patch(
            facecolor=LCOE_UNAVAILABLE_COLOR,
            edgecolor=COUNTRY_EDGE_COLOR,
            label=f"Not possible below {depth_label}",
        )
    )
    axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#AAAAAA",
        fontsize=10,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# 3D -> 2D aggregation
# --------------------------


def aggregate_3d_to_2d_lcoe_min(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
    """Select the minimum-LCOE depth independently in every grid cell."""
    temperature = ds["temperature"]
    lcoe = ds["lcoe"]
    pot = ds["potential"]

    lcoe_filled = lcoe.where(np.isfinite(lcoe), np.inf)
    argmin = lcoe_filled.argmin(dim=depth_dim)

    lcoe_min = lcoe.min(dim=depth_dim, skipna=True)
    valid = np.isfinite(lcoe_min)

    lcoe_2d = lcoe.isel({depth_dim: argmin}).where(valid)
    pot_2d = pot.isel({depth_dim: argmin}).where(valid)
    temperature_at_min = temperature.isel({depth_dim: argmin}).where(valid)

    depth_vals = ds[depth_dim]
    depth_at_min = depth_vals.isel({depth_dim: argmin}).where(valid)

    out = xr.Dataset(
        data_vars=dict(
            temperature_at_min=temperature_at_min.astype(np.float32),
            lcoe_2d=lcoe_2d.astype(np.float32),
            potential_2d=pot_2d.astype(np.float32),
            depth_at_min=depth_at_min.astype(np.float32),
        ),
        coords=dict(x=ds["x"], y=ds["y"]),
        attrs=dict(ds.attrs),
    )
    out["temperature_at_min"].attrs.update(temperature.attrs)
    out["lcoe_2d"].attrs.update(lcoe.attrs)
    out["potential_2d"].attrs.update(pot.attrs)
    out["depth_at_min"].attrs.update(getattr(depth_vals, "attrs", {}))
    return out


def aggregate_3d_to_2d_depth_categories(
    ds: xr.Dataset, depth_categories: Sequence[DepthCategory], depth_dim: str
) -> dict[str, xr.Dataset]:
    """Select the minimum-LCOE depth within each configured depth interval."""
    depth = ds[depth_dim]
    out: dict[str, xr.Dataset] = {}

    for cat in depth_categories:
        sel = (depth >= cat.zmin) & (depth < cat.zmax)

        if int(sel.sum()) == 0:
            empty = xr.Dataset(
                data_vars=dict(
                    temperature_at_min=(
                        ("y", "x"),
                        np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32),
                    ),
                    lcoe_2d=(
                        ("y", "x"),
                        np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32),
                    ),
                    potential_2d=(
                        ("y", "x"),
                        np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32),
                    ),
                    depth_at_min=(
                        ("y", "x"),
                        np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32),
                    ),
                ),
                coords=dict(x=ds["x"], y=ds["y"]),
                attrs=dict(ds.attrs),
            )
            out[cat.label] = empty
            continue

        sub = ds.sel({depth_dim: depth.where(sel, drop=True)})
        out[cat.label] = aggregate_3d_to_2d_lcoe_min(sub, depth_dim=depth_dim)

    return out


def _cycle_from_cube_path(p: str) -> str:
    name = Path(p).name
    return name.replace("_lcoe_cube.nc", "")


def _cycle_from_tif_name(path: str, suffix: str | None = None) -> str:
    """Extract the cycle token from a 2-D output filename.

    Examples include:
      kalina_2d_lcoe.tif
      kalina_2d_potential.tif
      kalina_2d_0-3km_lcoe.tif
      kalina_2d_3-6km_potential.tif.

    `suffix` is kept for backward compatibility but is no longer required.
    Returns the 'tech/cycle' part (e.g., 'kalina').
    """
    name = os.path.basename(path)

    m = re.match(
        r"^(?P<cycle>[A-Za-z0-9]+)_2d_(?:(?P<cat>.+)_)?(?P<kind>lcoe|potential)\.tif$",
        name,
    )

    # If caller provided suffix, you can optionally validate 'kind' against it
    return m.group("cycle").lower()


def main():
    """Reduce all Snakemake-requested cubes and write their outputs."""
    # Selected cycles from config (already normalized in Snakefile, but be defensive)
    cycles = [str(c).strip().lower() for c in (snakemake.params.get("cycles"))]

    method_2d_aggregation = (
        snakemake.params.get("method_2d_aggregation")
        or snakemake.config["method_2d_aggregation"]
    )
    depth_categories = _parse_depth_categories(
        snakemake.params.get("depth_categories", [])
    )

    # CRS (optional)
    crs = snakemake.config.get("crs") or snakemake.params.get("crs")
    max_depth = float(snakemake.params.get("max_depth", snakemake.config["max_depth"]))

    # Inputs/outputs are lists in the new rule
    cubes: list[str] = list(snakemake.input.cubes)
    lcoe_tifs: list[str] = list(snakemake.output.lcoe_tifs)
    pot_tifs: list[str] = list(snakemake.output.pot_tifs)
    depth_profile_figs: list[str] = (
        list(snakemake.output.depth_profile_figs)
        if hasattr(snakemake.output, "depth_profile_figs")
        else []
    )

    if len(cubes) != len(cycles):
        # Try to match by filename cycle token instead of position
        cubes_by_cycle = {_cycle_from_cube_path(p): p for p in cubes}
    else:
        cubes_by_cycle = {c: p for c, p in zip(cycles, cubes)}

    out_lcoe_by_cycle = {_cycle_from_tif_name(p, "_2d_lcoe.tif"): p for p in lcoe_tifs}
    out_pot_by_cycle = {
        _cycle_from_tif_name(p, "_2d_potential.tif"): p for p in pot_tifs
    }
    out_depth_profile_by_cycle = {
        Path(p).stem.replace("depth_profile_", "").lower(): p
        for p in depth_profile_figs
    }
    lcoe_maps_2d: dict[str, xr.DataArray] = {}
    potential_maps_2d: dict[str, xr.DataArray] = {}
    depth_maps_2d: dict[str, xr.DataArray] = {}
    temperature_maps_2d: dict[str, xr.DataArray] = {}
    potential_profiles_3d: dict[str, np.ndarray] = {}
    depths_m: np.ndarray | None = None

    for cycle in cycles:
        pass

    # Choose an outdir for extra products (depth_at_min.tif, PNGs) from declared outputs.
    # This keeps module pathvars in control of where products are written.
    extra_outdir = Path(next(iter(lcoe_tifs))).parent

    for cycle in cycles:
        cube_path = cubes_by_cycle.get(cycle)

        ds_3d = xr.open_dataset(cube_path)

        if method_2d_aggregation == "LCOE_minimization":
            if depths_m is None:
                depths_m = ds_3d["depth_m"].values.astype(np.float64)
            potential_profiles_3d[cycle] = _finite_mean_by_depth(
                ds_3d["potential"], value_scale=1000.0
            )
            ds_2d = aggregate_3d_to_2d_lcoe_min(ds_3d, depth_dim="depth_m")
            temperature_maps_2d[cycle] = ds_2d["temperature_at_min"].copy()
            lcoe_maps_2d[cycle] = ds_2d["lcoe_2d"].copy()
            potential_maps_2d[cycle] = ds_2d["potential_2d"].copy()
            depth_maps_2d[cycle] = ds_2d["depth_at_min"].copy()

            # Write exactly the outputs Snakemake expects
            write_geotiff(ds_2d["lcoe_2d"], Path(out_lcoe_by_cycle[cycle]), crs=crs)
            write_geotiff(ds_2d["potential_2d"], Path(out_pot_by_cycle[cycle]), crs=crs)

            # Optional extras (not DAG-critical)
            write_geotiff(
                ds_2d["temperature_at_min"],
                extra_outdir / f"{cycle}_2d_temperature_at_min.tif",
                crs=crs,
            )
            write_geotiff(
                ds_2d["depth_at_min"],
                extra_outdir / f"{cycle}_2d_depth_at_min.tif",
                crs=crs,
            )

        elif method_2d_aggregation == "depth_categories":
            # NOTE: your Snakefile outputs currently do NOT include category-suffixed files.
            # So this branch would need Snakefile changes to declare those outputs.
            ds_dict = aggregate_3d_to_2d_depth_categories(
                ds_3d, depth_categories=depth_categories, depth_dim="depth_m"
            )
            # Write category outputs as "extras" only unless you update Snakefile
            for cat_label, ds_cat in ds_dict.items():
                base = f"{cycle}_2d_{cat_label}"
                write_geotiff(
                    ds_cat["temperature_at_min"],
                    extra_outdir / f"{base}_temperature_at_min.tif",
                    crs=crs,
                )
                write_geotiff(
                    ds_cat["lcoe_2d"], extra_outdir / f"{base}_lcoe.tif", crs=crs
                )
                write_geotiff(
                    ds_cat["potential_2d"],
                    extra_outdir / f"{base}_potential.tif",
                    crs=crs,
                )
                write_geotiff(
                    ds_cat["depth_at_min"],
                    extra_outdir / f"{base}_depth_at_min.tif",
                    crs=crs,
                )

    if method_2d_aggregation == "LCOE_minimization" and hasattr(
        snakemake.output, "overview_fig"
    ):
        countries = _country_layer(
            Path(snakemake.input.shapes), lcoe_maps_2d[cycles[0]]
        )
        make_lcoe_overview_figure(
            lcoe_maps=lcoe_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.overview_fig),
            countries=countries,
            max_depth=max_depth,
        )
        make_lcoe_vs_potential_scatter_figure(
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.scatter_fig),
        )
        make_lcoe_potential_boxplot_figure(
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.technology_boxplot_fig),
        )
        make_lcoe_potential_split_boxplot_figure(
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.technology_split_boxplot_fig),
        )
        make_lcoe_potential_cumulative_four_panel_figure(
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            temperature_maps=temperature_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.technology_cumulative_four_panel_fig),
        )
        make_average_potential_by_depth_figure(
            potential_profiles=potential_profiles_3d,
            depths_m=depths_m,
            cycles=cycles,
            out_path=Path(snakemake.output.avg_potential_depth_fig),
        )
        for cycle in cycles:
            if cycle in out_depth_profile_by_cycle:
                make_depth_profile_figure(
                    lcoe_map=lcoe_maps_2d[cycle],
                    potential_map=potential_maps_2d[cycle],
                    depth_map=depth_maps_2d[cycle],
                    cycle=cycle,
                    out_path=Path(out_depth_profile_by_cycle[cycle]),
                )
        make_cumulative_potential_figure(
            x_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.cumulative_curve_fig),
            xlabel="LCOE threshold (€/MWh$_e$)",
        )
        make_cumulative_potential_figure(
            x_maps=depth_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.cumulative_depth_fig),
            xlabel="Depth threshold (km)",
            x_scale=1000.0,
        )
        make_cumulative_potential_figure(
            x_maps=temperature_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.cumulative_temperature_fig),
            xlabel="Temperature threshold (°C)",
            descending=True,
            ylabel="Cumulative potential above temperature (GW)",
        )
        make_temperature_lcoe_potential_heatmap_figure(
            temperature_maps=temperature_maps_2d,
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.temperature_lcoe_potential_fig),
        )
        make_temperature_vs_lcoe_scatter_figure(
            temperature_maps=temperature_maps_2d,
            lcoe_maps=lcoe_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.temperature_lcoe_scatter_fig),
        )
        make_temperature_lcoe_isopotential_figure(
            temperature_maps=temperature_maps_2d,
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.temperature_lcoe_isopotential_fig),
        )
        make_temperature_lcoe_potential_scatter_figure(
            temperature_maps=temperature_maps_2d,
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.temperature_lcoe_potential_scatter_fig),
        )
        make_three_panel_map_figure(
            maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.potential_overview_fig),
            colorbar_label="Potential (GW per grid cell)",
            cmap=MAP_CMAP,
            value_scale=1000.0,
            countries=countries,
        )
        shutil.copyfile(
            snakemake.output.potential_overview_fig,
            snakemake.output.potential_presentation_fig,
        )
        make_three_panel_map_figure(
            maps=depth_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.depth_overview_fig),
            colorbar_label="Depth at minimum LCOE (km)",
            cmap=LCOE_OVERVIEW_CMAP,
            value_scale=1000.0,
            countries=countries,
        )


if __name__ == "__main__":
    main()
