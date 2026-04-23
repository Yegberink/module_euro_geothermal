from typing import Dict, List, Sequence
from attr import dataclass
import xarray as xr
import numpy as np
from pathlib import Path
import rioxarray
import os
import re
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
TECH_COLORS = {"orc": "#0072B2", "flash": "#E69F00", "kalina": "#009E73"}
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
LCOE_BOUNDS = [0.0, 100.0, 200.0, 400.0, 600.0, 1.0e9]
LCOE_TICKS = [50.0, 150.0, 300.0, 500.0, 800.0]
LCOE_TICKLABELS = ["0-100", "100-200", "200-400", "400-600", "≥600"]
LCOE_COLORS = ["#481567", "#414487", "#2A788E", "#22A884", "#FDE725"]
MAP_CMAP = "viridis"

# --------------------------
# Data classes
# --------------------------

@dataclass(frozen=True)
class DepthCategory:
    zmin: float
    zmax: float

    @property
    def label(self) -> str:
        return f"{self.zmin/1000:g}-{self.zmax/1000:g}km"

# --------------------------
# Helpers
# --------------------------

def _parse_depth_categories(raw) -> List[DepthCategory]:
    raw = raw or []
    out: List[DepthCategory] = []
    for pair in raw:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"Invalid depth category entry: {pair!r} (expected [zmin,zmax])")
        out.append(DepthCategory(float(pair[0]), float(pair[1])))
    return out


def _ensure_raster_metadata(
    da: xr.DataArray,
    *,
    x_name: str = "x",
    y_name: str = "y",
    crs: str | None = None,
) -> xr.DataArray:
    # Ensure dims are (y, x) for rasters
    if da.dims != (y_name, x_name):
        da = da.transpose(y_name, x_name)

    da = da.rio.set_spatial_dims(x_dim=x_name, y_dim=y_name, inplace=False)

    crs_guess = crs or da.attrs.get("crs") or da.attrs.get("spatial_ref") or da.attrs.get("epsg")
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
    da2 = _ensure_raster_metadata(da, x_name=x_name, y_name=y_name, crs=crs)

    if np.isnan(nodata):
        da2 = da2.rio.write_nodata(np.nan, inplace=False)
    else:
        da2 = da2.rio.write_nodata(nodata, inplace=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    da2.rio.to_raster(str(out_path), compress="deflate")


def quick_map_png(
    da: xr.DataArray,
    out_path: Path,
    *,
    title: str = "",
    x_name: str = "x",
    y_name: str = "y",
    vmin=None,
    vmax=None,
):
    import matplotlib.pyplot as plt

    if da.dims != (y_name, x_name):
        da = da.transpose(y_name, x_name)

    x = da[x_name].values
    y = da[y_name].values
    extent = [x.min(), x.max(), y.min(), y.max()]

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        da.values,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        cmap=MAP_CMAP,
    )
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.colorbar(im, ax=ax, label=da.name or "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def _winner_map(lcoe_maps: dict[str, xr.DataArray], cycles: list[str]) -> np.ndarray:
    stack = np.stack([lcoe_maps[cycle].values for cycle in cycles], axis=0).astype(np.float32)
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
    potential = np.asarray(potential_da.values, dtype=np.float64).ravel() / potential_scale
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
):
    n_panels = len(cycles)
    ncols = n_panels
    nrows = 1

    reference_da = maps[cycles[0]]
    extent = _map_extent(reference_da)
    data_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    panel_w = 4.5
    panel_h = panel_w * data_aspect
    gap = 0.15
    margin_l = 0.1
    margin_r = 0.1
    margin_t = 0.1
    margin_b = 0.9

    fig_w = margin_l + ncols * panel_w + (ncols - 1) * gap + margin_r
    fig_h = margin_t + nrows * panel_h + (nrows - 1) * gap + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))

    def make_ax(col):
        x0 = (margin_l + col * (panel_w + gap)) / fig_w
        y0 = margin_b / fig_h
        w = panel_w / fig_w
        h = panel_h / fig_h
        return fig.add_axes([x0, y0, w, h])

    axes = [make_ax(i) for i in range(n_panels)]
    scaled_arrays = [(maps[cycle].values / value_scale).astype(np.float32) for cycle in cycles]
    vmin, vmax = _finite_limits(scaled_arrays)
    map_cmap = plt.get_cmap(cmap).copy()
    map_cmap.set_bad("#D9D9D9")

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        ax.set_facecolor("#D9D9D9")
        im = ax.imshow(
            scaled_arrays[idx],
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=map_cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="black",
        )
        _clean_map_axes(ax)
        ax.set_aspect("equal")

    cax = fig.add_axes([0.14, 0.075, 0.72, 0.04])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(length=0, labelsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
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
    fig, ax = plt.subplots(figsize=(6, 4))
    is_depth_curve = "Depth" in xlabel
    is_temperature_curve = "Temperature" in xlabel

    for cycle in cycles:
        x, y = _cumulative_potential_curve(
            x_maps[cycle],
            potential_maps[cycle],
            x_scale=x_scale,
            descending=descending,
        )
        if x.size == 0:
            continue

        ax.plot(
            x,
            y,
            color=TECH_COLORS.get(cycle, "#333333"),
            linewidth=2.0,
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
        label_value = f"{x[idx]:.1f} km" if is_depth_curve else f"{x[idx]:.0f} €/MWh$_e$"
        if is_temperature_curve:
            label_value = f"{x[idx]:.0f} °C"
        ax.annotate(
            label_value,
            (x[idx], y[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            color="black",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if descending:
        ax.invert_xaxis()
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_average_potential_by_depth_figure(
    potential_profiles: dict[str, np.ndarray],
    depths_m: np.ndarray,
    cycles: list[str],
    out_path: Path,
):
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
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_vs_potential_scatter_figure(
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
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
    ax.set_ylabel("Minimized potential (GW)")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    all_temp = []
    all_lcoe = []
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)
        if np.any(mask):
            all_temp.append(temp[mask])
            all_lcoe.append(lcoe[mask])

    if not all_temp or not all_lcoe:
        return

    temp_all = np.concatenate(all_temp)
    lcoe_all = np.concatenate(all_lcoe)

    temp_min = float(np.floor(np.nanmin(temp_all) / 5.0) * 5.0)
    temp_max = float(np.ceil(np.nanmax(temp_all) / 5.0) * 5.0)
    lcoe_min = float(np.nanmin(lcoe_all))
    lcoe_max = float(np.nanmax(lcoe_all))

    if not np.isfinite(temp_min) or not np.isfinite(temp_max) or temp_min == temp_max:
        return
    if not np.isfinite(lcoe_min) or not np.isfinite(lcoe_max) or lcoe_min == lcoe_max:
        return

    temp_edges = np.linspace(temp_min, temp_max, n_temp_bins + 1)
    lcoe_edges = np.linspace(lcoe_min, lcoe_max, n_lcoe_bins + 1)

    heatmaps = []
    vmax = 0.0
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)

        if np.any(mask):
            heatmap, _, _ = np.histogram2d(
                temp[mask],
                lcoe[mask],
                bins=[temp_edges, lcoe_edges],
                weights=pot[mask],
            )
        else:
            heatmap = np.zeros((len(temp_edges) - 1, len(lcoe_edges) - 1), dtype=np.float64)

        heatmaps.append(heatmap)
        vmax = max(vmax, float(np.nanmax(heatmap)))

    if vmax <= 0:
        return

    fig, axes = plt.subplots(1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True)
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
            fontsize=11,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_vs_lcoe_scatter_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
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
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temperature_lcoe_potential_scatter_figure(
    temperature_maps: dict[str, xr.DataArray],
    lcoe_maps: dict[str, xr.DataArray],
    potential_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
    potential_arrays = []
    for cycle in cycles:
        pot = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        pot = pot[np.isfinite(pot) & (pot > 0)]
        if pot.size:
            potential_arrays.append(pot)

    if not potential_arrays:
        return

    vmax = float(max(np.nanmax(p) for p in potential_arrays))
    if not np.isfinite(vmax) or vmax <= 0:
        return

    fig, axes = plt.subplots(1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True)
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
            fontsize=11,
            fontweight="bold",
            color="black",
        )
        ax.set_xlabel("Temperature at minimum LCOE (°C)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.grid(False)

    if scatter is None:
        plt.close(fig)
        return

    axes[0].set_ylabel("Minimized LCOE (€/MWh$_e$)")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.9, pad=0.04)
    cbar.set_label("Potential per grid cell (GW)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    all_temp = []
    all_lcoe = []
    for cycle in cycles:
        temp = np.asarray(temperature_maps[cycle].values, dtype=np.float64).ravel()
        lcoe = np.asarray(lcoe_maps[cycle].values, dtype=np.float64).ravel()
        pot = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)
        if np.any(mask):
            all_temp.append(temp[mask])
            all_lcoe.append(lcoe[mask])

    if not all_temp or not all_lcoe:
        return

    temp_all = np.concatenate(all_temp)
    lcoe_all = np.concatenate(all_lcoe)

    temp_min = float(np.floor(np.nanmin(temp_all) / 5.0) * 5.0)
    temp_max = float(np.ceil(np.nanmax(temp_all) / 5.0) * 5.0)
    lcoe_min = float(np.nanmin(lcoe_all))
    lcoe_max = float(np.nanmax(lcoe_all))

    if not np.isfinite(temp_min) or not np.isfinite(temp_max) or temp_min == temp_max:
        return
    if not np.isfinite(lcoe_min) or not np.isfinite(lcoe_max) or lcoe_min == lcoe_max:
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
        pot = np.asarray(potential_maps[cycle].values, dtype=np.float64).ravel() / 1000.0
        mask = np.isfinite(temp) & np.isfinite(lcoe) & np.isfinite(pot) & (pot > 0)

        if np.any(mask):
            hist, _, _ = np.histogram2d(
                temp[mask],
                lcoe[mask],
                bins=[temp_edges, lcoe_edges],
                weights=pot[mask],
            )
            surface = hist[::-1, :].cumsum(axis=0)[::-1, :].cumsum(axis=1)
        else:
            surface = np.zeros((len(temp_centers), len(lcoe_centers)), dtype=np.float64)

        surfaces.append(surface)
        vmax = max(vmax, float(np.nanmax(surface)))

    if vmax <= 0:
        return

    n_levels = 7
    levels = np.linspace(0.0, vmax, n_levels + 1)
    if np.allclose(levels[0], levels[-1]):
        return

    fig, axes = plt.subplots(1, len(cycles), figsize=(4.8 * len(cycles), 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(MAP_CMAP).copy()
    cmap.set_bad("#D9D9D9")

    contourf = None
    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        surface = np.where(surfaces[idx] > 0, surfaces[idx].T, np.nan)
        contourf = ax.contourf(
            temp_centers,
            lcoe_centers,
            surface,
            levels=levels,
            cmap=cmap,
            extend="max",
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
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f")
        ax.text(
            0.02,
            0.98,
            f"{PANEL_LETTERS[idx]}) {TECH_LABELS.get(cycle, cycle.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_lcoe_overview_figure(
    lcoe_maps: dict[str, xr.DataArray],
    cycles: list[str],
    out_path: Path,
):
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
    cbar_h = 0.5
    gap = 0.15          # gap between panels in inches
    margin_l = 0.1
    margin_r = 0.1
    margin_t = 0.1
    margin_b = 0.9      # room for colorbar + label

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

    lcoe_cmap = ListedColormap(LCOE_COLORS)
    lcoe_cmap.set_bad("#D9D9D9")
    lcoe_norm = BoundaryNorm(LCOE_BOUNDS, lcoe_cmap.N)

    winner_cmap = ListedColormap([TECH_COLORS[c] for c in cycles])
    winner_cmap.set_bad("#D9D9D9")

    winner = _winner_map(lcoe_maps, cycles)
    winner_masked = np.where(winner >= 0, winner, np.nan)

    for idx, panel in enumerate(panels):
        ax = axes[idx]
        ax.set_facecolor("#D9D9D9")

        if panel == "winner":
            ax.imshow(
                winner_masked,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=winner_cmap,
                vmin=-0.5,
                vmax=len(cycles) - 0.5,
                aspect="auto",      # axes box is already the right shape
            )
            title = "Cheapest technology"
        else:
            da = lcoe_maps[panel]
            ax.imshow(
                da.values,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap=lcoe_cmap,
                norm=lcoe_norm,
                aspect="auto",      # axes box is already the right shape
            )
            title = TECH_LABELS.get(panel, panel.upper())

        ax.text(
            0.02, 0.98,
            f"{PANEL_LETTERS[idx]}) {title}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold", color="black",
        )
        _clean_map_axes(ax)

    # Colorbar: placed manually in figure coordinates below all panels
    cbar_x0 = (margin_l + 0.1 * fig_w) / fig_w
    cbar_y0 = 0.38 / fig_h
    cbar_w = 1 - cbar_x0 - (margin_r + 0.1 * fig_w) / fig_w
    cbar_h_frac = 0.04 / fig_h * fig_w  # keep it thin

    cax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, cbar_h_frac])
    sm = plt.cm.ScalarMappable(norm=lcoe_norm, cmap=lcoe_cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.set_ticks(LCOE_TICKS)
    cbar.set_ticklabels(LCOE_TICKLABELS)
    cbar.set_label("Levelized Cost of Electricity (€/MWh$_e$)")
    cbar.ax.tick_params(length=0, labelsize=9)

    legend_handles = [
        Patch(facecolor=TECH_COLORS[c], edgecolor="none", label=TECH_LABELS.get(c, c.upper()))
        for c in cycles
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True, facecolor="white", framealpha=0.9,
        edgecolor="#AAAAAA", fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
# --------------------------
# 3D -> 2D aggregation
# --------------------------

def aggregate_3d_to_2d_lcoe_min(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
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
    ds: xr.Dataset,
    depth_categories: Sequence[DepthCategory],
    depth_dim: str,
) -> Dict[str, xr.Dataset]:

    depth = ds[depth_dim]
    out: Dict[str, xr.Dataset] = {}

    for cat in depth_categories:
        sel = (depth >= cat.zmin) & (depth < cat.zmax)

        if int(sel.sum()) == 0:
            empty = xr.Dataset(
                data_vars=dict(
                    temperature_at_min=(("y", "x"), np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32)),
                    lcoe_2d=(("y", "x"), np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32)),
                    potential_2d=(("y", "x"), np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32)),
                    depth_at_min=(("y", "x"), np.full((ds.sizes["y"], ds.sizes["x"]), np.nan, np.float32)),
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
    if not name.endswith("_lcoe_cube.nc"):
        raise ValueError(f"Unexpected cube filename (expected *_lcoe_cube.nc): {p}")
    return name.replace("_lcoe_cube.nc", "")

def _cycle_from_tif_name(path: str, suffix: str | None = None) -> str:
    """
    Extract cycle token from filenames like:
      kalina_2d_lcoe.tif
      kalina_2d_potential.tif
      kalina_2d_0-3km_lcoe.tif
      kalina_2d_3-6km_potential.tif

    `suffix` is kept for backward compatibility but is no longer required.
    Returns the 'tech/cycle' part (e.g., 'kalina').
    """
    name = os.path.basename(path)

    m = re.match(
        r"^(?P<cycle>[A-Za-z0-9]+)_2d_(?:(?P<cat>.+)_)?(?P<kind>lcoe|potential)\.tif$",
        name
    )
    if not m:
        raise ValueError(f"Unexpected output tif name: {name}")

    # If caller provided suffix, you can optionally validate 'kind' against it
    if suffix is not None:
        if suffix.endswith("_2d_lcoe.tif") and m.group("kind") != "lcoe":
            raise ValueError(f"Unexpected output tif name: {name} (expected *{suffix})")
        if suffix.endswith("_2d_potential.tif") and m.group("kind") != "potential":
            raise ValueError(f"Unexpected output tif name: {name} (expected *{suffix})")

    return m.group("cycle").lower()



def main():
    # Selected cycles from config (already normalized in Snakefile, but be defensive)
    cycles = [str(c).strip().lower() for c in (snakemake.params.get("cycles"))]

    method_2d_aggregation = snakemake.params.get("method_2d_aggregation") or snakemake.config["method_2d_aggregation"]
    depth_categories = _parse_depth_categories(snakemake.params.get("depth_categories", []))

    # CRS (optional)
    crs = snakemake.config.get("crs") or snakemake.params.get("crs")

    # Inputs/outputs are lists in the new rule
    cubes: List[str] = list(snakemake.input.cubes)
    lcoe_tifs: List[str] = list(snakemake.output.lcoe_tifs)
    pot_tifs: List[str] = list(snakemake.output.pot_tifs)

    if len(cubes) != len(cycles):
        # Try to match by filename cycle token instead of position
        cubes_by_cycle = { _cycle_from_cube_path(p): p for p in cubes }
    else:
        cubes_by_cycle = { c: p for c, p in zip(cycles, cubes) }

    out_lcoe_by_cycle = {_cycle_from_tif_name(p, "_2d_lcoe.tif"): p for p in lcoe_tifs}
    out_pot_by_cycle  = {_cycle_from_tif_name(p, "_2d_potential.tif"): p for p in pot_tifs}
    lcoe_maps_2d: dict[str, xr.DataArray] = {}
    potential_maps_2d: dict[str, xr.DataArray] = {}
    depth_maps_2d: dict[str, xr.DataArray] = {}
    temperature_maps_2d: dict[str, xr.DataArray] = {}
    potential_profiles_3d: dict[str, np.ndarray] = {}
    depths_m: np.ndarray | None = None

    for cycle in cycles:
        if cycle not in out_lcoe_by_cycle or cycle not in out_pot_by_cycle:
            raise ValueError(
                f"Outputs missing for cycle={cycle}. "
                f"Expected {cycle}_2d_lcoe.tif and {cycle}_2d_potential.tif "
                f"but got lcoe={sorted(out_lcoe_by_cycle)} pot={sorted(out_pot_by_cycle)}"
        )

    # Choose an outdir for extra products (depth_at_min.tif, PNGs).
    # Keep it scenario-specific so different scenarios do not overwrite one another.
    extra_outdir = (
        Path("results")
        / snakemake.wildcards["shapes"]
        / "scenarios"
        / snakemake.wildcards["scenario"]
        / "2d_potential_cost"
    )
    (extra_outdir / "figures").mkdir(parents=True, exist_ok=True)

    for cycle in cycles:
        cube_path = cubes_by_cycle.get(cycle)
        if cube_path is None:
            raise ValueError(f"Missing input cube for cycle={cycle}. Have cubes: {sorted(cubes_by_cycle)}")

        ds_3d = xr.open_dataset(cube_path)

        if method_2d_aggregation == "LCOE_minimization":
            if depths_m is None:
                depths_m = ds_3d["depth_m"].values.astype(np.float64)
            potential_profiles_3d[cycle] = _finite_mean_by_depth(ds_3d["potential"], value_scale=1000.0)
            ds_2d = aggregate_3d_to_2d_lcoe_min(ds_3d, depth_dim="depth_m")
            temperature_maps_2d[cycle] = ds_2d["temperature_at_min"].copy()
            lcoe_maps_2d[cycle] = ds_2d["lcoe_2d"].copy()
            potential_maps_2d[cycle] = ds_2d["potential_2d"].copy()
            depth_maps_2d[cycle] = ds_2d["depth_at_min"].copy()

            # Write exactly the outputs Snakemake expects
            write_geotiff(ds_2d["lcoe_2d"], Path(out_lcoe_by_cycle[cycle]), crs=crs)
            write_geotiff(ds_2d["potential_2d"], Path(out_pot_by_cycle[cycle]), crs=crs)

            # Optional extras (not DAG-critical)
            write_geotiff(ds_2d["temperature_at_min"], extra_outdir / f"{cycle}_2d_temperature_at_min.tif", crs=crs)
            write_geotiff(ds_2d["depth_at_min"], extra_outdir / f"{cycle}_2d_depth_at_min.tif", crs=crs)

        elif method_2d_aggregation == "depth_categories":
            # NOTE: your Snakefile outputs currently do NOT include category-suffixed files.
            # So this branch would need Snakefile changes to declare those outputs.
            ds_dict = aggregate_3d_to_2d_depth_categories(
                ds_3d,
                depth_categories=depth_categories,
                depth_dim="depth_m",
            )
            # Write category outputs as "extras" only unless you update Snakefile
            for cat_label, ds_cat in ds_dict.items():
                base = f"{cycle}_2d_{cat_label}"
                write_geotiff(ds_cat["temperature_at_min"], extra_outdir / f"{base}_temperature_at_min.tif", crs=crs)
                write_geotiff(ds_cat["lcoe_2d"], extra_outdir / f"{base}_lcoe.tif", crs=crs)
                write_geotiff(ds_cat["potential_2d"], extra_outdir / f"{base}_potential.tif", crs=crs)
                write_geotiff(ds_cat["depth_at_min"], extra_outdir / f"{base}_depth_at_min.tif", crs=crs)

    if method_2d_aggregation == "LCOE_minimization" and hasattr(snakemake.output, "overview_fig"):
        make_lcoe_overview_figure(
            lcoe_maps=lcoe_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.overview_fig),
        )
        make_lcoe_vs_potential_scatter_figure(
            lcoe_maps=lcoe_maps_2d,
            potential_maps=potential_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.scatter_fig),
        )
        make_average_potential_by_depth_figure(
            potential_profiles=potential_profiles_3d,
            depths_m=depths_m,
            cycles=cycles,
            out_path=Path(snakemake.output.avg_potential_depth_fig),
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
            colorbar_label="Potential (GW)",
            cmap=MAP_CMAP,
            value_scale=1000.0,
        )
        make_three_panel_map_figure(
            maps=depth_maps_2d,
            cycles=cycles,
            out_path=Path(snakemake.output.depth_overview_fig),
            colorbar_label="Depth at minimum LCOE (km)",
            cmap=f"{MAP_CMAP}_r",
            value_scale=1000.0,
        )


if __name__ == "__main__":
    main()
