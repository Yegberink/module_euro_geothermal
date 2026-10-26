"""All figure rendering used by the geothermal workflow.

Computation scripts prepare arrays and tables, then call the functions in this
module. Keeping the rendering here makes the set of workflow figures explicit
and prevents plotting dependencies from leaking into calculation code.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr
from matplotlib.colors import ListedColormap, Normalize, PowerNorm
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TECH_LABELS = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}
TECH_COLORS = {"orc": "#0072B2", "flash": "#E69F00", "kalina": "#009E73"}
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
LCOE_CMAP = "YlOrBr_r"
LCOE_COLORBAR_CUTOFF = 600.0
LCOE_NORM_GAMMA = 0.4
MAP_BACKGROUND_COLOR = "#DCEEF4"
UNAVAILABLE_COLOR = "#D9D9D9"
LCOE_UNAVAILABLE_COLOR = "#BDBDBD"
COUNTRY_EDGE_COLOR = "#4D5960"

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


def write_blank_figure(path, figsize=(6, 4)) -> None:
    """Write a valid empty placeholder when no plottable data are available."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _country_layer(shapes_path, reference_da: xr.DataArray) -> gpd.GeoDataFrame:
    countries = gpd.read_parquet(shapes_path)
    if "shape_class" in countries.columns:
        countries = countries.loc[countries["shape_class"] == "land"].copy()
    countries = countries.loc[
        countries.geometry.notna() & ~countries.geometry.is_empty
    ].copy()
    try:
        target_crs = reference_da.rio.crs
    except (AttributeError, ValueError):
        target_crs = None
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


def make_resource_overview(
    maps,
    cycles,
    out_path,
    shapes_path,
    max_depth,
    *,
    potential=False,
    selection_costs=None,
    heat=False,
    regions=None,
    reference_grid=None,
):
    """Render cost and potential overviews with the same map layout."""
    if not cycles:
        write_blank_figure(out_path)
        return

    reference = maps[cycles[0]] if reference_grid is None else reference_grid
    countries = (
        regions if regions is not None else _country_layer(shapes_path, reference)
    )
    if regions is not None and reference_grid is not None:
        countries = countries.to_crs(reference_grid.rio.crs)
    if countries.empty:
        write_blank_figure(out_path)
        return

    panels = cycles if heat else list(TECH_LABELS) + ["winner"]
    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols))
    if regions is None or reference_grid is not None:
        left, bottom, right, top = reference.rio.bounds()
    else:
        left, bottom, right, top = regions.total_bounds
    extent = [left, right, bottom, top]
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

        axes = []
        for i in range(len(panels)):
            row, col = divmod(i, ncols)
            x0 = (margin_l + col * (panel_w + gap)) / fig_w
            y0 = (margin_b + (nrows - 1 - row) * (panel_h + gap)) / fig_h
            axes.append(fig.add_axes([x0, y0, panel_w / fig_w, panel_h / fig_h]))
        lcoe_cmap = plt.get_cmap(LCOE_CMAP).copy()
        lcoe_cmap.set_bad((0, 0, 0, 0))
        winner_cmap = ListedColormap(
            [TECH_COLORS[c] for c in cycles if c in TECH_COLORS]
        )
        winner_cmap.set_bad((0, 0, 0, 0))

        scale = (1e6 if heat else 1000.0) if potential else 1.0
        plot_arrays = {
            cycle: (
                np.asarray(maps[cycle], dtype=float) / scale
                if regions is not None
                else _plot_values(maps[cycle], countries, value_scale=scale)
            )
            for cycle in maps
        }
        if regions is not None:
            # Zero supply is distinct from positive potential on the color scale.
            plot_arrays = {
                panel: np.where(values > 0, values, np.nan)
                for panel, values in plot_arrays.items()
            }
        # Keep absent technologies visible, including when all data are unavailable.
        for panel in panels:
            if panel != "winner" and panel not in plot_arrays:
                plot_arrays[panel] = np.full_like(
                    next(iter(plot_arrays.values())), np.nan
                )
        data_vmin, data_vmax = _finite_limits(plot_arrays.values())
        if not np.isfinite(data_vmax):
            data_vmin = data_vmax = 0.0
        lcoe_vmax = min(data_vmax, LCOE_COLORBAR_CUTOFF)
        lcoe_norm = (
            Normalize(vmin=data_vmin, vmax=data_vmax)
            if heat
            else PowerNorm(gamma=LCOE_NORM_GAMMA, vmin=70, vmax=LCOE_COLORBAR_CUTOFF)
        )
        colorbar_extend = "max" if not heat and data_vmax > lcoe_vmax else "neither"
        if potential:
            lcoe_norm = (
                Normalize(vmin=0, vmax=max(data_vmax, 1e-6))
                if heat or regions is not None
                else PowerNorm(gamma=LCOE_NORM_GAMMA, vmin=0, vmax=max(data_vmax, 1e-6))
            )
            colorbar_extend = "neither"
        if potential and not heat and regions is None:
            costs = {
                cycle: _plot_values(selection_costs[cycle], countries)
                for cycle in cycles
            }
            # Ties follow the configured cycle order.
            winner = _winner_map(costs, cycles)
            stack = np.stack([plot_arrays[cycle] for cycle in cycles], axis=0)
            selected = np.take_along_axis(
                stack, np.maximum(winner, 0)[None, ...], axis=0
            )[0]
            plot_arrays["winner"] = np.where(winner >= 0, selected, np.nan)
        elif not potential and not heat:
            winner = _winner_map({c: plot_arrays[c] for c in cycles}, cycles)

        for idx, panel in enumerate(panels):
            ax = axes[idx]
            _decorate_map(
                ax, countries, extent, unavailable_color=LCOE_UNAVAILABLE_COLOR
            )
            if panel == "winner" and not potential:
                ax.imshow(
                    np.where(winner >= 0, winner, np.nan),
                    origin="lower"
                    if reference.y.values[-1] >= reference.y.values[0]
                    else "upper",
                    extent=extent,
                    interpolation="nearest",
                    cmap=winner_cmap,
                    vmin=-0.5,
                    vmax=len(cycles) - 0.5,
                    aspect="auto",
                    zorder=2,
                )
                title = "Cheapest technology"
            elif regions is not None:
                countries.assign(_value=plot_arrays[panel]).plot(
                    column="_value",
                    ax=ax,
                    cmap=lcoe_cmap,
                    norm=lcoe_norm,
                    edgecolor="none",
                    zorder=2,
                    missing_kwds={"color": LCOE_UNAVAILABLE_COLOR},
                )
                title = (
                    "Cheapest technology potential"
                    if panel == "winner"
                    else TECH_LABELS.get(panel, "Heat")
                )
            else:
                ax.imshow(
                    plot_arrays[panel],
                    origin="lower"
                    if reference.y.values[-1] >= reference.y.values[0]
                    else "upper",
                    extent=extent,
                    interpolation="nearest",
                    cmap=lcoe_cmap,
                    norm=lcoe_norm,
                    aspect="auto",
                    zorder=2,
                )
                title = (
                    "Cheapest technology potential"
                    if panel == "winner"
                    else ("Heat" if heat else TECH_LABELS.get(panel, panel.upper()))
                )
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
            # GeoPandas otherwise adjusts the axes box to geographic aspect,
            # changing the panel spacing relative to raster figures.
            ax.set_aspect("auto")
            _clean_map_axes(ax)

        cbar_x0 = (margin_l + 0.1 * fig_w) / fig_w
        cbar_y0 = 0.38 / fig_h
        cbar_w = 1 - cbar_x0 - (margin_r + 0.1 * fig_w) / fig_w
        cax = fig.add_axes([cbar_x0, cbar_y0, cbar_w, 0.04 / fig_h * fig_w])
        sm = plt.cm.ScalarMappable(norm=lcoe_norm, cmap=lcoe_cmap)
        cbar = fig.colorbar(
            sm, cax=cax, orientation="horizontal", extend=colorbar_extend
        )
        if heat:
            label = (
                (
                    "Heat supplied (TWh$_{th}$/year per "
                    + ("region)" if regions is not None else "grid cell)")
                )
                if potential
                else "Levelized Cost of Heat (€2020/MWh$_{th}$)"
            )
        else:
            label = (
                (
                    "Potential (GW$_e$ per region)"
                    if regions is not None
                    else "Potential at minimum-LCOE depth (GW$_e$ per grid cell)"
                )
                if potential
                else "Levelized Cost of Electricity (€2020/MWh$_e$)"
            )
        cbar.set_label(label)
        if potential:
            cbar.set_ticks(np.linspace(0, max(data_vmax, 1e-6), 5))
        elif heat:
            cbar.set_ticks(np.linspace(lcoe_norm.vmin, lcoe_norm.vmax, 5))
        cbar.ax.tick_params(length=0, labelsize=11)

        handles = [
            Patch(
                facecolor=TECH_COLORS[c],
                edgecolor="none",
                label=TECH_LABELS.get(c, c.upper()),
            )
            for c in cycles
            if c in TECH_COLORS
        ]
        if potential:
            handles = []
        depth_label = (
            f"{max_depth / 1000:g} km" if max_depth >= 1000 else f"{max_depth:g} m"
        )
        handles.append(
            Patch(
                facecolor=LCOE_UNAVAILABLE_COLOR,
                edgecolor=COUNTRY_EDGE_COLOR,
                label="No supplied heat"
                if heat and regions is not None
                else "No potential"
                if regions is not None or (heat and potential)
                else f"Not possible below {depth_label}",
            )
        )
        axes[-1].legend(
            handles=handles,
            loc="lower left" if potential else "upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="#AAAAAA",
            fontsize=10,
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
