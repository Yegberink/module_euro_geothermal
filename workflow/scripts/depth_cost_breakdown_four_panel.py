"""Combine three optimal-depth maps with the LCOE cost breakdown."""

import sys
from pathlib import Path

import cost_breakdown
import make_2d
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def _cycle_from_cube(path):
    return Path(path).name.replace("_lcoe_cube.nc", "").lower()


def _depth_at_minimum_lcoe(cube_path):
    """Return the optimal-depth map in metres for one technology cube."""
    with xr.open_dataset(cube_path) as ds:
        lcoe = ds["lcoe"]
        argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim="depth_m")
        valid = np.isfinite(lcoe.min(dim="depth_m", skipna=True))
        depth = ds["depth_m"].isel(depth_m=argmin).where(valid).load()
        crs = ds.attrs.get("crs") or lcoe.attrs.get("crs")
    depth.attrs.update({"crs": crs, "units": "m"})
    return make_2d._ensure_raster_metadata(depth, crs=crs)


def make_four_panel_figure(cubes, shapes_path, output_path):
    """Render the three technology depth maps and cost boxplots."""
    cubes_by_technology = {_cycle_from_cube(path): path for path in cubes}
    technologies = [
        technology
        for technology in make_2d.TECH_LABELS
        if technology in cubes_by_technology
    ]
    if not technologies:
        make_2d._schemas.write_blank_png(output_path, figsize=(8.27, 8.27))
        return

    depth_maps = {
        technology: _depth_at_minimum_lcoe(cubes_by_technology[technology])
        for technology in technologies
    }
    technology_shares = [
        cost_breakdown.component_shares_for_cube(
            cubes_by_technology[technology], shapes_path
        )
        for technology in technologies
    ]

    reference = depth_maps[technologies[0]]
    countries = make_2d._country_layer(Path(shapes_path), reference)
    extent = make_2d._map_extent(reference)
    arrays = [
        make_2d._plot_values(depth_maps[technology], countries, value_scale=1000.0)
        for technology in technologies
    ]
    vmin, vmax = make_2d._finite_limits(arrays)
    cmap = plt.get_cmap(make_2d.LCOE_OVERVIEW_CMAP).copy()
    cmap.set_bad((0, 0, 0, 0))

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.27))
    map_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for index, (ax, technology, values) in enumerate(
        zip(map_axes, technologies, arrays)
    ):
        make_2d._decorate_map(ax, countries, extent, make_2d._map_crs(reference))
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
            f"{make_2d.PANEL_LETTERS[index]}) "
            f"{make_2d.TECH_LABELS.get(technology, technology.upper())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="black",
        )
        make_2d._clean_map_axes(ax)
        ax.set_box_aspect(1)

    cost_ax = axes[1, 1]
    cost_breakdown.plot_cost_breakdown(cost_ax, technology_shares, compact=True)
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
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.0),
        zorder=6,
    )

    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.98, bottom=0.12, wspace=0.16, hspace=0.16
    )
    cax = fig.add_axes([0.09, 0.045, 0.38, 0.022])
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=image.norm, cmap=image.cmap),
        cax=cax,
        orientation="horizontal",
    )
    colorbar.set_label("Depth at minimum LCOE (km)", fontsize=12)
    colorbar.ax.tick_params(length=0, labelsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """Write the combined depth-map and cost-breakdown figure."""
    make_four_panel_figure(
        list(snakemake.input["cubes"]),
        snakemake.input["shapes"],
        snakemake.output["fig"],
    )


if __name__ == "__main__":
    main()
