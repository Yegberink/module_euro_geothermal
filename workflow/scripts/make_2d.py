"""Reduce depth-resolved geothermal cubes to declared 2-D rasters."""

import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import _plots
import numpy as np
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def _ensure_raster_metadata(
    da: xr.DataArray, *, x_name: str = "x", y_name: str = "y", crs: str | None = None
) -> xr.DataArray:
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
    da: xr.DataArray, out_path, *, nodata: float = np.nan, crs: str | None = None
):
    """Write a georeferenced data array as a compressed GeoTIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if da.sizes.get("y", 1) == 0 or da.sizes.get("x", 1) == 0:
        da = xr.DataArray(
            np.full((1, 1), np.nan, dtype=np.float32),
            coords={"y": [0.0], "x": [0.0]},
            dims=("y", "x"),
            attrs=dict(da.attrs),
        )
    raster = _ensure_raster_metadata(da, crs=crs)
    raster = raster.rio.write_nodata(nodata, inplace=False)
    raster.rio.to_raster(str(out_path), compress="deflate")


def aggregate_3d_to_2d_lcoe_min(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
    """Select the minimum-LCOE depth independently in every grid cell."""
    temperature = ds["temperature"]
    lcoe = ds["lcoe"]
    potential = ds["potential"]
    argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim=depth_dim)
    valid = np.isfinite(lcoe.min(dim=depth_dim, skipna=True))
    depth_values = ds[depth_dim]

    out = xr.Dataset(
        data_vars={
            "temperature_at_min": temperature.isel({depth_dim: argmin})
            .where(valid)
            .astype(np.float32),
            "lcoe_2d": lcoe.isel({depth_dim: argmin}).where(valid).astype(np.float32),
            "potential_2d": potential.isel({depth_dim: argmin})
            .where(valid)
            .astype(np.float32),
            "depth_at_min": depth_values.isel({depth_dim: argmin})
            .where(valid)
            .astype(np.float32),
        },
        coords={"x": ds["x"], "y": ds["y"]},
        attrs=dict(ds.attrs),
    )
    out["temperature_at_min"].attrs.update(temperature.attrs)
    out["lcoe_2d"].attrs.update(lcoe.attrs)
    out["potential_2d"].attrs.update(potential.attrs)
    out["depth_at_min"].attrs.update(getattr(depth_values, "attrs", {}))
    return out


def aggregate_3d_to_2d_depth_categories(
    ds: xr.Dataset, depth_categories: Sequence[tuple[float, float]], depth_dim: str
) -> dict[str, xr.Dataset]:
    """Select the minimum-LCOE depth within each configured interval."""
    depth = ds[depth_dim]
    outputs = {}
    for zmin, zmax in depth_categories:
        label = f"{zmin / 1000:g}-{zmax / 1000:g}km"
        selected = (depth >= zmin) & (depth < zmax)
        if int(selected.sum()) == 0:
            shape = (ds.sizes["y"], ds.sizes["x"])
            empty = np.full(shape, np.nan, dtype=np.float32)
            outputs[label] = xr.Dataset(
                data_vars={
                    name: (("y", "x"), empty.copy())
                    for name in (
                        "temperature_at_min",
                        "lcoe_2d",
                        "potential_2d",
                        "depth_at_min",
                    )
                },
                coords={"x": ds["x"], "y": ds["y"]},
                attrs=dict(ds.attrs),
            )
            continue
        subset = ds.sel({depth_dim: depth.where(selected, drop=True)})
        outputs[label] = aggregate_3d_to_2d_lcoe_min(subset, depth_dim)
    return outputs


def _parse_tif_output(path: str) -> tuple[str, str | None, str]:
    name = os.path.basename(path)
    match = re.match(
        r"^(?P<cycle>[A-Za-z0-9]+)_2d_(?:(?P<category>.+)_)?"
        r"(?P<kind>lcoe|potential)\.tif$",
        name,
    )
    if match is None:
        raise ValueError(f"Unrecognized 2-D output filename: {path}")
    return (match.group("cycle").lower(), match.group("category"), match.group("kind"))


def main():
    """Reduce all requested cubes and render the model figures."""
    cycles = [str(c).strip().lower() for c in snakemake.params.cycles]
    method = snakemake.params.method_2d_aggregation
    depth_categories = [
        (float(pair[0]), float(pair[1]))
        for pair in (snakemake.params.depth_categories or [])
    ]
    max_depth = float(snakemake.params.max_depth)

    cubes_by_cycle = {
        Path(path).name.replace("_lcoe_cube.nc", "").lower(): path
        for path in snakemake.input.cubes
    }
    lcoe_outputs, potential_outputs = {}, {}
    for paths, outputs in (
        (snakemake.output.lcoe_tifs, lcoe_outputs),
        (snakemake.output.pot_tifs, potential_outputs),
    ):
        for path in paths:
            cycle, category, _ = _parse_tif_output(str(path))
            outputs[(cycle, category)] = str(path)

    lcoe_maps = {}
    potential_maps = {}
    temperature_maps = {}

    for cycle in cycles:
        with xr.open_dataset(cubes_by_cycle[cycle]) as cube:
            crs = cube.attrs["crs"]
            if method == "LCOE_minimization":
                reduced = aggregate_3d_to_2d_lcoe_min(cube, "depth_m").load()
                lcoe_maps[cycle] = _ensure_raster_metadata(reduced["lcoe_2d"], crs=crs)
                potential_maps[cycle] = _ensure_raster_metadata(
                    reduced["potential_2d"], crs=crs
                )
                temperature_maps[cycle] = _ensure_raster_metadata(
                    reduced["temperature_at_min"], crs=crs
                )
                write_geotiff(reduced["lcoe_2d"], lcoe_outputs[(cycle, None)], crs=crs)
                write_geotiff(
                    reduced["potential_2d"], potential_outputs[(cycle, None)], crs=crs
                )
            elif method == "depth_categories":
                reduced_by_category = aggregate_3d_to_2d_depth_categories(
                    cube, depth_categories, "depth_m"
                )
                for category, reduced in reduced_by_category.items():
                    write_geotiff(
                        reduced["lcoe_2d"], lcoe_outputs[(cycle, category)], crs=crs
                    )
                    write_geotiff(
                        reduced["potential_2d"],
                        potential_outputs[(cycle, category)],
                        crs=crs,
                    )
            else:
                raise ValueError(f"Unsupported 2-D aggregation method: {method}")

    if method == "LCOE_minimization" and hasattr(snakemake.output, "overview_fig"):
        _plots.make_resource_overview(
            maps=lcoe_maps,
            cycles=cycles,
            out_path=snakemake.output.overview_fig,
            shapes_path=snakemake.input.shapes,
            max_depth=max_depth,
        )
        _plots.make_resource_overview(
            maps=potential_maps,
            cycles=cycles,
            out_path=snakemake.output.potential_overview_fig,
            shapes_path=snakemake.input.shapes,
            max_depth=max_depth,
            potential=True,
            selection_costs=lcoe_maps,
        )
        _plots.make_lcoe_potential_cumulative_four_panel(
            lcoe_maps=lcoe_maps,
            potential_maps=potential_maps,
            temperature_maps=temperature_maps,
            cycles=cycles,
            out_path=snakemake.output.technology_cumulative_four_panel_fig,
        )


if __name__ == "__main__":
    main()
