"""Reduce depth-resolved geothermal cubes to declared 2-D rasters."""

import sys
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
    lcoe = ds["lcoe"]
    potential = ds["potential"]
    argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim=depth_dim)
    valid = np.isfinite(lcoe.min(dim=depth_dim, skipna=True))

    out = xr.Dataset(
        data_vars={
            "lcoe_2d": lcoe.isel({depth_dim: argmin}).where(valid).astype(np.float32),
            "potential_2d": potential.isel({depth_dim: argmin})
            .where(valid)
            .astype(np.float32),
        },
        coords={"x": ds["x"], "y": ds["y"]},
        attrs=dict(ds.attrs),
    )
    out["lcoe_2d"].attrs.update(lcoe.attrs)
    out["potential_2d"].attrs.update(potential.attrs)
    return out


def main():
    """Reduce all requested cubes and render the model figures."""
    cycles = [str(c).strip().lower() for c in snakemake.params.cycles]
    max_depth = float(snakemake.params.max_depth)

    cubes_by_cycle = {
        Path(path).name.replace("_lcoe_cube.nc", "").lower(): path
        for path in snakemake.input.cubes
    }
    lcoe_outputs = dict(zip(cycles, snakemake.output.lcoe_tifs, strict=True))
    potential_outputs = dict(zip(cycles, snakemake.output.pot_tifs, strict=True))

    lcoe_maps = {}
    potential_maps = {}

    for cycle in cycles:
        with xr.open_dataset(cubes_by_cycle[cycle]) as cube:
            crs = cube.attrs["crs"]
            reduced = aggregate_3d_to_2d_lcoe_min(cube, "depth_m").load()
            lcoe_maps[cycle] = _ensure_raster_metadata(reduced["lcoe_2d"], crs=crs)
            potential_maps[cycle] = _ensure_raster_metadata(
                reduced["potential_2d"], crs=crs
            )
            write_geotiff(reduced["lcoe_2d"], lcoe_outputs[cycle], crs=crs)
            write_geotiff(reduced["potential_2d"], potential_outputs[cycle], crs=crs)

    if hasattr(snakemake.output, "overview_fig"):
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


if __name__ == "__main__":
    main()
