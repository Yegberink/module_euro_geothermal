"""Reduce depth-resolved geothermal cubes to declared 2-D rasters."""

import sys

import _plots
import numpy as np
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def _ensure_raster_metadata(
    da: xr.DataArray, *, x_name: str = "x", y_name: str = "y", crs: str | None = None
) -> xr.DataArray:
    if da.sizes.get("y", 1) == 0 or da.sizes.get("x", 1) == 0:
        da = xr.DataArray(
            np.full((1, 1), np.nan, dtype=np.float32),
            coords={"y": [0.0], "x": [0.0]},
            dims=("y", "x"),
            attrs=dict(da.attrs),
        )
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
    cycles = snakemake.params.cycles
    max_depth = float(snakemake.params.max_depth)

    lcoe_maps = {}
    potential_maps = {}

    for cycle, cube_path, lcoe_path, potential_path in zip(
        cycles,
        snakemake.input.cubes,
        snakemake.output.lcoe_tifs,
        snakemake.output.pot_tifs,
        strict=True,
    ):
        with xr.open_dataset(cube_path) as cube:
            crs = cube.attrs["crs"]
            reduced = aggregate_3d_to_2d_lcoe_min(cube, "depth_m").load()
            lcoe_maps[cycle] = _ensure_raster_metadata(reduced["lcoe_2d"], crs=crs)
            potential_maps[cycle] = _ensure_raster_metadata(
                reduced["potential_2d"], crs=crs
            )
            for raster, path in (
                (lcoe_maps[cycle], lcoe_path),
                (potential_maps[cycle], potential_path),
            ):
                raster.rio.write_nodata(np.nan).rio.to_raster(path, compress="deflate")

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
