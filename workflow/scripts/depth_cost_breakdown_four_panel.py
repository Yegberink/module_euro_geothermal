"""Prepare optimal-depth and cost-share data for the four-panel figure."""

import sys
from pathlib import Path

import _plots
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  Register the xarray ``rio`` accessor.
import xarray as xr
from rasterio.features import geometry_mask

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


COMPONENTS = ("drilling", "stimulation", "pump", "power_plant", "fixed_om")


def _ensure_raster_metadata(da, crs=None):
    da = da.transpose("y", "x").rio.set_spatial_dims(
        x_dim="x", y_dim="y", inplace=False
    )
    crs = crs or da.attrs.get("crs")
    if crs is not None:
        da = da.rio.write_crs(crs, inplace=False)
    return da


def _depth_at_minimum_lcoe(cube_path):
    with xr.open_dataset(cube_path) as dataset:
        lcoe = dataset["lcoe"]
        argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim="depth_m")
        valid = np.isfinite(lcoe.min(dim="depth_m", skipna=True))
        depth = dataset["depth_m"].isel(depth_m=argmin).where(valid).load()
        crs = dataset.attrs.get("crs") or lcoe.attrs.get("crs")
    depth.attrs.update({"crs": crs, "units": "m"})
    return _ensure_raster_metadata(depth, crs=crs)


def _geographic_mask(shapes_path, reference, crs):
    shapes = gpd.read_parquet(shapes_path)
    if "shape_class" in shapes.columns:
        shapes = shapes.loc[shapes["shape_class"] == "land"].copy()
    shapes = shapes.loc[shapes.geometry.notna() & ~shapes.geometry.is_empty].copy()
    raster = _ensure_raster_metadata(reference, crs=crs)
    if raster.rio.crs is not None and shapes.crs != raster.rio.crs:
        shapes = shapes.to_crs(raster.rio.crs)
    return geometry_mask(
        shapes.geometry,
        out_shape=(raster.sizes["y"], raster.sizes["x"]),
        transform=raster.rio.transform(),
        invert=True,
    )


def _component_shares_for_cube(cube_path, shapes_path):
    """Return cell-level LCOE component shares at each cell's optimal depth."""
    with xr.open_dataset(cube_path) as dataset:
        required = ["lcoe", *(f"lcoe_{name}" for name in COMPONENTS)]
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(
                f"{cube_path} lacks cost variables {missing}; regenerate the LCOE cube."
            )
        lcoe = dataset["lcoe"]
        argmin = lcoe.where(np.isfinite(lcoe), np.inf).argmin(dim="depth_m")
        valid_depth = np.isfinite(lcoe.min(dim="depth_m", skipna=True))
        total = lcoe.isel(depth_m=argmin).where(valid_depth).load()
        inside = _geographic_mask(
            shapes_path, total, dataset.attrs.get("crs") or total.attrs.get("crs")
        )
        valid_total = inside & np.isfinite(total) & (total > 0)
        shares = {}
        for component in COMPONENTS:
            contribution = (
                dataset[f"lcoe_{component}"]
                .isel(depth_m=argmin)
                .where(valid_depth)
                .load()
            )
            shares[component] = (100 * contribution / total).where(valid_total)
    technology = Path(cube_path).name.replace("_lcoe_cube.nc", "").lower()
    return xr.Dataset(shares).expand_dims(technology=[technology])


def main():
    """Prepare the inputs and render the requested four-panel figure."""
    cubes = list(snakemake.input["cubes"])
    cubes_by_technology = {
        Path(path).name.replace("_lcoe_cube.nc", "").lower(): path for path in cubes
    }
    technologies = [t for t in _plots.TECH_LABELS if t in cubes_by_technology]
    depth_maps = {
        technology: _depth_at_minimum_lcoe(cubes_by_technology[technology])
        for technology in technologies
    }
    technology_shares = [
        _component_shares_for_cube(
            cubes_by_technology[technology], snakemake.input["shapes"]
        )
        for technology in technologies
    ]
    _plots.make_depth_cost_breakdown_four_panel(
        depth_maps=depth_maps,
        technology_shares=technology_shares,
        shapes_path=snakemake.input["shapes"],
        out_path=snakemake.output["fig"],
    )


if __name__ == "__main__":
    main()
