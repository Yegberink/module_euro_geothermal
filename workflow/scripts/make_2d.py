from typing import Dict, List, Sequence
from attr import dataclass
import xarray as xr
import numpy as np
from pathlib import Path
import rioxarray
import os
import re

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
    )
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.colorbar(im, ax=ax, label=da.name or "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# 3D -> 2D aggregation
# --------------------------

def aggregate_3d_to_2d_lcoe_min(ds: xr.Dataset, depth_dim: str) -> xr.Dataset:
    lcoe = ds["lcoe"]
    pot = ds["potential"]

    lcoe_filled = lcoe.where(np.isfinite(lcoe), np.inf)
    argmin = lcoe_filled.argmin(dim=depth_dim)

    lcoe_min = lcoe.min(dim=depth_dim, skipna=True)
    valid = np.isfinite(lcoe_min)

    lcoe_2d = lcoe.isel({depth_dim: argmin}).where(valid)
    pot_2d = pot.isel({depth_dim: argmin}).where(valid)

    depth_vals = ds[depth_dim]
    depth_at_min = depth_vals.isel({depth_dim: argmin}).where(valid)

    out = xr.Dataset(
        data_vars=dict(
            lcoe_2d=lcoe_2d.astype(np.float32),
            potential_2d=pot_2d.astype(np.float32),
            depth_at_min=depth_at_min.astype(np.float32),
        ),
        coords=dict(x=ds["x"], y=ds["y"]),
        attrs=dict(ds.attrs),
    )
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

    for cycle in cycles:
        if cycle not in out_lcoe_by_cycle or cycle not in out_pot_by_cycle:
            raise ValueError(
                f"Outputs missing for cycle={cycle}. "
                f"Expected {cycle}_2d_lcoe.tif and {cycle}_2d_potential.tif "
                f"but got lcoe={sorted(out_lcoe_by_cycle)} pot={sorted(out_pot_by_cycle)}"
        )

    # Choose an outdir for extra products (depth_at_min.tif, PNGs)
    # This is not part of Snakemake outputs, but is useful. Keep it deterministic.
    extra_outdir = Path("results") / snakemake.wildcards["shapes"] / "2d_potential_cost"
    (extra_outdir / "figures").mkdir(parents=True, exist_ok=True)

    for cycle in cycles:
        cube_path = cubes_by_cycle.get(cycle)
        if cube_path is None:
            raise ValueError(f"Missing input cube for cycle={cycle}. Have cubes: {sorted(cubes_by_cycle)}")

        ds_3d = xr.open_dataset(cube_path)

        if method_2d_aggregation == "LCOE_minimization":
            ds_2d = aggregate_3d_to_2d_lcoe_min(ds_3d, depth_dim="depth_m")

            # Write exactly the outputs Snakemake expects
            write_geotiff(ds_2d["lcoe_2d"], Path(out_lcoe_by_cycle[cycle]), crs=crs)
            write_geotiff(ds_2d["potential_2d"], Path(out_pot_by_cycle[cycle]), crs=crs)

            # Optional extras (not DAG-critical)
            write_geotiff(ds_2d["depth_at_min"], extra_outdir / f"{cycle}_2d_depth_at_min.tif", crs=crs)
            quick_map_png(ds_2d["lcoe_2d"], (extra_outdir / "figures") / f"{cycle}_2d_lcoe.png", title=f"{cycle} – LCOE")
            quick_map_png(ds_2d["potential_2d"], (extra_outdir / "figures") / f"{cycle}_2d_potential.png", title=f"{cycle} – Potential")
            quick_map_png(ds_2d["depth_at_min"], (extra_outdir / "figures") / f"{cycle}_2d_depth_at_min.png", title=f"{cycle} – Depth@min")

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
                write_geotiff(ds_cat["lcoe_2d"], extra_outdir / f"{base}_lcoe.tif", crs=crs)
                write_geotiff(ds_cat["potential_2d"], extra_outdir / f"{base}_potential.tif", crs=crs)
                write_geotiff(ds_cat["depth_at_min"], extra_outdir / f"{base}_depth_at_min.tif", crs=crs)


if __name__ == "__main__":
    main()
