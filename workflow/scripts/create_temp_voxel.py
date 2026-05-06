#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
import xarray as xr
import _schemas

DEFAULT_NODATA = -9999.0
CRS_STRING = "EPSG:3857"

RASTER_PATTERN = r"temperature(\d{4,5})\.asc"  # temperature0000.asc ... temperature10000.asc

def read_raster(path: Path):
    ds = rasterio.open(path)
    arr = ds.read(1)
    return ds, arr

def align_to_reference(arr, src_ds, ref_ds, ref_crs, resampling=Resampling.bilinear):
    dst_arr = np.empty((ref_ds.height, ref_ds.width), dtype=np.float32)
    src_crs = src_ds.crs if src_ds.crs is not None else ref_crs
    reproject(
        source=arr,
        destination=dst_arr,
        src_transform=src_ds.transform,
        src_crs=src_crs,
        dst_transform=ref_ds.transform,
        dst_crs=ref_crs,
        dst_width=ref_ds.width,
        dst_height=ref_ds.height,
        resampling=resampling,
    )
    return dst_arr

def main():
    # --- Get module I/O from Snakemake ---
    out_nc = Path(snakemake.output.nc)

    z_min = float(snakemake.params.get("z_min", 0))
    z_max = float(snakemake.params["z_max"])
    z_step = float(snakemake.params.get("z_step", 250))
    vert_interp_kind = snakemake.params.get("vert_interp_kind", "linear")
    target_depths = np.arange(z_min, z_max + z_step, z_step, dtype=np.float32)

    depth_files = {}
    for f in snakemake.input.asc:
        p = Path(f)
        m = re.match(RASTER_PATTERN, p.name, flags=re.IGNORECASE)
        if m:
            depth_files[int(m.group(1))] = p
    depth_files = dict(sorted(depth_files.items()))
    if not depth_files:
        da_interp = xr.DataArray(
            np.full((len(target_depths), 1, 1), np.nan, dtype=np.float32),
            coords={"depth_m": target_depths, "y": [0.0], "x": [0.0]},
            dims=("depth_m", "y", "x"),
            name="temperature",
            attrs={"units": "degree_Celsius"},
        )
        ds_out = da_interp.to_dataset()
        ds_out.attrs.update({
            "description": "3D temperature voxel built from depth rasters",
            "vertical_sampling_m": z_step,
            "source_files": "",
            "crs": CRS_STRING,
        })
        ds_out["temperature"].encoding["_FillValue"] = np.nan
        ds_out.to_netcdf(out_nc)
        _schemas.write_empty_html(snakemake.output.qc_html)
        return

    ref_depth = min(depth_files.keys())
    ref_path = depth_files[ref_depth]
    ref_ds, ref_arr = read_raster(ref_path)

    if ref_ds.crs is None:
        ref_crs = CRS.from_user_input(CRS_STRING)
    else:
        ref_crs = ref_ds.crs

    depths = []
    stack = []

    for depth_m, path in depth_files.items():
        src_ds, src_arr = read_raster(path)
        nodata = src_ds.nodata if src_ds.nodata is not None else DEFAULT_NODATA

        same_grid = (
            ((src_ds.crs == ref_crs) or (src_ds.crs is None)) and
            (src_ds.transform == ref_ds.transform) and
            (src_ds.width == ref_ds.width) and
            (src_ds.height == ref_ds.height)
        )
        if same_grid:
            arr_aligned = src_arr.astype(np.float32)
        else:
            arr_aligned = align_to_reference(src_arr, src_ds, ref_ds, ref_crs)

        arr_aligned = np.where(arr_aligned == nodata, np.nan, arr_aligned).astype(np.float32, copy=False)
        depths.append(depth_m)
        stack.append(arr_aligned)

        src_ds.close()

    depths = np.array(depths, dtype=np.int32)
    stack = np.stack(stack, axis=0)

    transform = ref_ds.transform
    xs = transform.c + (np.arange(ref_ds.width) + 0.5) * transform.a
    ys = transform.f + (np.arange(ref_ds.height) + 0.5) * transform.e  # e is negative

    if ys[0] > ys[-1]:
        ys = ys[::-1]
        stack = stack[:, ::-1, :]

    da = xr.DataArray(
        stack,
        coords={"depth_m": depths, "y": ys, "x": xs},
        dims=("depth_m", "y", "x"),
        name="temperature",
        attrs={"units": "degree_Celsius"},
    )

    da_interp = da.interp(depth_m=target_depths, method=vert_interp_kind)

    ds_out = da_interp.to_dataset()
    ds_out.attrs.update({
        "description": "3D temperature voxel built from depth rasters",
        "vertical_sampling_m": z_step,
        "source_files": ", ".join([Path(f).name for f in snakemake.input.asc]),
        "crs": CRS_STRING,
    })
    ds_out["temperature"].encoding["_FillValue"] = np.nan

    ds_out.to_netcdf(out_nc)

    ref_ds.close()
    # --- Optional interactive QC plot ---
    qc_path = Path(snakemake.output.qc_html)

    if qc_path is not None:
        import plotly.graph_objects as go

        depths = da_interp["depth_m"].values.astype(float)
        if depths.size == 0:
            _schemas.write_empty_html(qc_path)
            return
        # Use first slice as initial frame
        z0 = depths[0]
        img0 = da_interp.sel(depth_m=z0).values

        fig = go.Figure()

        fig.add_trace(go.Heatmap(z=img0))

        frames = []
        for d in depths:
            frames.append(go.Frame(data=[go.Heatmap(z=da_interp.sel(depth_m=d).values)],
                                name=str(int(d))))

        fig.frames = frames

        # Slider steps
        steps = []
        for d in depths:
            dname = str(int(d))
            steps.append({
                "label": dname,
                "method": "animate",
                "args": [[dname], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0}}],
            })

        fig.update_layout(
            title="Temperature voxel QC (depth slider)",
            xaxis_title="x",
            yaxis_title="y",
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": "Depth (m): "},
                "pad": {"t": 30},
                "steps": steps,
            }],
        )

        fig.write_html(str(qc_path), include_plotlyjs="cdn")

if __name__ == "__main__":
    main()
