from pathlib import Path
import geopandas as gpd
import xarray as xr

CRS_TARGET = "EPSG:3857"

def load_shapes(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Shapes file is empty: {path}")
    if "geometry" not in gdf:
        raise ValueError(f"No geometry column found in: {path}")
    return gdf

def bbox_3857_from_shapes(gdf: gpd.GeoDataFrame):
    gdf_3857 = gdf.to_crs(CRS_TARGET)
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    return (minx, miny, maxx, maxy)

def crop_ds_bbox_projected(ds, x_name="x", y_name="y", bbox=None, pad_m=20_000):
    minx, miny, maxx, maxy = bbox
    minx -= pad_m; maxx += pad_m
    miny -= pad_m; maxy += pad_m

    x = ds[x_name]
    y = ds[y_name]

    # Handle ascending/descending coords
    y_slice = slice(miny, maxy) if float(y[0]) < float(y[-1]) else slice(maxy, miny)
    x_slice = slice(minx, maxx) if float(x[0]) < float(x[-1]) else slice(maxx, minx)

    return ds.sel({x_name: x_slice, y_name: y_slice})

def write_depth_slider_html(da, qc_path: Path, title="Cropped temperature voxel QC (depth slider)"):
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise RuntimeError(
            "QC HTML output was requested, but plotly is not installed. "
            "Install it (e.g. `pixi add plotly`) or remove qc_html from outputs."
        ) from e

    depths = da["depth_m"].values.astype(float)
    z0 = depths[0]
    img0 = da.sel(depth_m=z0).values

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img0))

    frames = [
        go.Frame(data=[go.Heatmap(z=da.sel(depth_m=d).values)], name=str(int(d)))
        for d in depths
    ]
    fig.frames = frames

    steps = []
    for d in depths:
        dname = str(int(d))
        steps.append({
            "label": dname,
            "method": "animate",
            "args": [[dname], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        })

    fig.update_layout(
        title=title,
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
    print(f"✅ Wrote interactive QC HTML: {qc_path.resolve()}")


def main():
    voxel_nc = Path(snakemake.input.nc)
    shapes_path = Path(snakemake.input.shapes)
    out_nc = Path(snakemake.output.clipped_temp_voxel)

    pad_m = float(snakemake.params.get("pad_m", 25_000))

    ds = xr.open_dataset(voxel_nc, chunks="auto")

    gdf = load_shapes(shapes_path)
    bbox = bbox_3857_from_shapes(gdf)

    ds_small = crop_ds_bbox_projected(ds, x_name="x", y_name="y", bbox=bbox, pad_m=pad_m)

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    ds_small.to_netcdf(out_nc)
    print(f"✅ Wrote clipped voxel: {out_nc.resolve()}")
    # QC plot (if requested by rule)
    qc_path = Path(snakemake.output.qc_html)
    if qc_path is not None:
        # Choose the variable name in your voxel (adjust if different)
        var = "temperature"
        if var not in ds_small:
            raise KeyError(f"'{var}' not found in cropped dataset variables: {list(ds_small.data_vars)}")
        write_depth_slider_html(ds_small[var], qc_path)

if __name__ == "__main__":
    main()
