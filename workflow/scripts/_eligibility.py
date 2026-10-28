"""Resample supplied area-potential pixels once onto the model grid."""

import json
from pathlib import Path

import _schemas
import numpy as np
import xarray as xr
from affine import Affine


def source_grid(nc, power=(), heat=()):
    """Return the union of complete model cells, retaining exact source centres."""
    with xr.open_dataset(nc) as source:
        _schemas.validate_source_grid(source)
        x, y = source.x.values, source.y.values
        crs = source.attrs["crs"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    windows = []
    if power:
        with xr.open_dataset(power[0]) as cropped:
            ix = np.flatnonzero(np.isin(x, cropped.x.values))
            iy = np.flatnonzero(np.isin(y, cropped.y.values))
            windows.append((ix.min(), ix.max() + 1, iy.min(), iy.max() + 1))
    if heat:
        window = json.loads(Path(heat[0]).read_text())
        windows.append((*window["x"], *window["y"]))
    if not windows:
        raise ValueError("No model extent was supplied for eligibility.")
    xmin, xmax = min(w[0] for w in windows), max(w[1] for w in windows)
    ymin, ymax = min(w[2] for w in windows), max(w[3] for w in windows)
    # GeoTIFFs are north-up; consumers select coordinates in their original order.
    x, y = np.sort(x[xmin:xmax]), np.sort(y[ymin:ymax])[::-1]
    transform = Affine(abs(dx), 0, x[0] - abs(dx) / 2, 0, -abs(dy), y[0] + abs(dy) / 2)
    return {"x": x, "y": y, "crs": crs, "transform": transform}


def load_fraction(paths, x, y):
    """Load an exact source-grid subset; never interpolate or silently fill gaps."""
    if not paths:
        return xr.DataArray(
            np.ones((len(y), len(x))), dims=("y", "x"), coords={"x": x, "y": y}
        )
    import rioxarray

    with rioxarray.open_rasterio(paths[0], masked=True) as raster:
        selected = raster.squeeze("band", drop=True).sel(x=x, y=y).load()
    return selected.drop_vars("spatial_ref", errors="ignore")
