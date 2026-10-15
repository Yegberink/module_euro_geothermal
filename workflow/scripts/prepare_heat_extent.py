"""Resolve heat coverage to source-grid indices, preserving unchanged extents."""

import json
import sys
from pathlib import Path

import _schemas
import geopandas as gpd
import numpy as np
import xarray as xr
from _heat import METRIC_CRS
from shapely.geometry import box


def main():
    """Write only when the required grid window changes, not for demand edits."""
    networks = (
        _schemas.HeatNetworksSchema.validate(gpd.read_parquet(snakemake.input.networks))
        .sort_values("area_id", kind="stable")
        .reset_index(drop=True)
        .to_crs(METRIC_CRS)
    )
    shapes = _schemas.ShapeSchema.validate(gpd.read_parquet(snakemake.input.shapes))
    with xr.open_dataset(snakemake.input.nc) as ds:
        _schemas.SourceGridSchema.validate(ds)
        crs = ds.attrs["crs"]
        bounds = [shapes.to_crs(crs).total_bounds]
        if len(networks):
            # Only the extent is needed: avoid constructing thousands of
            # expensive buffers of detailed network polygons. Densify this
            # enclosing rectangle before reprojection to follow curved edges.
            minx, miny, maxx, maxy = networks.total_bounds
            pad = snakemake.params.buffer_km * 1000
            envelope = gpd.GeoSeries(
                [box(minx - pad, miny - pad, maxx + pad, maxy + pad)], crs=METRIC_CRS
            ).segmentize(10000)
            bounds.append(envelope.to_crs(crs).total_bounds)
        bounds = np.asarray(bounds)
        finite = np.isfinite(bounds).all(axis=1)
        if not finite.any():
            # Empty study areas still produce a georeferenced, empty model tile.
            window = {"x": [0, 1], "y": [0, 1], "empty": True}
        else:
            lo = bounds[finite, :2].min(axis=0)
            hi = bounds[finite, 2:].max(axis=0)
            window = {"empty": False}
            for axis, low, high in zip(["x", "y"], lo, hi):
                values = ds[axis].values
                # One-cell margin includes boundary cells and retains a usable transform.
                pad = abs(float(values[1] - values[0]))
                indices = np.flatnonzero((values >= low - pad) & (values <= high + pad))
                window[axis] = (
                    [int(indices.min()), int(indices.max()) + 1]
                    if len(indices)
                    else [0, 2]
                )
                window["empty"] |= not len(indices)
    path = Path(snakemake.output.extent)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(window, sort_keys=True) + "\n"
    # Demand or small buffer changes must not force a new cost pass when the
    # source-grid window is unchanged. Snakemake also checks small-file checksums.
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


if __name__ == "__main__":
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
