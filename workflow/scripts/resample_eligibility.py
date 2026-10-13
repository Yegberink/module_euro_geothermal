"""Convert supplied available areas to fractions and resample once."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
from _eligibility import source_grid
from _schemas import EligibilityFractionSchema
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


def pixel_areas(transform, rows):
    """WGS84 pixel areas in m², matching the area-potential module's ellipsoid."""
    a, b = 6378137, 6356752.3142
    eccentricity = np.sqrt(1 - (b / a) ** 2)

    def zone(latitude):
        sine = np.sin(np.radians(latitude))
        minus, plus = 1 - eccentricity * sine, 1 + eccentricity * sine
        return (
            np.pi
            * b**2
            * (np.log(plus / minus) / (2 * eccentricity) + sine / (plus * minus))
        )

    north = transform.f + np.asarray(rows) * transform.e
    south = north + transform.e
    return transform.a / 360 * (zone(north) - zone(south))


def resample(area_path, grid, output_path):
    """Normalize in bounded strips, then average directly onto model cells."""
    with rasterio.open(area_path) as source, TemporaryDirectory() as temporary:
        transform = source.transform
        if (
            source.count != 1
            or source.crs != rasterio.crs.CRS.from_epsg(4326)
            or transform.b != 0
            or transform.d != 0
            or transform.a <= 0
            or transform.e >= 0
            or not np.isclose(transform.a, -transform.e)
        ):
            raise ValueError(
                "Area potential must be a single-band, square, north-up WGS84 raster "
                "containing available square metres per pixel."
            )
        bounds = transform_bounds(
            grid["crs"],
            source.crs,
            *array_bounds(len(grid["y"]), len(grid["x"]), grid["transform"]),
            densify_pts=101,
        )
        extent = from_bounds(*bounds, transform=transform)
        col, row = np.floor([extent.col_off, extent.row_off]).astype(int) - 1
        end_col = int(np.ceil(extent.col_off + extent.width)) + 1
        end_row = int(np.ceil(extent.row_off + extent.height)) + 1
        width, total_rows = end_col - col, end_row - row
        normalized_path = Path(temporary) / "fractions.tif"
        profile = dict(
            driver="GTiff",
            width=width,
            height=total_rows,
            count=1,
            dtype="float32",
            crs=source.crs,
            transform=source.window_transform(Window(col, row, width, total_rows)),
            tiled=True,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(normalized_path, "w", **profile) as normalized:
            for start in range(0, total_rows, 256):
                height = min(256, total_rows - start)
                window = Window(col, row + start, width, height)
                areas = source.read(
                    1, window=window, masked=True, boundless=True
                ).filled(0)
                fractions = (
                    areas
                    / pixel_areas(
                        transform, np.arange(row + start, row + start + height)
                    )[:, None]
                )
                # Permit float32 roundoff in upstream pixel-area calculations.
                if (
                    not np.isfinite(fractions).all()
                    or ((fractions < 0) | (fractions > 1 + 1e-5)).any()
                ):
                    raise ValueError(
                        "Available area must be finite and between zero and pixel area."
                    )
                normalized.write(
                    np.clip(fractions, 0, 1).astype("float32"),
                    1,
                    window=Window(0, start, width, height),
                )

        fraction = np.zeros((len(grid["y"]), len(grid["x"])), dtype="float32")
        with rasterio.open(normalized_path) as normalized:
            reproject(
                source=rasterio.band(normalized, 1),
                destination=fraction,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=0,
                resampling=Resampling.average,
            )

    EligibilityFractionSchema.validate(fraction)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        width=len(grid["x"]),
        height=len(grid["y"]),
        count=1,
        dtype="float32",
        crs=grid["crs"],
        transform=grid["transform"],
        nodata=np.nan,
        compress="deflate",
    ) as output:
        output.write(fraction, 1)
        output.update_tags(
            units="1",
            description="Available area fraction [0,1]",
            source=str(area_path),
            resampling="average",
            missing_source_area="unavailable",
        )
    logging.info("Resampled %s onto %s model cells", area_path, fraction.size)
    return fraction


if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log[0], level=logging.INFO, force=True)
    grid = source_grid(snakemake.input.nc, snakemake.input.power, snakemake.input.heat)
    resample(snakemake.input.area_potential, grid, snakemake.output.fraction)
