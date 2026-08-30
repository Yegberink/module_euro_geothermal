"""Call the shared plotting helper for one regional-potential overview."""

import sys

import _plots

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)


def main():
    """Render the configured regional-potential overview."""
    _plots.make_regional_potential_overview(
        geoparquet_path=snakemake.input["geoparquet"],
        shapes_path=snakemake.input["shapes"],
        out_path=snakemake.output["fig"],
        cycles=[str(c).strip().lower() for c in snakemake.params["cycles"]],
        depth_category=snakemake.params["depth_category"],
        bin_id=snakemake.params["bin_id"],
    )


if __name__ == "__main__":
    main()
