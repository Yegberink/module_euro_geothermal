# Modelblocks - Geothermal module

This module calculates geothermal electricity production and levelized cost of
electricity (LCOE) from a European subsurface-temperature model based on
Limberger et al. (2014).

![Geothermal LCOE and least-cost conversion technology across Europe](./figures/lcoe_overview.png)

*LCOE for ORC, Flash, and Kalina cycles, with the least-cost technology
in each feasible grid cell. Grey areas are not feasible within the configured
maximum depth.*

## About

This is a modular `snakemake` workflow created as part of the
[Modelblocks project](https://www.modelblocks.org/). It can be imported directly
into any `snakemake` workflow.

For more information, consult the Modelblocks
[documentation](https://modelblocks.readthedocs.io/en/latest/), the
[integration example](./tests/integration/Snakefile), and the `snakemake`
[module documentation](https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html).

## How the module works

The module works through the following stages:

1. **Prepare inputs.** Download the temperature voxel model and GST-1 sediment
   thickness data, align the sediment data to the temperature grid, and crop the
   model to the supplied study-area polygons.
2. **Calculate geothermal performance.** Use the configured parameters for each
   selected power cycle to calculate depth-resolved energy potential and LCOE cubes. The model
   distinguishes conventional hydrothermal resources from Enhanced Geothermal
   Systems (EGS).
3. **Reduce the 3-D results to maps.** Either select the least-cost depth in each
   grid cell or retain separate configured depth bands. This stage produces the
   LCOE, potential, optimal-depth, and technology-comparison figures and rasters.
4. **Aggregate to regions.** Combine the gridded LCOE and potential results with
   the user-provided polygons using quantiles, fixed LCOE bins, or a single LCOE
   cutoff.
5. **Summarize potential.** Report study-wide potential by technology and for
   the portfolio selecting the lowest-LCOE technology in each cell.

## Rule graph

The graph below includes regional results, model diagnostics, and the
study-wide potential summary for one configured run.

![Snakemake rule graph for the geothermal workflow](./figures/rulegraph.svg)

## Configuration

See the [configuration guide](./config/README.md) for all workflow controls,
model parameters and aggregation choices. A complete
worked configuration is available in [`config/config.yaml`](./config/config.yaml).

## Input and output structure

The required polygon input and the module's public result path variables are
documented in [`INTERFACE.yaml`](./INTERFACE.yaml). The input must be a
GeoParquet file with a declared coordinate reference system, unique `shape_id`
values, and valid Polygon or MultiPolygon geometry. An optional `shape_class`
column may be included; rows whose value is `maritime` are excluded from
regional aggregation.

The regional `geothermal_regions.parquet` output contains one row for each
region, technology, depth category, and price bin. Its `potential` values are
net electric capacity in MW and `mean_lcoe` is the potential-weighted mean LCOE
in €2020/MWh. See the [configuration guide](./config/README.md#inputs-and-outputs)
for the complete column contract and the study-wide TSV outputs.

## Using this module

Import the module into another Snakemake workflow and map its path variables to
the parent workflow:

```python
module module_geothermal:
    snakefile: github("modelblocks-org/module_geothermal", path="workflow/Snakefile", branch="main")
    config: config["module_geothermal"]
    pathvars:
        logs="modules/geothermal/logs",
        resources="modules/geothermal/resources",
        results="modules/geothermal/results",
        shapes="modules/geo_boundaries/{shapes}/results/shapes.parquet"

use rule * from module_geothermal as module_geothermal_*
```

Request `<results>/{shapes}/geothermal_regions.parquet` for regional results
and `<results>/{shapes}/overall_potential_summary.tsv` for study-wide totals.
Each invocation uses one parameter set from config for all selected cycles.
To run different parameter sets, import the module separately with distinct
resource and result directories.

## Development

We use [`pixi`](https://pixi.sh/) as our package manager for development. Once
installed, run the following to clone this repository and install all
dependencies:

```shell
git clone git@github.com:modelblocks-org/module_geothermal.git
cd module_geothermal
pixi install --all
```

Pixi separates the development environment from the minimal `module`
environment used by Snakemake rules. After changing runtime dependencies,
regenerate the Conda environment and platform pins with:

```shell
pixi run export-snakemake-env module
```

Run the quick checks used during development with:

```shell
pixi run test-quick
```

Run isolated workflow checks (without downloads) and small numerical checks with:

```shell
pixi run pytest tests/test_workflow_config.py
pixi run -e module python -m unittest discover -s tests -p test_model_outputs.py
```

The integration definition is retained for dedicated workflow testing, but is
not part of the quick check. To run it explicitly:

```shell
pixi shell
cd tests/integration/
snakemake --use-conda --cores 2
```

## References

- [3D subsurface temperature model of Europe](https://doi.org/10.5281/zenodo.22149195), Zenodo dataset, version 1, CC BY 4.0.
- Limberger et al. (2014), [Assessing the prospective resource base for enhanced geothermal systems in Europe](https://doi.org/10.5194/gtes-2-55-2014), *Geothermal Energy Science* 2, 55–71.
- [Global Sediment Thickness Dataset v1 (GST-1)](https://doi.org/10.5281/zenodo.15320474), Zenodo dataset, CC BY 4.0.
