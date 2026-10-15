# Modelblocks - Geothermal module

This module calculates geothermal electricity and district-heating potential,
with levelized costs of electricity (LCOE) and heat (LCOH), from a European subsurface-temperature model based on
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

With `carrier_out: heat`, the workflow instead interpolates the shallowest depth
reaching the configured district-heating temperature, calculates heat costs once
per cell, and allocates heat jointly to nearby networks. `heat_and_power` runs
both branches and reports competing resource uses. See the
[heat configuration and allocation guide](./config/README.md#district-heating)
for formulas, assumptions, and output units.

## Rule graph

The graph below shows the power integration example, including regional
results, model diagnostics, and the study-wide potential summary.

![Snakemake rule graph for the geothermal workflow](./figures/rulegraph.png)

## Configuration

See the [configuration guide](./config/README.md) for all workflow controls,
model parameters and aggregation choices. A complete
worked configuration is available in [`config/config.yaml`](./config/config.yaml).

All monetary inputs in euros and all cost results use **constant 2020 euros
(EUR2020)**

## Input and output structure

The required polygon input and the module's public result path variables are
documented in [`INTERFACE.yaml`](./INTERFACE.yaml). The input must be a
GeoParquet file with a declared coordinate reference system, unique `shape_id`
values, and valid Polygon or MultiPolygon geometry. An optional `shape_class`
column may be included; rows whose value is `maritime` are excluded from
regional aggregation.

The regional `geothermal_regions.parquet` output distinguishes `category` and
`carrier` as well as region, technology, depth category, and price bin.
Power `potential` is net electric capacity in MW; heat `potential` is allocated
annual heat in MWh/year. Always interpret it together with `potential_unit`.
`mean_lcoe` and `mean_lcoh` are weighted costs in €2020/MWh of the respective carrier.
See the [configuration guide](./config/README.md#inputs-and-outputs)
for the complete column contract and the study-wide TSV outputs.

## Using this module

Import the module into another Snakemake workflow and map its path variables to
the parent workflow:

```python
module module_geothermal:
    snakefile: github("Yegberink/module_geothermal", path="workflow/Snakefile", branch="main")
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
For heat modes, provide `<resources>/user/{shapes}/areas.parquet` (or override
the `district_heating` path variable), set `carrier_out: heat` or
`heat_and_power`, and request the same regional target. Network results and
the allocation ledger are also produced and have public path variables.
The study-wide TSV and electrical diagnostic figures describe unconstrained
electricity resources; they are available in modes evaluating power. The
regional power overview shows capacity remaining after heat reservation.
Each invocation uses one parameter set from config for all selected cycles.
To run different parameter sets, import the module separately with distinct
resource and result directories.

## Figures and HTML report

`potential_overview.png` uses the LCOE overview layout to show each technology's
potential at its minimum-LCOE depth (GW per grid cell), with a shared color scale
and a fourth panel showing the potential of the cheapest technology in each cell. It is included
in the report alongside the LCOE overview when using `LCOE_minimization`.


Request regional results, the study-wide summary, and depth/cost diagnostics
explicitly, then repeat those targets with `--report results/<shapes>/report.html`.
Snakemake reports contain annotated outputs in the requested dependency graph;
`--report` renders existing results and does not execute missing jobs.

Regional results include the LCOE overview and cumulative-potential figures when
`method_2d_aggregation` selects `LCOE_minimization`. The regional map additionally
requires `cutoff`. The independent depth/cost figure is at
`<resources>/automatic/{shapes}/figures/depth_cost_breakdown_four_panel.png` and the
summary is available through `<overall_potential_summary>`. These electrical
outputs are available in power and combined modes.

See [`tests/local_test.py`](tests/local_test.py) for complete build and report
commands requesting real outputs. Generated HTML reports and rule graphs are
written under `results/`. Parent workflows can request the same output paths in
their target inputs and render their report under their results directory.

## Development

We use [`pixi`](https://pixi.sh/) as our package manager for development. Once
installed, run the following to clone this repository and install all
dependencies:

```shell
git clone git@github.com:Yegberink/module_geothermal.git
cd module_geothermal
pixi install --all
```

This is a multi-environment project (see [`pixi.toml`](./pixi.toml)):

- `default`: development and integration testing, including Snakemake, Conda,
  and pytest. Do not use this environment in Snakemake rules.
- `module`: runtime dependencies used by Snakemake rules. After changing these
  dependencies, export the environment YAML and platform pin files:

```shell
pixi run export-snakemake-env module
```

## Testing

Run the standard integration suite, including interface validation, environment
export checks, required-file checks, and a minimal imported workflow:

```shell
pixi run test-integration
```

To run the integration workflow directly:

```shell
pixi shell
cd tests/integration/
snakemake --use-conda --cores 2
```

For optional full workflow tests over three shape sets, including HTML reports
and PNG rule graphs, run:

```shell
pixi run pytest tests/local_test.py
```

These longer tests use the power integration configuration, disable eligibility,
and download their shape inputs. Heat networks and eligibility rasters are not
bundled test inputs. Graph rendering requires Graphviz's `dot` executable.
The CI integration task does not run these optional tests.

Run both workflow lint checks with:

```shell
pixi run snakemake --lint --snakefile workflow/Snakefile
pixi run snakemake --lint --snakefile tests/integration/Snakefile
```

## References

- [3D subsurface temperature model of Europe](https://doi.org/10.5281/zenodo.22149195), Zenodo dataset, version 1, CC BY 4.0.
- Limberger et al. (2014), [Assessing the prospective resource base for enhanced geothermal systems in Europe](https://doi.org/10.5194/gtes-2-55-2014), *Geothermal Energy Science* 2, 55–71.
- [Global Sediment Thickness Dataset v1 (GST-1)](https://doi.org/10.5281/zenodo.15320474), Zenodo dataset, CC BY 4.0.

Land-use exclusions can optionally constrain both heat and power using a supplied
area-potential raster. The workflow resamples it once onto the subsurface grid.
See the [eligibility configuration](config/README.md#land-eligibility).
