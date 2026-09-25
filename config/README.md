# Configuration

This module calculates depth-resolved geothermal potential and LCOE, reduces
the results to two-dimensional rasters, and aggregates them to user-provided
regions. The complete annotated example is
[`config/config.yaml`](./config.yaml); configuration values are validated by
[`workflow/internal/config.schema.yaml`](../workflow/internal/config.schema.yaml).

When the workflow is imported as a module, place these settings under the key
passed to the module's `config` directive, for example:

```yaml
module_geothermal:
  which_cycles: [orc]
  carrier_out: power
  # Add the remaining model parameters from config/config.yaml.
```

## Workflow controls

Each run uses the configured model parameters for every selected conversion
cycle and shape set. Stable defaults fill omitted optional settings; supplied
values take precedence. Model settings are tracked by Snakemake so changes
trigger recalculation.

## Conversion technologies

`which_cycles` selects one or more of `orc`, `flash`, and `kalina`.
`carrier_out` must be `power`; this initial public release models geothermal
electricity only. A turbine outlet temperature must be supplied for every
selected electricity cycle:

```yaml
which_cycles: [orc, flash, kalina]
carrier_out: power
temp_turbine_out:
  ORC: 70.0
  Flash: 150.0
  Kalina: 80.0
min_dt: 10.0
```

`min_dt` is the minimum difference in °C between production and turbine outlet
temperature for electricity generation to be feasible.

## Reservoir and drilling assumptions

| Setting | Unit | Meaning |
| --- | --- | --- |
| `max_depth` | m | Deepest evaluated layer; must be a multiple of 250 m. |
| `include_egs` | boolean | Whether Enhanced Geothermal Systems may be used. |
| `EGS_cutoff` | m | Depth below which only EGS is assumed. |
| `SED_NAN_POLICY` | — | `always_egs` treats missing sediment thickness as 0 m; `depth_only` permits EGS only below `EGS_cutoff`. |
| `drilling_cost_curve` | — | Name of the active entry in `drilling_cost_curve_definitions`. |
| `stimulation_cost_per_doublet` | million €2020 | EGS stimulation cost per doublet. |
| `well_spacing` | km | Distance between production and injection wells. |
| `flowrate_egs` | kg/s | Mass flow rate per EGS doublet. |
| `flowrate_conventional` | kg/s | Mass flow rate per conventional doublet. |

The supplied example defines `thermogis`, `shamoushaki_2021`, `getem_2017`, and
`getem_2025` drilling-cost curves. Custom named definitions may be added under
`drilling_cost_curve_definitions` and selected with `drilling_cost_curve`.
The source temperature model covers 0–10 km, so `max_depth`, `EGS_cutoff`, and
all configured depth intervals must stay within that range.

## Plant economics

| Setting | Unit | Meaning |
| --- | --- | --- |
| `orc_money_per_kw` | €2020/kW | ORC plant CAPEX. |
| `flash_money_per_kw` | €2020/kW | Flash plant CAPEX. |
| `kalina_money_per_kw` | €2020/kW | Kalina plant CAPEX. |
| `pump_cost` | million €2020 | Pump cost per doublet pair. |
| `lifetime` | years | Economic lifetime. |
| `capacity_factor` | fraction | Annual availability for generation, from 0 to 1. |
| `fixed_OM_rate` | fraction/year | Fixed O&M as a fraction of CAPEX. |
| `parasitic_fraction` | fraction | Share of gross output consumed by auxiliary equipment. |

## Spatial aggregation

The workflow first converts each depth-resolved LCOE cube to two-dimensional
rasters with `method_2d_aggregation`:

- `LCOE_minimization` selects the least-cost feasible depth in each grid cell
  and calculates potential at that depth.
- `depth_categories` retains a separate LCOE and potential raster for every
  interval in `depth_categories`, such as `[0, 3000]` and `[3000, 6000]`.

It then aggregates the rasters to the polygons supplied through the `shapes`
path variable. Select the regional method with `method_region_aggregation`:

- `quantiles` divides each regional LCOE distribution into `n_quantiles`
  classes.
- `fixed_bins` applies the intervals in `fixed_bins` uniformly, using strings
  such as `"0-50"` and `"50-100"` in €2020/MWh.
- `cutoff` reports the potential at or below `lcoe_cutoff` and the mean LCOE of
  that potential. The cutoff is expressed in €2020/MWh.

Use `quantiles` or `fixed_bins` when downstream models need a cost-supply
distribution. Configuration validation requires valid model values, a defined
selected drilling-cost curve, and ordered, non-overlapping depth and LCOE bins.

## Inputs and outputs

Configuration controls model behaviour; file placement is controlled through
Snakemake path variables. See [`INTERFACE.yaml`](../INTERFACE.yaml) for the
required polygon input and public module outputs, and the
[integration example](../tests/integration/Snakefile) for a complete import.

The `shapes` input is GeoParquet and must have a declared CRS, unique
`shape_id` values, and valid Polygon or MultiPolygon geometry. `shape_class` is
optional; rows with `shape_class: maritime` are excluded from aggregation.

`geothermal_regions.parquet` has the following public columns:

| Column | Meaning |
| --- | --- |
| `technology` | Conversion cycle (`orc`, `flash`, or `kalina`). |
| `depth_category` | Configured depth interval, or `all_depths` after LCOE minimization. |
| `region_id` | String form of the input `shape_id`. |
| `price_bin`, `bin_id` | Human-readable and path-safe identifiers for the LCOE interval. |
| `bin_left`, `bin_right` | LCOE interval bounds in €2020/MWh; the lower bound is empty for cutoff aggregation. |
| `potential` | Net electric capacity in MW. |
| `mean_lcoe` | Potential-weighted mean LCOE in €2020/MWh. |
| `n_cells` | Number of contributing raster cells. |
| `geometry` | Aggregation-region polygon in the input CRS. |

The regional output and its `geothermal_region_outputs.tsv` manifest live
under `<results>/{shapes}/`. Model cubes and rasters live under
`<resources>/automatic/{shapes}/documents/`, and diagnostic figures under
`<resources>/automatic/{shapes}/figures/`. The LCOE overview remains in
`<results>/{shapes}/lcoe_overview.png`; regional-potential figures live under
`<results>/{shapes}/figures/`.

`<results>/{shapes}/overall_potential_summary.tsv` contains `technology` and
`potential_tw` columns. It reports study-wide potential in TW for each selected
technology and a `lowest_lcoe_technology` portfolio choosing the lowest-cost
available technology in each cell. For depth-category aggregation, totals are
summed across configured depth bands. These are raster-wide totals; regional
price bins and cutoffs are not applied to this summary.

The LCOE overview and cumulative-potential figures are produced for
`LCOE_minimization`. The regional-potential overview additionally requires
`cutoff` aggregation. Depth/cost diagnostics are available for both 2-D methods.
Technology comparisons within one run remain available.

### Migrating existing imports

Remove `baseline_scenario`, `perform_sensitivity`, `scenarios`, `sens_tech`,
`active_sensitivity_parameters`, and `sensitivity_parameters` from old configs;
these keys are no longer accepted. Remove the scenario directory (including
`base`) from result targets and path overrides. Replace the former study-wide
comparison tables with `overall_potential_summary.tsv`; its columns are now
`technology` and `potential_tw`. Existing generated files are not migrated or
deleted automatically.

## Development environments

The repository separates its Pixi development environment from the minimal
`module` environment used by Snakemake rules. After changing runtime
dependencies, run `pixi run export-snakemake-env module` to regenerate
`workflow/envs/module.yaml` and its platform pin files. Run
`pixi run test-quick` for the fast validation suite.
