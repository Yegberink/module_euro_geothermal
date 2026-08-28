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
  baseline_scenario: base
  perform_sensitivity: false
  which_cycles: [orc]
  carrier_out: power
  # Add the remaining model parameters from config/config.yaml.
```

## Workflow and scenario controls

| Setting | Meaning |
| --- | --- |
| `baseline_scenario` | Name used for the reference scenario, normally `base`. |
| `perform_sensitivity` | Enables the study-wide sensitivity figures and summaries. Set it to `false` for a baseline-only run. |
| `scenarios` | Optional mapping or list of named scenario overrides. When supplied, it takes precedence over automatic scenario expansion. |

Explicit scenarios inherit all baseline model values and replace only the
listed settings:

```yaml
baseline_scenario: base
perform_sensitivity: true
scenarios:
  base: {}
  shallow:
    max_depth: 3000
  long_lifetime:
    lifetime: 40.0
```

## Conversion technologies

`which_cycles` selects one or more of `orc`, `flash`, and `kalina`.
`carrier_out` accepts `power`, `heat`, or `both`; currently only `power` is
fully implemented. A turbine outlet temperature must be supplied for every
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

`cutoff` is the quickest choice for large sensitivity sweeps. Use `quantiles`
or `fixed_bins` when downstream models need a cost-supply distribution.

## Sensitivity analysis

Set `perform_sensitivity: true`, list the variables to vary in
`active_sensitivity_parameters`, and define each variable in
`sensitivity_parameters`. The automatic sweep changes one parameter at a time
while keeping all other baseline assumptions fixed.

Numeric parameters support either symmetric percentage changes:

```yaml
active_sensitivity_parameters: [flowrate_egs]
sensitivity_parameters:
  flowrate_egs:
    max_dev_pct: 30
    steps: 1
```

or explicit values:

```yaml
active_sensitivity_parameters: [max_depth]
sensitivity_parameters:
  max_depth:
    absolute_values: [3000, 9000]
```

For a boolean parameter, provide a boolean placeholder and the workflow flips
the baseline value. For a categorical parameter, provide the values to explore:

```yaml
active_sensitivity_parameters: [include_egs, drilling_cost_curve]
sensitivity_parameters:
  include_egs: true
  drilling_cost_curve:
    - thermogis
    - shamoushaki_2021
    - getem_2017
    - getem_2025
```

`sens_tech` (`orc`, `flash`, or `kalina`) selects the reference technology for
sensitivity summary plots. See the repository's
[`config/config.yaml`](./config.yaml) for the full set of supported sensitivity
parameters and example ranges.

## Inputs and outputs

Configuration controls model behaviour; file placement is controlled through
Snakemake path variables. See [`INTERFACE.yaml`](../INTERFACE.yaml) for the
required polygon input and public module outputs, and the
[integration example](../tests/integration/Snakefile) for a complete import.
