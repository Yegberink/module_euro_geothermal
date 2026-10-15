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
cycle and shape set. Supply all settings required by the selected modes; the
schema rejects missing required values. Unused settings can be omitted. See
[`tests/integration/test_config.yaml`](../tests/integration/test_config.yaml)
for a minimal power configuration. Model settings are tracked by Snakemake so changes
trigger recalculation.

## Conversion technologies

`which_cycles` selects one or more of `orc`, `flash`, and `kalina`.
`carrier_out` must be supplied as `power`, `heat`, or `heat_and_power`.
Heat-only targets do not calculate electricity or require cycle parameters.
A turbine outlet temperature must be supplied for every
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

## District heating

```yaml
carrier_out: heat_and_power  # use heat to skip electricity calculations
heat_production_temperature: 80.0
heat_reinjection_temperature: 40.0
thermal_plant_cost_per_kw: 234.0
heat_capacity_factor: 0.88
heat_network_buffer_km: 10.0
```

The `district_heating` path variable defaults to
`<resources>/user/{shapes}/areas.parquet`. Its GeoParquet must declare a CRS and
contain valid Polygon/MultiPolygon geometries, unique non-null `area_id` values,
and finite nonnegative `heat_potential_mwh_per_year`. IDs are normalized to
strings. This demand represents **connected annual demand**, not the larger
`heat_demand_mwh_per_year` total present in the supplied BENLDE data. Original
network attributes, including distribution costs, are retained.

### Thermal model

The model interpolates the shallowest production-temperature crossing between
adjacent finite layers. Exact positive-depth matches are accepted; missing
layers are not bridged and temperatures are not extrapolated. The interpolated
depth must meet `max_depth` and sediment/EGS restrictions. Infeasible cells have
NaN costs, potential, and depth.

One heat cost pass evaluates that depth, independently of electricity cycles,
electricity depth bands, and electricity price settings. Heat retains the
power model's nominal **100 km² per source cell** and doublet density
`100 / (2 × well_spacing²)`. These nominal areas are not recomputed from map
projection distortion. The heat equations are:

```text
thermal_MW = total_flow_kg_per_s × 4250 × (production_C − reinjection_C) / 1e6
annual_heat_MWh = thermal_MW × 8760 × heat_capacity_factor
thermal_plant_EUR = thermal_MW × 1000 × thermal_plant_cost_per_kw
LCOH = (CAPEX × annuity_factor + CAPEX × fixed_OM_rate) / annual_heat_MWh
annuity_factor = 0.075 / (1 − 1.075 ** (−lifetime))
```

CAPEX includes drilling, EGS stimulation when applicable, pumps, and the
thermal plant. Costs use €2020, including the thermal-plant default of
€2020 234/kWth. Flow rates, drilling curves, sediment NaN policy, and lifetime
are shared with the electrical model; reinjection must be cooler than
production. Disabling EGS rejects depths below the sediment/EGS cutoff.
Heat does not use turbine efficiency or the electricity parasitic fraction.
Network investment, transmission losses, and additional pumping-electricity
costs are excluded. Availability is annual; seasonal demand is not modeled.

The extent includes study-region bounds and network buffers on the original
temperature grid. Extent preparation stores grid indices and preserves an
unchanged file when only demand or a buffer within the same grid window changes.
Such changes rerun allocation without a new LCOH pass. Expanding the required
grid window correctly reruns heat calculation.

### Joint heat allocation

Cell centres within the configured distance of each network polygon are
candidates. Queries use EPSG:3035 and a spatial index with the `dwithin`
predicate. This avoids constructing expensive buffers of detailed polygons
and avoids scanning all cells for each network. Cost and supply rasters remain
on the source grid.

For each connected component of the network–cell graph, sparse linear
programming first maximizes total annual heat supplied and then minimizes its
annual geothermal cost at that maximum supply. Networks may share cells;
total use never exceeds cell potential or network demand. The global optimum
can route a flexible network to a more expensive cell so a network with fewer
alternatives can still be served. It does not guarantee equal shares or
prioritize particular networks when supply is scarce. Solver failures are
reported explicitly rather than silently returning a greedy allocation.

Fractional development scales capacity and investment together; there is no
whole-doublet rounding or additional part-load penalty. Network LCOH is
`sum(allocated_heat × cell_LCOH) / supplied_heat`. Networks with no supply have
zero supplied heat, their entire demand unmet, and null geothermal LCOH.

Heat follows the region containing the network polygon's centroid, calculated
in EPSG:3035. Exact-boundary ties use sorted region IDs. For a centroid in a gap,
choose the nearest region intersecting its network polygon and log the fallback.
A network without an intersecting non-maritime region causes an explicit error.
**FIXME:** replace centroid attribution with boundary-aware demand allocation
if needed. Power remains attributed to the source-cell region.

### Combined resource alternatives

`heat_and_power` represents alternative resource uses, **not cogeneration**:

| Category | Carrier | Meaning |
| --- | --- | --- |
| `power` | `power` | Eligible capacity remaining after proportional heat reservation. |
| `heat` | `heat` | Allocated heat without a retained power alternative. |
| `contested` | `heat` | Allocated heat with at least one retained power alternative. |
| `contested` | `power` | Reserved fraction of each retained power alternative. |

Power eligibility applies the selected cycles, depth bands, and price settings.
Bin boundaries are calculated before reserving heat capacity. If 20% of a cell
is used for heat, 20% of every eligible electrical alternative is contested and
80% remains available under `power`. Heat at that cell is reported once,
regardless of the number of cycles or depth bands. If no electrical alternative
survives the price settings, its allocated heat is `heat`.

Contested alternatives cannot be added together, nor can competing electrical
cycles be treated as independent resources. Their regional totals can lie in
different regions because heat follows the consuming network. Preserve the
cell-to-network allocation ledger when constructing cross-carrier constraints.
The ledger records source coordinates and consumer region; pair it with power
rasters to reconstruct cell-level alternatives. Heat-only mode has only `heat`
rows and applies no electricity price or depth aggregation settings.

### Heat outputs

| Output | Contents |
| --- | --- |
| `geothermal_heat_networks.parquet` | Original polygons/attributes plus `region_id`, `geothermal_supplied_mwh_per_year`, `geothermal_unmet_mwh_per_year`, `geothermal_lcoh_eur_per_mwh`. |
| `geothermal_heat_allocations.parquet` | Positive allocations: `area_id`, consumer `region_id`, `cell_id`, `row`, `col`, source `x`/`y`, `heat_mwh_per_year`, `cell_fraction`, `lcoh_eur_per_mwh`. |
| `heat_lcoh.tif` | Full-cell geothermal LCOH, €2020/MWh thermal. |
| `heat_potential.tif` | Full-cell annual geothermal heat, MWh/year. |
| `heat_depth.tif` | Interpolated drilling depth, metres. |
| `heat_allocated.tif` | Total heat allocated from each cell, MWh/year. |
| `heat_used_fraction.tif` | Allocated heat divided by full-cell heat potential. |

The first two outputs have public path variables and live in the result
directory. Heat rasters live in the automatic `documents` directory alongside
power rasters. All heat rasters use the same north-up grid and NaN nodata;
feasible unused cells have zero allocation and fraction. Ledger `row`, `col`,
and row-major `cell_id` refer to that grid; coordinates use its CRS. Technical
LCOH, potential, and depth maps describe supply before demand allocation.

In the regional Parquet, heat uses `depth_category: target_temperature`,
`bin_id: all`, and `price_bin: all`, with null bin bounds. Regions without
supplied heat have zero heat potential, zero contributing cells, and null LCOH.
In combined mode both heat categories are emitted for every non-maritime region;
electrical rows retain the existing convention of omitting empty bins.

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

The public interface uses the `shapes` wildcard. Internal rules additionally use
`cycle` (`orc`, `flash`, or `kalina`) and `dataset` (a download filename from
`workflow/internal/settings.yaml`). These internal wildcards do not occur in
public input/output defaults and are therefore not listed in `INTERFACE.yaml`.

The `shapes` input is GeoParquet and must have a declared CRS, unique
`shape_id` values, and valid Polygon or MultiPolygon geometry. `shape_class` is
optional; rows with `shape_class: maritime` are excluded from aggregation.

`geothermal_regions.parquet` has the following public columns:

| Column | Meaning |
| --- | --- |
| `category` | `power`, `heat`, or `contested`; distinguishes resource alternatives. |
| `carrier` | `power` or `heat`. |
| `potential_unit` | `MW_e` for power or `MWh_th/year` for heat. |
| `technology` | Conversion cycle (`orc`, `flash`, or `kalina`), or `geothermal_heat`. |
| `depth_category` | Configured depth interval, or `all_depths` after LCOE minimization. |
| `region_id` | String form of the input `shape_id`. |
| `price_bin`, `bin_id` | Human-readable and path-safe identifiers for the LCOE interval. |
| `bin_left`, `bin_right` | LCOE interval bounds in €2020/MWh; the lower bound is empty for cutoff aggregation. |
| `potential` | Net electric MW or allocated annual thermal MWh, as specified by `potential_unit`. |
| `mean_lcoe` | Capacity-weighted LCOE in €2020/MWh electric; null for heat. |
| `mean_lcoh` | Supplied-energy-weighted LCOH in €2020/MWh thermal; null for power. |
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
price bins, cutoffs, and heat reservations are not applied to this summary.
It and the electrical diagnostics remain unconstrained power analyses in
combined mode. The regional potential overview filters to the remaining
`power` category. These power-only targets are unavailable in heat-only mode.

The LCOE overview and cumulative-potential figures are produced for
`LCOE_minimization`. The regional-potential overview additionally requires
`cutoff` aggregation. Depth/cost diagnostics are available for both 2-D methods.
Technology comparisons within one run remain available.

Request `<results>/{shapes}/report_inputs.txt` to build every figure applicable to
the selected configuration plus the study-wide potential table. After this target
completes, render the same target with `--report <results>/{shapes}/report.html`.
Figures and the table are annotated for native Snakemake reports with labels
and categories. The `report_inputs` path variable can redirect this preparation
target in a parent workflow. See the [report commands](../README.md#figures-and-html-report).


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
`pixi run test-integration` for the standard validation suite.


## Land eligibility

Set `eligibility.enabled: true` (enabled in the example configuration) and supply
`<resources>/user/{shapes}/area_potential_geothermal.tif`, or override the
`area_potential` path variable. For `euro_all`, the default input is
`resources/user/euro_all/area_potential_geothermal.tif`.
Omitting eligibility or setting `eligibility.enabled: false` uses the full model
cell area and requires no area-potential input.

The input is the area-potential module output: one band of **available square
metres per pixel**, on a square, north-up WGS84 (EPSG:4326) grid. Exclusions and
buffers must already be applied by that module. This workflow no longer downloads
exclusion datasets or generates an eligibility grid from individual criteria.

`resample_eligibility` divides available area by each source pixel's WGS84
ellipsoidal area, then uses average resampling directly onto the exact subsurface
cell grid. Nodata source pixels contribute zero, and cells outside raster coverage
receive zero. Normalization is streamed through a temporary file to limit memory;
there is only one resampling operation and no intermediate fine-grid output.
The shared output is
`<resources>/automatic/{shapes}/eligibility/eligibility_fraction.tif`, with finite
fractions in **0–1**. Its extent covers the union of heat and power cells; both
calculations read exact grid subsets without further interpolation.

The model uses its nominal `100 km² × eligibility_fraction` available area before
calculating doublets, flow, thermal output, energy and investment. Positive fractions
leave LCOE/LCOH unchanged. Zero fractions follow existing infeasible-cell conventions
and supply no energy. Heat allocation and combined-mode resource reservation use
the reduced resource consistently. All exclusion choices and source-data limitations
now belong to the supplied area-potential dataset.
