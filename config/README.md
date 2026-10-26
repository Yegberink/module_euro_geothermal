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
selected electricity cycle.
See the annotated [configuration](config.yaml) for cycle temperatures and units.

`min_dt` is the minimum difference in °C between production and turbine outlet
temperature for electricity generation to be feasible.

## District heating

Set `carrier_out` to `heat` or `heat_and_power` and supply the district-heating
settings in the annotated [configuration](config.yaml).

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

One heat cost pass evaluates that depth, independently of electricity cycles
and electricity price settings. Heat retains the
power model's nominal **100 km² per source cell** and doublet density
`100 / (2 × well_spacing²)`. These nominal areas are not recomputed from map
projection distortion. The heat equations are:

```text
thermal_MW = total_flow_kg_per_s × Cp_water × (production_C − reinjection_C) / 1e6
annual_heat_MWh = thermal_MW × 8760 × heat_capacity_factor
thermal_plant_EUR = thermal_MW × 1000 × thermal_plant_cost_per_kw
LCOH = (CAPEX × annuity_factor + CAPEX × fixed_OM_rate) / annual_heat_MWh
annuity_factor = discount_rate / (1 − (1 + discount_rate) ** (−lifetime))
```

The water heat capacity is defined in
[`calculate_LCOH.py`](../workflow/scripts/calculate_LCOH.py).

CAPEX includes drilling, EGS stimulation when applicable, pumps, and the
thermal plant. Costs use €2020; the annotated configuration defines plant costs.
Flow rates, drilling curves, lifetime, and discount rate
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

`heat_allow_cross_region_supply` controls whether candidates may lie outside
the network's assigned region. Set it to `false` to require each source-cell
centre to belong to that region, or `true` to permit sources across region
boundaries. Centres on shared boundaries or in overlapping polygons are assigned
to the first region in sorted string-ID order, consistently for local allocation
and aggregation. Both settings retain `heat_network_buffer_km`
as the distance limit. Region boundaries come from the supplied `shapes` input;
the network attribution described below also applies to networks spanning
multiple regions. A network with insufficient local supply retains unmet demand
when cross-region supply is disabled.

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

Power eligibility applies the selected cycles and price settings.
Bin boundaries are calculated before reserving heat capacity. If 20% of a cell
is used for heat, 20% of every eligible electrical alternative is contested and
80% remains available under `power`. Heat at that cell is reported once,
regardless of the number of cycles. If no electrical alternative
survives the price settings, its allocated heat is `heat`.

Contested alternatives cannot be added together, nor can competing electrical
cycles be treated as independent resources. Their regional totals can lie in
different regions because heat follows the consuming network. The cell-to-network
allocation ledger preserves the links needed to construct cross-carrier constraints. Heat-only mode applies no
electricity price settings.

### Allocation outputs

Allocation files live under `<resources>/automatic/{shapes}/documents/` and
have no public path variables. The four raster categories are:

| Output | Contents |
| --- | --- |
| `{cycle}_power_potential.tif` | Eligible electrical capacity remaining after heat reservation, MW_e. |
| `{cycle}_contested_power_potential.tif` | Eligible electrical capacity reserved by heat allocation, MW_e; combined mode. |
| `heat_allocated_potential.tif` | Allocated heat without an eligible electrical alternative, MWh_th/year. |
| `contested_heat_potential.tif` | Allocated heat with an eligible electrical alternative, MWh_th/year. |

Power files are separate for each selected technology. Their sum within a
technology recovers eligible capacity before heat reservation; technologies
remain competing alternatives. Heat files describe source locations, and their
sum recovers total allocated heat. In heat-only mode the contested heat raster
is zero at feasible cells.

`heat_allocations.parquet` records positive cell-to-network links in both local
and cross-region supply modes: `area_id`, consumer `region_id`, `cell_id`, `row`,
`col`, source `x`/`y`, `heat_mwh_per_year`, `cell_fraction`, `lcoh_eur_per_mwh`,
and `network_distance_m`. Coordinates use the source grid CRS; row, column,
and row-major cell IDs refer to the heat grid. Distance is the shortest straight-line
distance from the cell centre to the supplied network polygon in EPSG:3035 metres,
with zero for centres inside or on its boundary. Regional heat statistics use
these links, including when a cell supplies multiple networks.

Technical heat rasters also include `heat_lcoh.tif`, `heat_potential.tif`, and
`heat_depth.tif`, describing cost, full-cell annual heat, and drilling depth
before demand allocation. `heat_allocated.tif` stores total allocated heat and
`heat_used_fraction.tif` its fraction of full-cell potential. Heat rasters use
the same north-up grid and NaN nodata; feasible unused cells have zero allocation.
Allocation power rasters use zero for price-excluded feasible cells and NaN for
physically infeasible cells.

Regional heat potential is the annual heat that geothermal energy can supply
to networks assigned to that region, including imports when enabled. Sum its
`heat` and `contested` heat-carrier rows for total supplied heat. Heat uses
`price_bin: all` and null bin bounds. Regions without supplied heat retain a
zero `heat` row with zero contributing cells and null LCOH. Zero contested-heat
rows are omitted; positive contested-power rows are retained even when heat is
consumed in another region. Empty electrical price bins are omitted.

## Reservoir and drilling assumptions

| Setting | Unit | Meaning |
| --- | --- | --- |
| `max_depth` | m | Deepest evaluated layer; must be a multiple of 250 m. |
| `include_egs` | boolean | Whether Enhanced Geothermal Systems may be used. |
| `EGS_cutoff` | m | Depth below which only EGS is assumed. |
| `drilling_cost_curve` | — | Drilling cost relation shared by heat and power; choices are listed in the [configuration](config.yaml). |
| `stimulation_cost_per_doublet` | million €2020 | EGS stimulation cost per doublet. |
| `well_spacing` | km | Distance between production and injection wells. |
| `flowrate_egs` | kg/s | Mass flow rate per EGS doublet. |
| `flowrate_conventional` | kg/s | Mass flow rate per conventional doublet. |

Missing sediment thickness after resampling is treated as 0 m, so those cells
require EGS at every positive depth.

Curve formulas, coefficients, and source price years are defined in
[`_economics.py`](../workflow/scripts/_economics.py). The GETEM curves are fitted
to bundled per-well depth/cost data.
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
| `discount_rate` | fraction/year | Annual discount rate shared by heat and power, from 0 to 1; the example uses 0.075 (7.5%). Zero gives undiscounted costs. |
| `capacity_factor` | fraction | Annual availability for generation, from 0 to 1. |
| `fixed_OM_rate` | fraction/year | Fixed O&M as a fraction of CAPEX. |
| `parasitic_fraction` | fraction | Share of gross output consumed by auxiliary equipment. |

### Currency conversion assumptions

All euro-denominated inputs and results use **constant 2020 euros (EUR2020)**,

Heat and power share the conversion in
[`_economics.py`](../workflow/scripts/_economics.py):

```text
EUR2020 = source-year USD × (US CPI2020 / US CPI_source_year) / 1.1422
```

The CPI values are US city average, all-items CPI-U, not seasonally adjusted,
annual averages (1982–84 = 100). The exchange rate is the 2020 annual average,
**1 EUR = 1.1422 USD**. 

| Source price year | US CPI-U | Multiplier from USD to EUR2020 |
| --- | ---: | ---: |
| 2017 | 245.120 | 0.924404 |
| 2020 | 258.811 | 0.875503 |
| 2021 | 270.970 | 0.836218 |
| 2025 | 321.943 | 0.703820 |


## Spatial aggregation

The workflow converts each depth-resolved LCOE cube to two-dimensional rasters
by selecting the least-cost feasible depth in each grid cell for each technology,
with potential evaluated at that depth.

It then aggregates the rasters to the polygons supplied through the `shapes`
path variable. Select the regional method with `method_region_aggregation`:

- `quantiles` divides each input raster's finite LCOE distribution into `n_quantiles`
  classes, using the same boundaries for all regions.
- `fixed_bins` applies the intervals in `fixed_bins` uniformly, using strings
  in €2020/MWh.
- `cutoff` reports the potential at or below `lcoe_cutoff` and the mean LCOE of
  that potential. The cutoff is expressed in €2020/MWh.

Use `quantiles` or `fixed_bins` when downstream models need a cost-supply
distribution. Configuration validation requires valid model values, a supported
selected drilling-cost curve, and ordered, non-overlapping LCOE bins.

## Inputs and outputs

The supplied temperature grid must declare its CRS; missing metadata is an
ingestion error.

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
| `region_id` | String form of the input `shape_id`. |
| `category` | `power`, `heat`, or `contested`; distinguishes resource alternatives. |
| `carrier` | `power` or `heat`. |
| `technology` | Conversion cycle (`orc`, `flash`, or `kalina`), or `geothermal_heat`. |
| `price_bin` | Human-readable identifier for the LCOE interval, or `all` for heat. |
| `bin_left` | Lower LCOE bound in €2020/MWh; null for cutoff aggregation and heat. |
| `bin_right` | Upper LCOE bound in €2020/MWh; null for heat. |
| `potential` | Net electric capacity or annual geothermal heat supplied to the consuming region. |
| `potential_unit` | `MW_e` for power or `MWh_th/year` for heat. |
| `mean_lcoe` | Capacity-weighted LCOE in €2020/MWh electric; null for heat. |
| `mean_lcoh` | Supplied-energy-weighted LCOH in €2020/MWh thermal; null for power. |
| `mean_network_distance_m` | Supplied-energy-weighted cell-to-network distance in metres per consumer region and heat category; null for power or zero heat supply. |
| `n_cells` | Number of contributing raster cells. |
| `geometry` | Aggregation-region polygon in the input CRS. |

The columns occur in the order above. The regional table lives under
`<results>/{shapes}/`. Model cubes, allocation links, and rasters live under
`<resources>/automatic/{shapes}/documents/`. All figures live under
`<results>/{shapes}/figures/`.

Electrical diagnostics describe unconstrained power resources in combined mode.
The regional potential overview filters to the remaining `power` category.
Power figures are unavailable in heat-only mode.

Electricity figures include LCOE, raster potential, and regional potential
four-panel maps. Regional figures sum the configured price bins and retain panels
for technologies with no potential. Heat figures include single-panel LCOH,
allocated source-cell supply, and allocated regional supply maps. The heat supply
raster map includes only cells supplying networks; LCOH uses the full raster. Heat quantities are shown
as annual thermal energy; electricity potential is capacity.

See the [report instructions](../README.md#figures-and-html-report)
and [`tests/local_test.py`](../tests/local_test.py). Render reports under `results/`.

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
Supply `eligibility.enabled` explicitly. Setting it to `false` uses the full model
cell area and requires no area-potential input.

The input is the area-potential module output: one band of **available square
metres per pixel**, on a square, north-up WGS84 (EPSG:4326) grid. Exclusions and
buffers must already be applied by that module.

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
belong to the supplied area-potential dataset.
