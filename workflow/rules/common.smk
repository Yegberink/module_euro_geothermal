import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(workflow.basedir, "scripts"))
from scenario_config import NON_MODEL_KEYS, build_scenario_rows, scenario_mapping


def norm_cycle(c):
    return str(c).strip().lower()


def depthcat_labels():
    return [
        f"{int(a/1000)}-{int(b/1000)}km"
        for a, b in config.get("depth_categories") or []
    ]


CYCLES = [norm_cycle(c) for c in config["which_cycles"]]
DEPTHCAT_LABELS = depthcat_labels()
METHOD_2D = config["method_2d_aggregation"]
METHOD_REGION = config["method_region_aggregation"]
SHAPES = config.get("target_shapes")
WRITE_SLIDERS = bool(config.get("write_qc_sliders"))
PERFORM_SENSITIVITY = bool(config.get("perform_sensitivity"))
SCENARIO_TABLE = config.get("scenario_table", "<scenario_table>")
BASELINE_SCENARIO = str(config.get("baseline_scenario", "base"))
N_QUANTILES = config.get("n_quantiles", 5)
FIXED_BINS = config.get("fixed_bins", [])
LCOE_CUTOFF = config.get("lcoe_cutoff", None)
TEMPERATURE_GRID_PATTERN = workflow.pathvars.apply("<temperature_grid>")
LCOE_CUBE_PATTERN = workflow.pathvars.apply("<lcoe_cube>")
LCOE_2D_PATTERN = workflow.pathvars.apply("<lcoe_2d>")
POTENTIAL_2D_PATTERN = workflow.pathvars.apply("<potential_2d>")
LCOE_2D_DEPTH_CATEGORY_PATTERN = workflow.pathvars.apply("<lcoe_2d_depth_category>")
POTENTIAL_2D_DEPTH_CATEGORY_PATTERN = workflow.pathvars.apply(
    "<potential_2d_depth_category>"
)
GEOTHERMAL_REGIONS_PATTERN = workflow.pathvars.apply("<geothermal_regions>")

SCENARIO_ROWS = build_scenario_rows(config)
SCEN = scenario_mapping(SCENARIO_ROWS)
SCENARIOS = list(SCEN.keys())
SCENARIOS_NO_BASE = [s for s in SCENARIOS if s != BASELINE_SCENARIO]
TARGET_SCENARIOS = SCENARIOS if PERFORM_SENSITIVITY else [BASELINE_SCENARIO]
BASELINE_ONLY_SCENARIOS = [BASELINE_SCENARIO]


def scen_value(wc, key, default=None):
    v = SCEN.get(wc.scenario, {}).get(key)
    return default if v is None else v


def scen_cast(wc, key, default, cast):
    v = scen_value(wc, key, default)
    if cast is bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
        return bool(v)
    return cast(v)


scen_float = lambda wc, key, default: scen_cast(wc, key, default, float)
scen_int = lambda wc, key, default: scen_cast(wc, key, default, lambda v: int(float(v)))
scen_bool = lambda wc, key, default: scen_cast(wc, key, default, bool)
scen_str = lambda wc, key, default: scen_cast(wc, key, default, str)


def scen_temp_turbine_out(wc):
    base = config["temp_turbine_out"]
    return {
        c: float(scen_value(wc, f"temp_turbine_out_{c}", base[c]))
        for c in ("ORC", "Flash", "Kalina")
    }


def drilling_cost_curve_definitions(input):
    definitions = deepcopy(config.get("drilling_cost_curve_definitions", {}))
    if "getem_2017" in definitions:
        definitions["getem_2017"]["csv_path"] = input.drilling_costs_2017
    if "getem_2025" in definitions:
        definitions["getem_2025"]["csv_path"] = input.drilling_costs_2025
    return definitions


def _2d_lcoe_tifs(shapes, scenario):
    if METHOD_2D == "LCOE_minimization":
        return expand(
            LCOE_2D_PATTERN,
            shapes=shapes,
            scenario=scenario,
            cycle=CYCLES,
        )
    return expand(
        LCOE_2D_DEPTH_CATEGORY_PATTERN,
        shapes=shapes,
        scenario=scenario,
        cycle=CYCLES,
        cat=DEPTHCAT_LABELS,
    )


def _2d_pot_tifs(shapes, scenario):
    if METHOD_2D == "LCOE_minimization":
        return expand(
            POTENTIAL_2D_PATTERN,
            shapes=shapes,
            scenario=scenario,
            cycle=CYCLES,
        )
    return expand(
        POTENTIAL_2D_DEPTH_CATEGORY_PATTERN,
        shapes=shapes,
        scenario=scenario,
        cycle=CYCLES,
        cat=DEPTHCAT_LABELS,
    )


def _cycle_pathvar_outputs(pathvar):
    pattern = workflow.pathvars.apply(pathvar)
    return [
        pattern.format(shapes="{shapes}", scenario="{scenario}", cycle=cycle)
        for cycle in CYCLES
    ]


def _depth_category_pathvar_outputs(pathvar):
    pattern = workflow.pathvars.apply(pathvar)
    return [
        pattern.format(
            shapes="{shapes}",
            scenario="{scenario}",
            cycle=cycle,
            cat=cat,
        )
        for cycle in CYCLES
        for cat in DEPTHCAT_LABELS
    ]


def _2d_files(suffix):
    if suffix == "lcoe":
        pattern = LCOE_2D_PATTERN
    elif suffix == "potential":
        pattern = POTENTIAL_2D_PATTERN
    else:
        pattern = "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_{suffix}.tif"
    return expand(
        pattern,
        shapes=SHAPES,
        scenario=TARGET_SCENARIOS,
        cycle=CYCLES,
        suffix=suffix,
    )


all_geothermal_inputs = [*_2d_files("lcoe"), *_2d_files("potential")]

if METHOD_2D == "LCOE_minimization":
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_overview.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_vs_potential_scatter.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/average_potential_per_cell_by_depth.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_lcoe.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_depth.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_temperature.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_temperature_lcoe_heatmap.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_vs_lcoe_scatter.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_isopotential_contours.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_potential_scatter.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_overview.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/depth_at_min_overview.png",
        shapes=SHAPES,
        scenario=BASELINE_ONLY_SCENARIOS,
    )

all_geothermal_inputs += expand(
    GEOTHERMAL_REGIONS_PATTERN,
    shapes=SHAPES,
    scenario=BASELINE_ONLY_SCENARIOS,
)

_sens_target = lambda pattern, scenarios=TARGET_SCENARIOS: expand(
    pattern, shapes=SHAPES, scenario=scenarios
)

all_geothermal_inputs += [
    *_sens_target(
        "<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet"
    ),
    *_sens_target("<results>/{shapes}/scenarios/{scenario}/sensitivity/summary.png"),
    *_sens_target("<results>/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv"),
    *_sens_target(
        "<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_base_dual_axis.png"
    ),
    *_sens_target(
        "<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_lcoe.png"
    ),
    *_sens_target(
        "<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_potential.png"
    ),
]

if PERFORM_SENSITIVITY:
    all_geothermal_inputs += [
        *expand(
            "<results>/{shapes}/scenarios/sensitivity/boxplot_lcoe_top_6_winner.png",
            shapes=SHAPES,
        ),
        *expand(
            "<results>/{shapes}/scenarios/sensitivity/boxplot_potential_top_6_winner.png",
            shapes=SHAPES,
        ),
        *expand(
            "<results>/{shapes}/scenarios/sensitivity/barplot_switch_rate_top_6.png",
            shapes=SHAPES,
        ),
    ]


rule all_geothermal:
    input:
        all_geothermal_inputs,
