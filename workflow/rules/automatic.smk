"""Rules used to download automatic resource files."""
import glob
from pathlib import Path
from snakemake.io import protected, temp
import pandas as pd
import numpy as np
import sys

# -----------------------------
# Helpers
# -----------------------------

def effective_max_depth(cfg):
    max_depth = float(cfg["max_depth"])
    if not bool(cfg["include_egs"]):
        egs_cutoff = float(cfg["EGS_cutoff"])
        if max_depth > egs_cutoff:
            print(
                f"⚠️ include_egs=False, so max_depth ({max_depth:g} m) must be <= "
                f"EGS_cutoff ({egs_cutoff:g} m). Using EGS_cutoff."
            )
        max_depth = min(max_depth, egs_cutoff)
    return max_depth


def norm_cycle(c):
    c = str(c).strip().lower()
    if c in ("orc", "flash", "kalina"):
        return c
    raise ValueError(f"Unknown cycle in config.which_cycles: {c!r}")


def depthcat_labels():
    return [f"{int(a/1000)}-{int(b/1000)}km" for a, b in config.get("depth_categories") or []]


def _read_scenario_table(path: str) -> dict:
    df = pd.read_csv(path, sep="\t", comment="#").replace({np.nan: None})
    if "scenario" not in df.columns:
        raise ValueError(f"{path} must have a 'scenario' column.")
    return df.astype({"scenario": str}).set_index("scenario").to_dict(orient="index")


# -----------------------------
# Global config-derived settings
# -----------------------------

CYCLES              = [norm_cycle(c) for c in config["which_cycles"]]
DEPTHCAT_LABELS     = depthcat_labels()
METHOD_2D           = config["method_2d_aggregation"]
METHOD_REGION       = config["method_region_aggregation"]
SHAPES              = config.get("target_shapes")
WRITE_SLIDERS       = bool(config.get("write_qc_sliders"))
PERFORM_SENSITIVITY = bool(config.get("perform_sensitivity"))
SCENARIO_TABLE      = "resources/automatic/scenario_table.tsv"
BASELINE_SCENARIO   = "base"
N_QUANTILES         = config.get("n_quantiles", 5)
FIXED_BINS          = config.get("fixed_bins", [])
LCOE_CUTOFF         = config.get("lcoe_cutoff", None)

# Keys that are workflow-control settings, not model parameters.
# These are excluded from the base_config row passed to build_scenario_table.py.
_NON_MODEL_KEYS = {
    "target_shapes",
    "perform_sensitivity",
    "baseline_scenario",
    "scenario_table",
    "active_sensitivity_parameters",
    "sensitivity_parameters",
    "write_qc_sliders",
    "method_2d_aggregation",
    "method_region_aggregation",
    "depth_categories",
    "n_quantiles",
    "fixed_bins",
    "lcoe_cutoff",
    "qc_stride",
    "sens_tech",
    "which_cycles",
    "drilling_cost_curve_definitions",
}

# -----------------------------
# Scenario table bootstrap
# -----------------------------

# Import the same logic used by the script so behaviour is identical.
sys.path.insert(0, str(Path(workflow.basedir) / "scripts"))
from build_scenario_table import _flatten_base_config, build_scenario_table as _build_scen

_base_cfg = {k: v for k, v in config.items() if k not in _NON_MODEL_KEYS}
_base_row = _flatten_base_config(_base_cfg)
_df = _build_scen(
    base_row              = _base_row,
    active_parameters     = config.get("active_sensitivity_parameters", []),
    parameter_definitions = config.get("sensitivity_parameters", {}),
    baseline_scenario     = BASELINE_SCENARIO,
)
SCEN = _df.set_index("scenario").replace({np.nan: None}).to_dict(orient="index")
del _base_cfg, _base_row, _df

SCENARIOS = list(SCEN.keys())

if BASELINE_SCENARIO not in SCEN:
    raise ValueError(
        f"baseline_scenario={BASELINE_SCENARIO!r} not found in scenario table. "
        f"Available: {SCENARIOS}"
    )

SCENARIOS_NO_BASE = [s for s in SCENARIOS if s != BASELINE_SCENARIO]
TARGET_SCENARIOS  = SCENARIOS if PERFORM_SENSITIVITY else [BASELINE_SCENARIO]

# -----------------------------
# Scenario helpers
# -----------------------------

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
scen_int   = lambda wc, key, default: scen_cast(wc, key, default, lambda v: int(float(v)))
scen_bool  = lambda wc, key, default: scen_cast(wc, key, default, bool)
scen_str   = lambda wc, key, default: scen_cast(wc, key, default, str)


def scen_temp_turbine_out(wc):
    base = config["temp_turbine_out"]
    return {
        c: float(scen_value(wc, f"temp_turbine_out_{c}", base[c]))
        for c in ("ORC", "Flash", "Kalina")
    }


# -----------------------------
# Input helpers for 2D tifs
# -----------------------------

def _2d_lcoe_tifs(shapes, scenario):
    """Return the list of 2D LCOE tif paths for a given shapes/scenario."""
    if METHOD_2D == "LCOE_minimization":
        return expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_lcoe.tif",
            shapes=shapes, scenario=scenario, cycle=CYCLES,
        )
    else:
        return expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_{cat}_lcoe.tif",
            shapes=shapes, scenario=scenario, cycle=CYCLES, cat=DEPTHCAT_LABELS,
        )


def _2d_pot_tifs(shapes, scenario):
    """Return the list of 2D potential tif paths for a given shapes/scenario."""
    if METHOD_2D == "LCOE_minimization":
        return expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_potential.tif",
            shapes=shapes, scenario=scenario, cycle=CYCLES,
        )
    else:
        return expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_{cat}_potential.tif",
            shapes=shapes, scenario=scenario, cycle=CYCLES, cat=DEPTHCAT_LABELS,
        )


# -----------------------------
# Aggregate target inputs
# -----------------------------

def _2d_files(suffix):
    return expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_{suffix}.tif",
        shapes=SHAPES, scenario=TARGET_SCENARIOS, cycle=CYCLES, suffix=suffix,
    )


all_geothermal_inputs = [*_2d_files("lcoe"), *_2d_files("potential")]

if METHOD_2D == "LCOE_minimization":
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_overview.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_vs_potential_scatter.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/average_potential_per_cell_by_depth.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_lcoe.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_depth.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_temperature.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_temperature_lcoe_heatmap.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_vs_lcoe_scatter.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_isopotential_contours.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_potential_scatter.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_overview.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )
    all_geothermal_inputs += expand(
        "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/depth_at_min_overview.png",
        shapes=SHAPES, scenario=TARGET_SCENARIOS,
    )

all_geothermal_inputs += expand(
    "results/{shapes}/scenarios/{scenario}/region_aggregation/geothermal_regions.parquet",
    shapes=SHAPES, scenario=TARGET_SCENARIOS,
)

_sens_target = lambda pattern, scenarios=TARGET_SCENARIOS: expand(
    pattern, shapes=SHAPES, scenario=scenarios
)

all_geothermal_inputs += [
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet"),
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/summary.png"),
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv"),
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_base_dual_axis.png"),
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_lcoe.png"),
    *_sens_target("results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_potential.png"),
]

if PERFORM_SENSITIVITY:
    all_geothermal_inputs += [
        *expand(
            "results/{shapes}/scenarios/sensitivity/boxplot_lcoe_top_6_winner.png",
            shapes=SHAPES,
        ),
        *expand(
            "results/{shapes}/scenarios/sensitivity/boxplot_potential_top_6_winner.png",
            shapes=SHAPES,
        ),
        *expand(
            "results/{shapes}/scenarios/sensitivity/barplot_switch_rate_top_6.png",
            shapes=SHAPES,
        ),
    ]

# -----------------------------
# Default target
# -----------------------------

rule all_geothermal:
    input: all_geothermal_inputs
    default_target: True

# -----------------------------
# Rules
# -----------------------------

rule build_scenario_table:
    """Generate the scenario TSV from config sensitivity settings."""
    output:
        tsv=SCENARIO_TABLE,
    params:
        baseline_scenario             = BASELINE_SCENARIO,
        active_sensitivity_parameters = config.get("active_sensitivity_parameters", []),
        sensitivity_parameters        = config.get("sensitivity_parameters", {}),
        base_config                   = {k: v for k, v in config.items() if k not in _NON_MODEL_KEYS},
    script:
        "../scripts/build_scenario_table.py"


rule build_temperature_voxel:
    input:
        asc=sorted(glob.glob("resources/automatic/grids_depth/temperature*.asc"))
    output:
        nc      = temp("results/{shapes}/temp_voxel.nc"),
        qc_html = "results/{shapes}/figures/temp_voxel_depth_slider.html",
    params:
        z_min=0, z_max=10000, z_step=250, vert_interp_kind="linear"
    script:
        "../scripts/create_temp_voxel.py"


rule crop_to_speed_up:
    input:
        nc     = "results/{shapes}/temp_voxel.nc",
        shapes = "resources/user/{shapes}/shapes.parquet",
    params:
        pad_m=25_000
    output:
        clipped_temp_voxel = protected("results/{shapes}/clipped_temp_voxel.nc"),
        qc_html            = "results/{shapes}/figures/clipped_temp_voxel_depth_slider.html",
    script:
        "../scripts/crop_to_speed_up.py"


rule calculate_LCOE:
    input:
        nc             = "results/{shapes}/clipped_temp_voxel.nc",
        sed            = "resources/automatic/Global_sedmap_1deg.nc",
        eff            = "resources/automatic/efficiency_curves.csv",
        scenario_table = SCENARIO_TABLE,
    output:
        cube = "results/{shapes}/scenarios/{scenario}/geothermal_lcoe/{cycle}_lcoe_cube.nc",
        **({
            "qc_lcoe":      "results/{shapes}/scenarios/{scenario}/geothermal_lcoe/figures/lcoe_{cycle}_depth_slider.html",
            "qc_potential": "results/{shapes}/scenarios/{scenario}/geothermal_lcoe/figures/potential_{cycle}_depth_slider.html",
        } if WRITE_SLIDERS else {}),
    params:
        cycle                        = lambda wc: wc.cycle,
        write_qc_sliders             = WRITE_SLIDERS,
        carrier_out                  = lambda wc: scen_str(wc,   "carrier_out",                  config["carrier_out"]),
        egs_cutoff                   = lambda wc: scen_float(wc, "EGS_cutoff",                   config["EGS_cutoff"]),
        sed_nan_policy               = lambda wc: scen_str(wc,   "SED_NAN_POLICY",               config["SED_NAN_POLICY"]),
        include_egs                  = lambda wc: scen_bool(wc,  "include_egs",                  config["include_egs"]),
        temp_turbine_out             = lambda wc: scen_temp_turbine_out(wc),
        drilling_cost_curve          = lambda wc: scen_str(wc,   "drilling_cost_curve",          config.get("drilling_cost_curve", "thermogis")),
        drilling_cost_curve_definitions = config.get("drilling_cost_curve_definitions", {}),
        max_depth                    = lambda wc: scen_float(wc, "max_depth",                    config["max_depth"]),
        stimulation_cost_per_doublet = lambda wc: scen_float(wc, "stimulation_cost_per_doublet", config.get("stimulation_cost_per_doublet")),
        pump_cost                    = lambda wc: scen_float(wc, "pump_cost",                    config["pump_cost"]),
        lifetime                     = lambda wc: scen_float(wc, "lifetime",                     config["lifetime"]),
        capacity_factor              = lambda wc: scen_float(wc, "capacity_factor",              config["capacity_factor"]),
        fixed_OM_rate                = lambda wc: scen_float(wc, "fixed_OM_rate",                config["fixed_OM_rate"]),
        parasitic_fraction           = lambda wc: scen_float(wc, "parasitic_fraction",           config["parasitic_fraction"]),
        well_spacing                 = lambda wc: scen_float(wc, "well_spacing",                 config["well_spacing"]),
        flowrate_egs                 = lambda wc: scen_float(wc, "flowrate_egs",                 config["flowrate_egs"]),
        flowrate_conventional        = lambda wc: scen_float(wc, "flowrate_conventional",        config["flowrate_conventional"]),
        orc_money_per_kw             = lambda wc: scen_float(wc, "orc_money_per_kw",             config["orc_money_per_kw"]),
        flash_money_per_kw           = lambda wc: scen_float(wc, "flash_money_per_kw",           config["flash_money_per_kw"]),
        kalina_money_per_kw          = lambda wc: scen_float(wc, "kalina_money_per_kw",          config["kalina_money_per_kw"]),
        min_dt                       = lambda wc: scen_float(wc, "min_dt",                       config["min_dt"]),
        qc_stride                    = config.get("qc_stride", 1),
    script:
        "../scripts/calculate_LCOE.py"


rule make_2d:
    input:
        cubes=lambda wc: expand(
            "results/{shapes}/scenarios/{scenario}/geothermal_lcoe/{cycle}_lcoe_cube.nc",
            shapes=wc.shapes, scenario=wc.scenario, cycle=CYCLES,
        )
    output:
        lcoe_tifs=(
            expand(
                "results/{{shapes}}/scenarios/{{scenario}}/2d_potential_cost/{cycle}_2d_lcoe.tif",
                cycle=CYCLES,
            )
            if METHOD_2D == "LCOE_minimization" else
            expand(
                "results/{{shapes}}/scenarios/{{scenario}}/2d_potential_cost/{cycle}_2d_{cat}_lcoe.tif",
                cycle=CYCLES, cat=DEPTHCAT_LABELS,
            )
        ),
        pot_tifs=(
            expand(
                "results/{{shapes}}/scenarios/{{scenario}}/2d_potential_cost/{cycle}_2d_potential.tif",
                cycle=CYCLES,
            )
            if METHOD_2D == "LCOE_minimization" else
            expand(
                "results/{{shapes}}/scenarios/{{scenario}}/2d_potential_cost/{cycle}_2d_{cat}_potential.tif",
                cycle=CYCLES, cat=DEPTHCAT_LABELS,
            )
        ),
        **(
            {
                "overview_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_overview.png",
                "scatter_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_vs_potential_scatter.png",
                "avg_potential_depth_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/average_potential_per_cell_by_depth.png",
                "cumulative_curve_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_lcoe.png",
                "cumulative_depth_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_depth.png",
                "cumulative_temperature_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_temperature.png",
                "temperature_lcoe_potential_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_temperature_lcoe_heatmap.png",
                "temperature_lcoe_scatter_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_vs_lcoe_scatter.png",
                "temperature_lcoe_isopotential_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_isopotential_contours.png",
                "temperature_lcoe_potential_scatter_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_potential_scatter.png",
                "potential_overview_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_overview.png",
                "depth_overview_fig": "results/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/depth_at_min_overview.png",
            }
            if METHOD_2D == "LCOE_minimization" else
            {}
        ),
    params:
        method_2d_aggregation = METHOD_2D,
        depth_categories      = config.get("depth_categories"),
        depthcat_labels       = DEPTHCAT_LABELS,
        cycles                = CYCLES,
    script:
        "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for each scenario."""
    input:
        lcoe_tifs = lambda wc: _2d_lcoe_tifs(wc.shapes, wc.scenario),
        pot_tifs  = lambda wc: _2d_pot_tifs(wc.shapes, wc.scenario),
        shapes    = "resources/user/{shapes}/shapes.parquet",
    output:
        geoparquet = "results/{shapes}/scenarios/{scenario}/region_aggregation/geothermal_regions.parquet",
        figdir     = directory("results/{shapes}/scenarios/{scenario}/region_aggregation/figures"),
    params:
        cycles                    = CYCLES,
        method_region_aggregation = METHOD_REGION,
        n_quantiles               = N_QUANTILES,
        fixed_bins                = FIXED_BINS,
        lcoe_cutoff               = LCOE_CUTOFF,
    script:
        "../scripts/aggregate_to_regions.py"


rule scenario_summary_fig:
    input:
        base_lcoe = lambda wc: expand(
            "results/{shapes}/scenarios/{baseline}/2d_potential_cost/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes, baseline=BASELINE_SCENARIO, cycle=CYCLES,
        ),
        scen_lcoe = lambda wc: expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes, scenario=wc.scenario, cycle=CYCLES,
        ),
        base_pot = lambda wc: expand(
            "results/{shapes}/scenarios/{baseline}/2d_potential_cost/{cycle}_2d_potential.tif",
            shapes=wc.shapes, baseline=BASELINE_SCENARIO, cycle=CYCLES,
        ),
        scen_pot = lambda wc: expand(
            "results/{shapes}/scenarios/{scenario}/2d_potential_cost/{cycle}_2d_potential.tif",
            shapes=wc.shapes, scenario=wc.scenario, cycle=CYCLES,
        ),
    output:
        fig          = "results/{shapes}/scenarios/{scenario}/sensitivity/summary.png",
        metrics      = "results/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv",
        boxplot_data = "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
    params:
        baseline  = BASELINE_SCENARIO,
        cycles    = CYCLES,
        sens_tech = config.get("sens_tech", "orc"),
    script:
        "../scripts/sensitivity_summary.py"

rule sensitivity_boxplot_per_scenario:
    input:
        boxplot_data = "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
    output:
        base_fig      = "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_base_dual_axis.png",
        lcoe_fig      = "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_lcoe.png",
        potential_fig = "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_potential.png",
    script:
        "../scripts/sensitivity_boxplot_per_scenario.py"

if PERFORM_SENSITIVITY:
    rule sensitivity_vis:
        input:
            scenario_table = SCENARIO_TABLE,
            metrics        = lambda wc: expand(
                "results/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv",
                shapes=wc.shapes, scenario=SCENARIOS_NO_BASE,
            ),
            boxplot_data   = lambda wc: expand(
                "results/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
                shapes=wc.shapes, scenario=SCENARIOS,
            ),
        output:
            boxplot_lcoe_top6_winner      = "results/{shapes}/scenarios/sensitivity/boxplot_lcoe_top_6_winner.png",
            boxplot_potential_top6_winner = "results/{shapes}/scenarios/sensitivity/boxplot_potential_top_6_winner.png",
            barplot_switch_rate_top6     = "results/{shapes}/scenarios/sensitivity/barplot_switch_rate_top_6.png",
        params:
            baseline    = BASELINE_SCENARIO,
            metric_name = "mean_abs_pct_change_sens",
            top_n       = 6,
            include_all = False,
        script:
            "../scripts/sensitivity_vis.py"
