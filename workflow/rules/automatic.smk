"""Rules for automatic geothermal resource processing."""

import os
import re
import sys

sys.path.insert(
    0, os.path.dirname(workflow.source_path("../scripts/scenario_config.py"))
)
from scenario_config import build_scenario_rows


def depthcat_labels():
    return [
        f"{int(start/1000)}-{int(stop/1000)}km"
        for start, stop in config.get("depth_categories") or []
    ]


cycles = config["which_cycles"]
depth_categories_labels = depthcat_labels()
baseline_scenario = config["baseline_scenario"]
baseline_scenario_re = re.escape(baseline_scenario)
scenario_rows = build_scenario_rows(config)
scenarios = [row["scenario"] for row in scenario_rows]
scenarios_no_base = [s for s in scenarios if s != baseline_scenario]
make_2d_figure_outputs = (
    {
        "overview_fig": "<results>/{shapes}/{scenario}/lcoe_overview.png",
        "scatter_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_vs_potential_scatter.png",
        "avg_potential_depth_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/average_potential_per_cell_by_depth.png",
        "cumulative_curve_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/cumulative_potential_vs_lcoe.png",
        "cumulative_depth_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/cumulative_potential_vs_depth.png",
        "cumulative_temperature_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/cumulative_potential_vs_temperature.png",
        "temperature_lcoe_potential_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/potential_temperature_lcoe_heatmap.png",
        "temperature_lcoe_scatter_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/temperature_vs_lcoe_scatter.png",
        "temperature_lcoe_isopotential_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/temperature_lcoe_isopotential_contours.png",
        "temperature_lcoe_potential_scatter_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/temperature_lcoe_potential_scatter.png",
        "potential_overview_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/potential_overview.png",
        "depth_overview_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/depth_at_min_overview.png",
    }
    if config["method_2d_aggregation"] == "LCOE_minimization"
    else {}
)


rule build_scenario_table:
    """Generate the scenario TSV from config sensitivity settings."""
    output:
        tsv="<resources>/automatic/scenario_table.tsv",
    params:
        config=config,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/build_scenario_table.py"


rule build_temperature_voxel:
    input:
        asc=expand(
            "<resources>/automatic/grids_depth/temperature{depth}.asc",
            depth=sorted(
                glob_wildcards(
                    workflow.pathvars.apply(
                        "<resources>/automatic/grids_depth/temperature{depth}.asc"
                    )
                ).depth
            ),
        ),
    output:
        nc=temp("<resources>/automatic/{shapes}/generic/temp_voxel.nc"),
        qc_html="<resources>/automatic/{shapes}/generic/figures/temp_voxel_depth_slider.html",
    params:
        z_min=0,
        z_max=10000,
        z_step=250,
        vert_interp_kind="linear",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/create_temp_voxel.py"


rule crop_to_speed_up:
    input:
        nc=rules.build_temperature_voxel.output.nc,
        shapes="<shapes>",
    output:
        clipped_temp_voxel="<resources>/automatic/{shapes}/generic/clipped_temp_voxel.nc",
        qc_html="<resources>/automatic/{shapes}/generic/figures/clipped_temp_voxel_depth_slider.html",
    params:
        pad_m=25_000,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/crop_to_speed_up.py"


rule calculate_LCOE:
    input:
        nc=rules.crop_to_speed_up.output.clipped_temp_voxel,
        sed="<resources>/automatic/Global_sedmap_1deg.nc",
        eff="<resources>/automatic/efficiency_curves.csv",
        drilling_costs_2017="<resources>/automatic/baseline_drilling_2017.csv",
        drilling_costs_2025="<resources>/automatic/baseline_drilling_2025.csv",
        scenario_table=rules.build_scenario_table.output.tsv,
    output:
        **(
            {
                "qc_lcoe": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_{cycle}_depth_slider.html",
                "qc_potential": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/potential_{cycle}_depth_slider.html",
            }
            if config["write_qc_sliders"]
            else {}
        ),
        cube="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_lcoe_cube.nc",
    params:
        cycle=lambda wc: wc.cycle,
        write_qc_sliders=config["write_qc_sliders"],
        qc_stride=config["qc_stride"],
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/calculate_LCOE.py"


rule make_2d_base:
    input:
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
    output:
        **make_2d_figure_outputs,
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=depth_categories_labels,
        cycles=cycles,
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario=baseline_scenario_re,
    script:
        "../scripts/make_2d.py"


rule make_2d_scenario:
    input:
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
    output:
        **make_2d_figure_outputs,
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=depth_categories_labels,
        cycles=cycles,
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario=f"(?!{baseline_scenario_re}$).+",
    script:
        "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for each scenario."""
    input:
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
                cat=depth_categories_labels,
            )
        ),
        shapes="<shapes>",
    output:
        geoparquet="<results>/{shapes}/{scenario}/geothermal_regions.parquet",
        potential_overview="<results>/{shapes}/{scenario}/potential_overview_all_depths_bin_75.png",
    params:
        figdir="<resources>/automatic/{shapes}/scenarios/{scenario}/figures",
        cycles=cycles,
        method_region_aggregation=config["method_region_aggregation"],
        n_quantiles=config.get("n_quantiles", 5),
        fixed_bins=config.get("fixed_bins", []),
        lcoe_cutoff=config.get("lcoe_cutoff"),
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/aggregate_to_regions.py"


rule scenario_summary_fig:
    input:
        base_lcoe=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes,
            scenario=baseline_scenario,
            cycle=cycles,
        ),
        scen_lcoe=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
        base_pot=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
            shapes=wc.shapes,
            scenario=baseline_scenario,
            cycle=cycles,
        ),
        scen_pot=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
    output:
        fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/sensitivity_summary.png",
        metrics="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_metrics.tsv",
        boxplot_data="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_boxplot_data.parquet",
    params:
        baseline=baseline_scenario,
        cycles=cycles,
        sens_tech=config["sens_tech"],
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/sensitivity_summary.py"


rule sensitivity_boxplot_per_scenario:
    input:
        scenario_table=rules.build_scenario_table.output.tsv,
        boxplot_data="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_boxplot_data.parquet",
    output:
        base_fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/boxplot_base_dual_axis.png",
        lcoe_fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/boxplot_lcoe.png",
        potential_fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/boxplot_potential.png",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/sensitivity_boxplot_per_scenario.py"


if config["perform_sensitivity"]:

    rule sensitivity_vis:
        input:
            scenario_table=rules.build_scenario_table.output.tsv,
            metrics=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_metrics.tsv",
                shapes=wc.shapes,
                scenario=scenarios_no_base,
            ),
            boxplot_data=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_boxplot_data.parquet",
                shapes=wc.shapes,
                scenario=scenarios,
            ),
            geothermal_regions=lambda wc: expand(
                "<results>/{shapes}/{scenario}/geothermal_regions.parquet",
                shapes=wc.shapes,
                scenario=scenarios,
            ),
            potential_overviews=lambda wc: expand(
                "<results>/{shapes}/{scenario}/potential_overview_all_depths_bin_75.png",
                shapes=wc.shapes,
                scenario=scenarios,
            ),
        output:
            boxplot_lcoe_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_lcoe_top_6_winner.png",
            boxplot_potential_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_potential_top_6_winner.png",
            barplot_switch_rate_top6="<resources>/automatic/{shapes}/scenarios/_overview/barplot_switch_rate_top_6.png",
        params:
            baseline=baseline_scenario,
            metric_name="mean_abs_pct_change_sens",
            top_n=6,
            include_all=False,
        conda:
            "../envs/module_geothermal.yaml"
        script:
            "../scripts/sensitivity_vis.py"
