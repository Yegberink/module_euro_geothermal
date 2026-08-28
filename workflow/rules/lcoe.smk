"""Scenario setup, input preparation, and LCOE calculation rules."""

import re


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
        "potential_presentation_fig": "<results>/{shapes}/{scenario}/potential_overview_presentation.png",
        "scatter_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_vs_potential_scatter.png",
        "technology_boxplot_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_potential_boxplots.png",
        "technology_split_boxplot_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_potential_split_boxplots.png",
        "technology_cumulative_four_panel_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/lcoe_potential_cumulative_four_panel.png",
        "avg_potential_depth_fig": "<resources>/automatic/{shapes}/scenarios/{scenario}/figures/average_potential_per_cell_by_depth.png",
        "depth_profile_figs": expand(
            "<resources>/automatic/{{shapes}}/scenarios/{{scenario}}/figures/depth_profile_{cycle}.png",
            cycle=cycles,
        ),
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
    log:
        "<logs>/build_scenario_table.log",
    message:
        "Build the configured geothermal scenario table."
    params:
        config=config,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/build_scenario_table.py"


rule crop_to_speed_up:
    input:
        nc="<resources>/automatic/temperature_voxel.nc",
        shapes="<shapes>",
    output:
        clipped_temp_voxel="<resources>/automatic/{shapes}/generic/clipped_temp_voxel.nc",
    log:
        "<logs>/{shapes}/crop_to_speed_up.log",
    message:
        "Crop the European temperature model to {wildcards.shapes}."
    params:
        pad_m=25_000,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/crop_to_speed_up.py"


rule resample_gst1_sediment:
    """Align GST-1 sediment thickness to the full temperature grid."""
    input:
        archive=rules.download_gst1.output.archive,
        temperature="<resources>/automatic/temperature_voxel.nc",
    output:
        sediment="<resources>/automatic/GST1_sediment_thickness.nc",
    log:
        "<logs>/automatic/resample_gst1_sediment.log",
    message:
        "Align GST-1 sediment thickness with the temperature grid."
    params:
        member=internal["resources"]["automatic"]["gst1"]["member"],
        doi=internal["resources"]["automatic"]["gst1"]["doi"],
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/resample_gst1.py"


rule calculate_LCOE:
    input:
        nc=rules.crop_to_speed_up.output.clipped_temp_voxel,
        sed=rules.resample_gst1_sediment.output.sediment,
        eff=workflow.source_path("../internal/efficiency_curves.csv"),
        drilling_costs_2017=workflow.source_path(
            "../internal/baseline_drilling_2017.csv"
        ),
        drilling_costs_2025=workflow.source_path(
            "../internal/baseline_drilling_2025.csv"
        ),
        scenario_table=rules.build_scenario_table.output.tsv,
    output:
        cube="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_lcoe_cube.nc",
    log:
        "<logs>/{shapes}/{scenario}/{cycle}/calculate_LCOE.log",
    message:
        "Calculate {wildcards.cycle} geothermal potential and LCOE for {wildcards.scenario}."
    params:
        cycle=lambda wc: wc.cycle,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/calculate_LCOE.py"
