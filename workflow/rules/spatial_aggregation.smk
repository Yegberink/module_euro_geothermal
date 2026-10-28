"""Rules for reducing LCOE cubes to 2D rasters and regional outputs."""

if POWER_ENABLED:

    rule make_2d:
        input:
            plots=workflow.source_path("../scripts/_plots.py"),
            shapes="<shapes>",
            cubes=lambda wc: expand(
                rules.calculate_LCOE.output.cube,
                shapes=wc.shapes,
                scenario=wc.scenario,
                cycle=CYCLES,
            ),
        output:
            **MAKE_2D_FIGURE_OUTPUTS,
            lcoe_tifs=expand(
                "<resources>/automatic/{shapes}/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            ),
            capex_tifs=expand(
                "<resources>/automatic/{shapes}/{scenario}/documents/{cycle}_2d_specific_capex.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            ),
            pot_tifs=expand(
                "<resources>/automatic/{shapes}/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            ),
        log:
            "<logs>/{shapes}/{scenario}/make_2d.log",
        conda:
            "../envs/module.yaml"
        params:
            cycles=CYCLES,
            scenario=lambda wc: config["scenario"][wc.scenario],
        message:
            "Reduce geothermal results for {wildcards.shapes} ({wildcards.scenario}) to 2-D rasters."
        script:
            "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for the configured parameters."""
    input:
        **AGGREGATION_INPUTS,
        plots=workflow.source_path("../scripts/_plots.py"),
        shapes="<shapes>",
    output:
        **AGGREGATION_OUTPUTS,
        geoparquet="<geothermal_regions>",
    log:
        "<logs>/{shapes}/{scenario}/aggregate_to_regions.log",
    conda:
        "../envs/module.yaml"
    params:
        **AGGREGATION_PARAMS,
        carrier_out=config["carrier_out"],
        scenario=lambda wc: config["scenario"][wc.scenario],
        cycles=CYCLES,
    message:
        "Aggregate geothermal results to {wildcards.shapes} ({wildcards.scenario})."
    script:
        "../scripts/aggregate_to_regions.py"
