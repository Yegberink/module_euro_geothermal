"""Rules for study-wide potential."""


rule overall_potential:
    """Tabulate study-wide potential by technology."""
    input:
        lcoe_tifs=lambda wc: expand(
            (
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_lcoe.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_lcoe.tif"
            ),
            shapes=wc.shapes,
            cycle=CYCLES,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": DEPTH_CATEGORY_LABELS}
            ),
        ),
        potential_tifs=lambda wc: expand(
            (
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_potential.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_potential.tif"
            ),
            shapes=wc.shapes,
            cycle=CYCLES,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": DEPTH_CATEGORY_LABELS}
            ),
        ),
    output:
        summary="<overall_potential_summary>",
    log:
        "<logs>/{shapes}/overall_potential.log",
    message:
        "Summarize geothermal potential for {wildcards.shapes}."
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/overall_potential.py"
