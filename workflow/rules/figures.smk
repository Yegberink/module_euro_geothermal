"""Rules for single-run model figures."""


rule depth_cost_breakdown_four_panel:
    """Combine the three optimal-depth maps and cost breakdown in one figure."""
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            cycle=CYCLES,
        ),
    output:
        fig=report(
            "<resources>/automatic/{shapes}/figures/depth_cost_breakdown_four_panel.png",
            category="Electricity potential",
            subcategory="{shapes}",
            labels={"Figure": "Optimal depth and cost breakdown"},
        ),
    log:
        "<logs>/{shapes}/depth_cost_breakdown_four_panel.log",
    conda:
        "../envs/module.yaml"
    message:
        "Plot depth and cost diagnostics for {wildcards.shapes}."
    script:
        "../scripts/depth_cost_breakdown_four_panel.py"
