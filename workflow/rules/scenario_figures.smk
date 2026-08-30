"""Rules for per-scenario cost and sensitivity figures."""


rule depth_cost_breakdown_four_panel:
    """Combine the three optimal-depth maps and cost breakdown in one figure."""
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/depth_cost_breakdown_four_panel.png",
    log:
        "<logs>/{shapes}/{scenario}/depth_cost_breakdown_four_panel.log",
    message:
        "Plot depth and cost diagnostics for geothermal scenario {wildcards.scenario}."
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/depth_cost_breakdown_four_panel.py"


rule sensitivity_summary:
    input:
        base_lcoe=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes,
            scenario=BASELINE_SCENARIO,
            cycle=CYCLES,
        ),
        scen_lcoe=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
        base_pot=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
            shapes=wc.shapes,
            scenario=BASELINE_SCENARIO,
            cycle=CYCLES,
        ),
        scen_pot=lambda wc: expand(
            "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        metrics="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_metrics.tsv",
        boxplot_data="<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_boxplot_data.parquet",
    log:
        "<logs>/{shapes}/{scenario}/sensitivity_summary.log",
    message:
        "Compare geothermal scenario {wildcards.scenario} with the baseline."
    params:
        baseline=BASELINE_SCENARIO,
        cycles=CYCLES,
        sens_tech=config["sens_tech"],
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/sensitivity_summary.py"
