"""Rules for study-wide potential and sensitivity analysis."""


rule overall_potential:
    """Tabulate study-wide potential by scenario and its sensitivity envelope."""
    input:
        scenario_table=rules.build_scenario_table.output.tsv,
        lcoe_tifs=lambda wc: expand(
            (
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif"
            ),
            shapes=wc.shapes,
            scenario=SCENARIOS,
            cycle=CYCLES,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": DEPTH_CATEGORY_LABELS}
            ),
        ),
        potential_tifs=lambda wc: expand(
            (
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif"
            ),
            shapes=wc.shapes,
            scenario=SCENARIOS,
            cycle=CYCLES,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": DEPTH_CATEGORY_LABELS}
            ),
        ),
    output:
        by_scenario="<overall_potential_by_scenario>",
        summary="<overall_potential_summary>",
    log:
        "<logs>/{shapes}/overall_potential.log",
    message:
        "Summarize geothermal potential across all configured scenarios."
    params:
        baseline=BASELINE_SCENARIO,
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/overall_potential.py"


if config["perform_sensitivity"]:

    rule sensitivity_vis:
        input:
            scenario_table=rules.build_scenario_table.output.tsv,
            shapes="<shapes>",
            reference_lcoe=(
                f"<resources>/automatic/{{shapes}}/scenarios/{BASELINE_SCENARIO}/documents/{CYCLES[0]}_2d_lcoe.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else []
            ),
            metrics=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_metrics.tsv",
                shapes=wc.shapes,
                scenario=NON_BASELINE_SCENARIOS,
            ),
            boxplot_data=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/sensitivity_boxplot_data.parquet",
                shapes=wc.shapes,
                scenario=SCENARIOS,
            ),
        output:
            boxplot_lcoe_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_lcoe_top_6_winner.png",
            boxplot_potential_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_potential_top_6_winner.png",
            map_switch_transition_top4="<resources>/automatic/{shapes}/scenarios/_overview/map_switch_transition_top_4.png",
        log:
            "<logs>/{shapes}/sensitivity_vis.log",
        message:
            "Plot study-wide geothermal sensitivity results for {wildcards.shapes}."
        params:
            baseline=BASELINE_SCENARIO,
            metric_name="mean_abs_pct_change_sens",
            top_n=6,
            include_all=False,
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/sensitivity_vis.py"
