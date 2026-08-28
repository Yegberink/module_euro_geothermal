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
            scenario=scenarios,
            cycle=cycles,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": depth_categories_labels}
            ),
        ),
        potential_tifs=lambda wc: expand(
            (
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif"
            ),
            shapes=wc.shapes,
            scenario=scenarios,
            cycle=cycles,
            **(
                {}
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else {"cat": depth_categories_labels}
            ),
        ),
    output:
        by_scenario="<results>/{shapes}/overall_potential_by_scenario.tsv",
        summary="<results>/{shapes}/overall_potential_summary.tsv",
    log:
        "<logs>/{shapes}/overall_potential.log",
    params:
        baseline=baseline_scenario,
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/overall_potential.py"


if config["perform_sensitivity"]:

    rule orc_temperature_bin_overview:
        input:
            scenario_table=rules.build_scenario_table.output.tsv,
            shapes="<shapes>",
            temperatures=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/orc_2d_temperature_at_min.tif",
                shapes=wc.shapes,
                scenario=scenarios,
            ),
            potentials=lambda wc: expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/orc_2d_potential.tif",
                shapes=wc.shapes,
                scenario=scenarios,
            ),
        output:
            fig="<results>/{shapes}/orc_potential_temperature_bins_by_scenario.png",
            data="<results>/{shapes}/orc_potential_temperature_bins_by_scenario.tsv",
        log:
            "<logs>/{shapes}/orc_temperature_bin_overview.log",
        params:
            scenarios=scenarios,
            bin_width=10,
        conda:
            "../envs/module_geothermal.yaml"
        script:
            "../scripts/orc_temperature_bin_overview.py"

    rule sensitivity_vis:
        input:
            scenario_table=rules.build_scenario_table.output.tsv,
            shapes="<shapes>",
            overall_potential=rules.overall_potential.output.summary,
            reference_lcoe=(
                f"<resources>/automatic/{{shapes}}/scenarios/{baseline_scenario}/documents/{cycles[0]}_2d_lcoe.tif"
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else []
            ),
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
        output:
            boxplot_lcoe_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_lcoe_top_6_winner.png",
            boxplot_potential_top6_winner="<resources>/automatic/{shapes}/scenarios/_overview/boxplot_potential_top_6_winner.png",
            barplot_switch_rate_top6="<resources>/automatic/{shapes}/scenarios/_overview/barplot_switch_rate_top_6.png",
            map_switch_transition_top4="<resources>/automatic/{shapes}/scenarios/_overview/map_switch_transition_top_4.png",
        log:
            "<logs>/{shapes}/sensitivity_vis.log",
        params:
            baseline=baseline_scenario,
            metric_name="mean_abs_pct_change_sens",
            top_n=6,
            include_all=False,
        conda:
            "../envs/module_geothermal.yaml"
        script:
            "../scripts/sensitivity_vis.py"
