"""Rules for per-scenario cost and sensitivity figures."""



rule cost_breakdown:
    """Plot technology-specific LCOE cost-share boxplots."""
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
    output:
        fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/cost_breakdown.png",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/cost_breakdown.py"


rule depth_cost_breakdown_four_panel:
    """Combine the three optimal-depth maps and cost breakdown in one figure."""
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=cycles,
        ),
    output:
        fig="<resources>/automatic/{shapes}/scenarios/{scenario}/figures/depth_cost_breakdown_four_panel.png",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/depth_cost_breakdown_four_panel.py"



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


