"""Rules for reducing LCOE cubes to 2D rasters and regional outputs."""

rule make_2d_base:
    input:
        shapes="<shapes>",
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
        temp_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_temperature_at_min.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=cycles,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else []
        ),
    log:
        "<logs>/{shapes}/{scenario}/make_2d.log",
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=depth_categories_labels,
        cycles=cycles,
        max_depth=lambda wc: next(
            row["max_depth"] for row in scenario_rows if row["scenario"] == wc.scenario
        ),
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario=baseline_scenario_re,
    script:
        "../scripts/make_2d.py"


rule make_2d_scenario:
    input:
        shapes="<shapes>",
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
    log:
        "<logs>/{shapes}/{scenario}/make_2d.log",
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=depth_categories_labels,
        cycles=cycles,
        max_depth=lambda wc: next(
            row["max_depth"] for row in scenario_rows if row["scenario"] == wc.scenario
        ),
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
        manifest="<results>/{shapes}/{scenario}/geothermal_region_outputs.tsv",
    log:
        "<logs>/{shapes}/{scenario}/aggregate_to_regions.log",
    params:
        cycles=cycles,
        method_region_aggregation=config["method_region_aggregation"],
        n_quantiles=config.get("n_quantiles", 5),
        fixed_bins=config.get("fixed_bins", []),
        lcoe_cutoff=config.get("lcoe_cutoff"),
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/aggregate_to_regions.py"
