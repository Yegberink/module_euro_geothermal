"""Rules for reducing LCOE cubes to 2D rasters and regional outputs."""


rule make_2d_base:
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        **MAKE_2D_FIGURE_OUTPUTS,
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
    log:
        "<logs>/{shapes}/{scenario}/make_2d.log",
    message:
        "Reduce baseline geothermal results for {wildcards.shapes} to 2-D rasters."
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=DEPTH_CATEGORY_LABELS,
        cycles=CYCLES,
        max_depth=lambda wc: next(
            row["max_depth"] for row in SCENARIO_ROWS if row["scenario"] == wc.scenario
        ),
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        scenario=BASELINE_SCENARIO_RE,
    script:
        "../scripts/make_2d.py"


rule make_2d_scenario:
    input:
        shapes="<shapes>",
        cubes=lambda wc: expand(
            rules.calculate_LCOE.output.cube,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        **MAKE_2D_FIGURE_OUTPUTS,
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
    log:
        "<logs>/{shapes}/{scenario}/make_2d.log",
    message:
        "Reduce geothermal scenario {wildcards.scenario} to 2-D rasters."
    params:
        method_2d_aggregation=config["method_2d_aggregation"],
        depth_categories=config.get("depth_categories"),
        depthcat_labels=DEPTH_CATEGORY_LABELS,
        cycles=CYCLES,
        max_depth=lambda wc: next(
            row["max_depth"] for row in SCENARIO_ROWS if row["scenario"] == wc.scenario
        ),
    conda:
        "../envs/module.yaml"
    wildcard_constraints:
        scenario=f"(?!{BASELINE_SCENARIO_RE}$).+",
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
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
            )
            if config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/scenarios/{scenario}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                scenario="{scenario}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        shapes="<shapes>",
    output:
        geoparquet="<geothermal_regions>",
        manifest="<results>/{shapes}/{scenario}/geothermal_region_outputs.tsv",
    log:
        "<logs>/{shapes}/{scenario}/aggregate_to_regions.log",
    message:
        "Aggregate geothermal scenario {wildcards.scenario} to {wildcards.shapes}."
    params:
        cycles=CYCLES,
        method_region_aggregation=config["method_region_aggregation"],
        n_quantiles=config.get("n_quantiles", 5),
        fixed_bins=config.get("fixed_bins", []),
        lcoe_cutoff=config.get("lcoe_cutoff"),
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/aggregate_to_regions.py"


if (
    config["method_2d_aggregation"] == "LCOE_minimization"
    and config["method_region_aggregation"] == "cutoff"
):

    rule regional_potential_overview:
        """Plot regional potential below the configured LCOE cutoff."""
        input:
            geoparquet="<geothermal_regions>",
            shapes="<shapes>",
        output:
            fig=(
                f"<results>/{{shapes}}/{{scenario}}/figures/"
                f"potential_overview_all_depths_cutoff_{CUTOFF_LABEL}.png"
            ),
        log:
            "<logs>/{shapes}/{scenario}/regional_potential_overview.log",
        message:
            "Plot geothermal potential below the configured LCOE cutoff for {wildcards.scenario}."
        params:
            cycles=CYCLES,
            depth_category="all_depths",
            bin_id=f"cutoff_{CUTOFF_LABEL}",
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/regional_potential_overview.py"
