"""Rules for reducing LCOE cubes to 2D rasters and regional outputs."""

if POWER_ENABLED:

    rule make_2d:
        input:
            shapes="<shapes>",
            cubes=lambda wc: expand(
                rules.calculate_LCOE.output.cube,
                shapes=wc.shapes,
                cycle=CYCLES,
            ),
        output:
            **MAKE_2D_FIGURE_OUTPUTS,
            lcoe_tifs=(
                expand(
                    "<resources>/automatic/{shapes}/documents/{cycle}_2d_lcoe.tif",
                    shapes="{shapes}",
                    cycle=CYCLES,
                )
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else expand(
                    "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_lcoe.tif",
                    shapes="{shapes}",
                    cycle=CYCLES,
                    cat=DEPTH_CATEGORY_LABELS,
                )
            ),
            pot_tifs=(
                expand(
                    "<resources>/automatic/{shapes}/documents/{cycle}_2d_potential.tif",
                    shapes="{shapes}",
                    cycle=CYCLES,
                )
                if config["method_2d_aggregation"] == "LCOE_minimization"
                else expand(
                    "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_potential.tif",
                    shapes="{shapes}",
                    cycle=CYCLES,
                    cat=DEPTH_CATEGORY_LABELS,
                )
            ),
        log:
            "<logs>/{shapes}/make_2d.log",
        message:
            "Reduce geothermal results for {wildcards.shapes} to 2-D rasters."
        params:
            method_2d_aggregation=config["method_2d_aggregation"],
            depth_categories=config.get("depth_categories"),
            depthcat_labels=DEPTH_CATEGORY_LABELS,
            cycles=CYCLES,
            max_depth=config["max_depth"],
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for the configured parameters."""
    input:
        lcoe_tifs=(
            expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            )
            if POWER_ENABLED and config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_lcoe.tif",
                shapes="{shapes}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        pot_tifs=(
            expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            )
            if POWER_ENABLED and config["method_2d_aggregation"] == "LCOE_minimization"
            else expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_{cat}_potential.tif",
                shapes="{shapes}",
                cycle=CYCLES,
                cat=DEPTH_CATEGORY_LABELS,
            )
        ),
        heat_fraction=([rules.allocate_heat.output.fraction] if HEAT_ENABLED else []),
        heat_allocations=(
            [rules.allocate_heat.output.allocations] if HEAT_ENABLED else []
        ),
        shapes="<shapes>",
        # Include standalone diagnostics in the normal results DAG so --report
        # collects the complete figure bundle without a separate report target.
        diagnostics=lambda wc: (
            expand(rules.depth_cost_breakdown_four_panel.output.fig, shapes=wc.shapes)
            if POWER_ENABLED
            else []
        ),
        summary=(["<overall_potential_summary>"] if POWER_ENABLED else []),
    output:
        geoparquet="<geothermal_regions>",
        manifest="<results>/{shapes}/geothermal_region_outputs.tsv",
        regional_fig=(
            report(
                f"<results>/{{shapes}}/figures/"
                f"potential_overview_all_depths_cutoff_{CUTOFF_LABEL}.png",
                caption="../report/regional.rst",
                category="Regional potential",
                subcategory="{shapes}",
                labels={"Figure": "Regional electricity potential"},
            )
            if CUTOFF_LABEL is not None
            else []
        ),
    log:
        "<logs>/{shapes}/aggregate_to_regions.log",
    message:
        "Aggregate geothermal results to {wildcards.shapes}."
    params:
        carrier_out=config["carrier_out"],
        cycles=CYCLES,
        method_region_aggregation=config.get("method_region_aggregation"),
        n_quantiles=config.get("n_quantiles"),
        fixed_bins=config.get("fixed_bins", []),
        lcoe_cutoff=config.get("lcoe_cutoff"),
        bin_id=f"cutoff_{CUTOFF_LABEL}",
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/aggregate_to_regions.py"
