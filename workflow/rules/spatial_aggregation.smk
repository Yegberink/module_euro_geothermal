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
        conda:
            "../envs/module.yaml"
        params:
            method_2d_aggregation=config["method_2d_aggregation"],
            depth_categories=(
                config["depth_categories"]
                if config["method_2d_aggregation"] == "depth_categories"
                else []
            ),
            depthcat_labels=DEPTH_CATEGORY_LABELS,
            cycles=CYCLES,
            max_depth=config["max_depth"],
        message:
            "Reduce geothermal results for {wildcards.shapes} to 2-D rasters."
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
    output:
        geoparquet="<geothermal_regions>",
        manifest="<results>/{shapes}/geothermal_region_outputs.tsv",
        regional_fig=(
            report(
                f"<results>/{{shapes}}/figures/"
                f"potential_overview_all_depths_cutoff_{CUTOFF_LABEL}.png",
                category="Regional potential",
                subcategory="{shapes}",
                labels={"Figure": "Regional electricity potential"},
            )
            if CUTOFF_LABEL is not None
            else []
        ),
    log:
        "<logs>/{shapes}/aggregate_to_regions.log",
    conda:
        "../envs/module.yaml"
    params:
        carrier_out=config["carrier_out"],
        cycles=CYCLES,
        method_region_aggregation=(
            config["method_region_aggregation"] if POWER_ENABLED else None
        ),
        n_quantiles=(
            config["n_quantiles"]
            if POWER_ENABLED and config["method_region_aggregation"] == "quantiles"
            else None
        ),
        fixed_bins=(
            config["fixed_bins"]
            if POWER_ENABLED and config["method_region_aggregation"] == "fixed_bins"
            else []
        ),
        lcoe_cutoff=(
            config["lcoe_cutoff"]
            if POWER_ENABLED and config["method_region_aggregation"] == "cutoff"
            else None
        ),
        bin_id=f"cutoff_{CUTOFF_LABEL}",
    message:
        "Aggregate geothermal results to {wildcards.shapes}."
    script:
        "../scripts/aggregate_to_regions.py"
