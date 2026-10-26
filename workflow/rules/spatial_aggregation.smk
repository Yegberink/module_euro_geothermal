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
            lcoe_tifs=expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_lcoe.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            ),
            capex_tifs=expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_specific_capex.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            ),
            pot_tifs=expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_2d_potential.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            ),
        log:
            "<logs>/{shapes}/make_2d.log",
        conda:
            "../envs/module.yaml"
        params:
            cycles=CYCLES,
            max_depth=config["max_depth"],
        message:
            "Reduce geothermal results for {wildcards.shapes} to 2-D rasters."
        script:
            "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for the configured parameters."""
    input:
        lcoe_tifs=(rules.make_2d.output.lcoe_tifs if POWER_ENABLED else []),
        pot_tifs=(rules.make_2d.output.pot_tifs if POWER_ENABLED else []),
        capex_tifs=(rules.make_2d.output.capex_tifs if POWER_ENABLED else []),
        heat_capex=(
            [rules.calculate_LCOH.output.specific_capex] if HEAT_ENABLED else []
        ),
        heat_fraction=([rules.allocate_heat.output.fraction] if HEAT_ENABLED else []),
        heat_allocated=([rules.allocate_heat.output.allocated] if HEAT_ENABLED else []),
        heat_lcoh=([rules.calculate_LCOH.output.lcoh] if HEAT_ENABLED else []),
        heat_allocations=(
            rules.allocate_heat.output.allocations if HEAT_ENABLED else []
        ),
        shapes="<shapes>",
    output:
        geoparquet="<geothermal_regions>",
        power=expand(
            "<resources>/automatic/{shapes}/documents/{cycle}_power_potential.tif",
            shapes="{shapes}",
            cycle=CYCLES,
        ),
        contested_power=(
            expand(
                "<resources>/automatic/{shapes}/documents/{cycle}_contested_power_potential.tif",
                shapes="{shapes}",
                cycle=CYCLES,
            )
            if HEAT_ENABLED
            else []
        ),
        heat=(
            ["<resources>/automatic/{shapes}/documents/heat_allocated_potential.tif"]
            if HEAT_ENABLED
            else []
        ),
        contested_heat=(
            ["<resources>/automatic/{shapes}/documents/contested_heat_potential.tif"]
            if HEAT_ENABLED
            else []
        ),
        regional_fig=(
            report(
                "<results>/{shapes}/figures/regional_power_potential.pdf",
                category="power",
                subcategory="{shapes}",
                labels={"Figure": "Regional electricity potential"},
            )
            if POWER_ENABLED
            else []
        ),
        heat_potential_fig=(
            report(
                "<results>/{shapes}/figures/heat_potential_overview.pdf",
                category="heat",
                subcategory="{shapes}",
                labels={"Figure": "Heat supplied to networks"},
            )
            if HEAT_ENABLED
            else []
        ),
        heat_lcoh_fig=(
            report(
                "<results>/{shapes}/figures/lcoh_overview.pdf",
                category="heat",
                subcategory="{shapes}",
                labels={"Figure": "LCOH raster"},
            )
            if HEAT_ENABLED
            else []
        ),
        regional_heat_fig=(
            report(
                "<results>/{shapes}/figures/regional_heat_potential.pdf",
                category="heat",
                subcategory="{shapes}",
                labels={"Figure": "Regional heating potential"},
            )
            if HEAT_ENABLED
            else []
        ),
    log:
        "<logs>/{shapes}/aggregate_to_regions.log",
    conda:
        "../envs/module.yaml"
    params:
        carrier_out=config["carrier_out"],
        max_depth=config["max_depth"],
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
    message:
        "Aggregate geothermal results to {wildcards.shapes}."
    script:
        "../scripts/aggregate_to_regions.py"
