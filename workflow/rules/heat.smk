"""Independent thermal economics and joint district-heating allocation."""

if HEAT_ENABLED:

    rule prepare_heat_extent:
        input:
            nc="<resources>/automatic/temperature_voxel.nc",
            shapes="<shapes>",
            networks="<district_heating>",
        output:
            extent="<resources>/automatic/{shapes}/generic/heat_extent.json",
        log:
            "<logs>/{shapes}/prepare_heat_extent.log",
        conda:
            "../envs/module.yaml"
        params:
            buffer_km=config["heat_network_buffer_km"],
        script:
            "../scripts/prepare_heat_extent.py"

    rule calculate_LCOH:
        input:
            eligibility=ELIGIBILITY_INPUT,
            nc="<resources>/automatic/temperature_voxel.nc",
            extent=rules.prepare_heat_extent.output.extent,
            sed=rules.resample_gst1_sediment.output.sediment,
            drilling_csv=DRILLING_CSV,
        output:
            specific_capex="<resources>/automatic/{shapes}/{scenario}/documents/heat_specific_capex.tif",
            lcoh="<resources>/automatic/{shapes}/{scenario}/documents/heat_lcoh.tif",
            potential="<resources>/automatic/{shapes}/{scenario}/documents/heat_potential.tif",
            depth="<resources>/automatic/{shapes}/{scenario}/documents/heat_depth.tif",
        log:
            "<logs>/{shapes}/{scenario}/calculate_LCOH.log",
        conda:
            "../envs/module.yaml"
        params:
            max_depth=lambda wc: config["scenario"][wc.scenario]["max_depth"],
            include_egs=config["include_egs"],
            EGS_cutoff=lambda wc: config["scenario"][wc.scenario]["EGS_cutoff"],
            drilling_cost_curve=config["drilling_cost_curve"],
            stimulation_cost_per_doublet=lambda wc: config["scenario"][wc.scenario][
                "stimulation_cost_per_doublet"
            ],
            pump_cost=lambda wc: config["scenario"][wc.scenario]["pump_cost"],
            lifetime=lambda wc: config["scenario"][wc.scenario]["lifetime"],
            discount_rate=config["discount_rate"],
            fixed_OM_rate=lambda wc: config["scenario"][wc.scenario]["fixed_OM_rate"],
            well_spacing=lambda wc: config["scenario"][wc.scenario]["well_spacing"],
            flowrate_egs=lambda wc: config["scenario"][wc.scenario]["flowrate_egs"],
            flowrate_conventional=lambda wc: config["scenario"][wc.scenario][
                "flowrate_conventional"
            ],
            heat_production_temperature=config["heat_production_temperature"],
            heat_reinjection_temperature=config["heat_reinjection_temperature"],
            thermal_plant_cost_per_kw=config["thermal_plant_cost_per_kw"],
            heat_capacity_factor=config["heat_capacity_factor"],
        message:
            "Calculate geothermal LCOH per cell for {wildcards.shapes} ({wildcards.scenario})."
        script:
            "../scripts/calculate_LCOH.py"

    rule allocate_heat:
        input:
            lcoh=rules.calculate_LCOH.output.lcoh,
            potential=rules.calculate_LCOH.output.potential,
            networks="<district_heating>",
            shapes="<shapes>",
        output:
            allocations=[
                "<resources>/automatic/{shapes}/{scenario}/documents/heat_allocations.parquet"
            ],
            allocated="<resources>/automatic/{shapes}/{scenario}/documents/heat_allocated.tif",
            fraction="<resources>/automatic/{shapes}/{scenario}/documents/heat_used_fraction.tif",
        log:
            "<logs>/{shapes}/{scenario}/allocate_heat.log",
        conda:
            "../envs/module.yaml"
        params:
            buffer_km=config["heat_network_buffer_km"],
            allow_cross_region_supply=config["heat_allow_cross_region_supply"],
        script:
            "../scripts/allocate_heat.py"
