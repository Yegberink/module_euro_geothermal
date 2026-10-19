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
            lcoh="<resources>/automatic/{shapes}/documents/heat_lcoh.tif",
            potential="<resources>/automatic/{shapes}/documents/heat_potential.tif",
            depth="<resources>/automatic/{shapes}/documents/heat_depth.tif",
        log:
            "<logs>/{shapes}/calculate_LCOH.log",
        conda:
            "../envs/module.yaml"
        params:
            max_depth=config["max_depth"],
            include_egs=config["include_egs"],
            EGS_cutoff=config["EGS_cutoff"],
            drilling_cost_curve=config["drilling_cost_curve"],
            stimulation_cost_per_doublet=config["stimulation_cost_per_doublet"],
            pump_cost=config["pump_cost"],
            lifetime=config["lifetime"],
            discount_rate=config["discount_rate"],
            fixed_OM_rate=config["fixed_OM_rate"],
            well_spacing=config["well_spacing"],
            flowrate_egs=config["flowrate_egs"],
            flowrate_conventional=config["flowrate_conventional"],
            heat_production_temperature=config["heat_production_temperature"],
            heat_reinjection_temperature=config["heat_reinjection_temperature"],
            thermal_plant_cost_per_kw=config["thermal_plant_cost_per_kw"],
            heat_capacity_factor=config["heat_capacity_factor"],
        message:
            "Calculate geothermal LCOH once per cell for {wildcards.shapes}."
        script:
            "../scripts/calculate_LCOH.py"

    rule allocate_heat:
        input:
            lcoh=rules.calculate_LCOH.output.lcoh,
            potential=rules.calculate_LCOH.output.potential,
            networks="<district_heating>",
            shapes="<shapes>",
        output:
            networks="<geothermal_heat_networks>",
            allocations="<geothermal_heat_allocations>",
            allocated="<resources>/automatic/{shapes}/documents/heat_allocated.tif",
            fraction="<resources>/automatic/{shapes}/documents/heat_used_fraction.tif",
        log:
            "<logs>/{shapes}/allocate_heat.log",
        conda:
            "../envs/module.yaml"
        params:
            buffer_km=config["heat_network_buffer_km"],
        script:
            "../scripts/allocate_heat.py"
