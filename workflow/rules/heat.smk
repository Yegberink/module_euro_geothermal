"""Independent thermal economics and joint district-heating allocation."""

if HEAT_ENABLED:

    rule prepare_heat_extent:
        input:
            nc="<resources>/automatic/temperature_voxel.nc",
            shapes="<shapes>",
            networks="<district_heating>",
        output:
            extent="<resources>/automatic/{shapes}/generic/heat_extent.json",
        params:
            buffer_km=config["heat_network_buffer_km"],
        log:
            "<logs>/{shapes}/prepare_heat_extent.log",
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/prepare_heat_extent.py"

    rule calculate_LCOH:
        input:
            nc="<resources>/automatic/temperature_voxel.nc",
            extent=rules.prepare_heat_extent.output.extent,
            sed=rules.resample_gst1_sediment.output.sediment,
            drilling_costs_2017=workflow.source_path(
                "../internal/baseline_drilling_2017.csv"
            ),
            drilling_costs_2025=workflow.source_path(
                "../internal/baseline_drilling_2025.csv"
            ),
        output:
            lcoh="<resources>/automatic/{shapes}/documents/heat_lcoh.tif",
            potential="<resources>/automatic/{shapes}/documents/heat_potential.tif",
            depth="<resources>/automatic/{shapes}/documents/heat_depth.tif",
        params:
            model=heat_model_parameters(config),
        log:
            "<logs>/{shapes}/calculate_LCOH.log",
        message:
            "Calculate geothermal LCOH once per cell for {wildcards.shapes}."
        conda:
            "../envs/module.yaml"
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
        params:
            buffer_km=config["heat_network_buffer_km"],
        log:
            "<logs>/{shapes}/allocate_heat.log",
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/allocate_heat.py"
