"""Input preparation and LCOE calculation rules."""


rule crop_to_speed_up:
    input:
        nc="<resources>/automatic/temperature_voxel.nc",
        shapes="<shapes>",
    output:
        clipped_temp_voxel="<resources>/automatic/{shapes}/generic/clipped_temp_voxel.nc",
    log:
        "<logs>/{shapes}/crop_to_speed_up.log",
    conda:
        "../envs/module.yaml"
    params:
        pad_m=25_000,
    message:
        "Crop the European temperature model to {wildcards.shapes}."
    script:
        "../scripts/crop_to_speed_up.py"


rule resample_gst1_sediment:
    """Align GST-1 sediment thickness to the full temperature grid."""
    input:
        archive="<resources>/automatic/GST1_WGS84.zip",
        temperature="<resources>/automatic/temperature_voxel.nc",
    output:
        sediment="<resources>/automatic/GST1_sediment_thickness.nc",
    log:
        "<logs>/automatic/resample_gst1_sediment.log",
    conda:
        "../envs/module.yaml"
    params:
        member=AUTOMATIC_DATASETS["GST1_WGS84.zip"]["member"],
        doi=AUTOMATIC_DATASETS["GST1_WGS84.zip"]["doi"],
    message:
        "Align GST-1 sediment thickness with the temperature grid."
    script:
        "../scripts/resample_gst1.py"


if POWER_ENABLED:

    rule calculate_LCOE:
        input:
            eligibility=ELIGIBILITY_INPUT,
            nc=rules.crop_to_speed_up.output.clipped_temp_voxel,
            sed=rules.resample_gst1_sediment.output.sediment,
            eff=workflow.source_path("../internal/efficiency_curves.csv"),
            drilling_csv=drilling_curve_inputs,
        output:
            cube="<resources>/automatic/{shapes}/documents/{cycle}_lcoe_cube.nc",
        log:
            "<logs>/{shapes}/{cycle}/calculate_LCOE.log",
        conda:
            "../envs/module.yaml"
        params:
            cycle=lambda wc: wc.cycle,
            max_depth=config["max_depth"],
            include_egs=config["include_egs"],
            EGS_cutoff=config["EGS_cutoff"],
            SED_NAN_POLICY=config["SED_NAN_POLICY"],
            drilling_cost_curve=config["drilling_cost_curve"],
            drilling_cost_curve_definitions=config["drilling_cost_curve_definitions"],
            stimulation_cost_per_doublet=config["stimulation_cost_per_doublet"],
            pump_cost=config["pump_cost"],
            lifetime=config["lifetime"],
            discount_rate=config["discount_rate"],
            capacity_factor=config["capacity_factor"],
            fixed_OM_rate=config["fixed_OM_rate"],
            parasitic_fraction=config["parasitic_fraction"],
            orc_money_per_kw=config["orc_money_per_kw"],
            flash_money_per_kw=config["flash_money_per_kw"],
            kalina_money_per_kw=config["kalina_money_per_kw"],
            well_spacing=config["well_spacing"],
            flowrate_egs=config["flowrate_egs"],
            flowrate_conventional=config["flowrate_conventional"],
            min_dt=config["min_dt"],
            temp_turbine_out=config["temp_turbine_out"],
        message:
            "Calculate {wildcards.cycle} geothermal potential and LCOE for {wildcards.shapes}."
        script:
            "../scripts/calculate_LCOE.py"
