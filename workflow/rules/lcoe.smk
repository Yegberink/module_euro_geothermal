"""Input preparation and LCOE calculation rules."""


rule crop_to_speed_up:
    input:
        nc="<resources>/automatic/temperature_voxel.nc",
        shapes="<shapes>",
    output:
        clipped_temp_voxel="<resources>/automatic/{shapes}/generic/clipped_temp_voxel.nc",
    log:
        "<logs>/{shapes}/crop_to_speed_up.log",
    message:
        "Crop the European temperature model to {wildcards.shapes}."
    params:
        pad_m=25_000,
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/crop_to_speed_up.py"


rule resample_gst1_sediment:
    """Align GST-1 sediment thickness to the full temperature grid."""
    input:
        archive=rules.download_gst1.output.archive,
        temperature="<resources>/automatic/temperature_voxel.nc",
    output:
        sediment="<resources>/automatic/GST1_sediment_thickness.nc",
    log:
        "<logs>/automatic/resample_gst1_sediment.log",
    message:
        "Align GST-1 sediment thickness with the temperature grid."
    params:
        member=internal["resources"]["automatic"]["gst1"]["member"],
        doi=internal["resources"]["automatic"]["gst1"]["doi"],
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/resample_gst1.py"


rule calculate_LCOE:
    input:
        nc=rules.crop_to_speed_up.output.clipped_temp_voxel,
        sed=rules.resample_gst1_sediment.output.sediment,
        eff=workflow.source_path("../internal/efficiency_curves.csv"),
        drilling_costs_2017=workflow.source_path(
            "../internal/baseline_drilling_2017.csv"
        ),
        drilling_costs_2025=workflow.source_path(
            "../internal/baseline_drilling_2025.csv"
        ),
    output:
        cube="<resources>/automatic/{shapes}/documents/{cycle}_lcoe_cube.nc",
    log:
        "<logs>/{shapes}/{cycle}/calculate_LCOE.log",
    message:
        "Calculate {wildcards.cycle} geothermal potential and LCOE for {wildcards.shapes}."
    params:
        cycle=lambda wc: wc.cycle,
        model=model_parameters(config),
    conda:
        "../envs/module.yaml"
    script:
        "../scripts/calculate_LCOE.py"
