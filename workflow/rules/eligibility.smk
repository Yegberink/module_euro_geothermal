"""Align supplied area potential to the shared heat and power grid."""

if ELIGIBILITY_ENABLED:

    rule resample_eligibility:
        input:
            nc="<resources>/automatic/temperature_voxel.nc",
            power=(
                ["<resources>/automatic/{shapes}/generic/clipped_temp_voxel.nc"]
                if POWER_ENABLED
                else []
            ),
            heat=(
                ["<resources>/automatic/{shapes}/generic/heat_extent.json"]
                if HEAT_ENABLED
                else []
            ),
            area_potential="<area_potential>",
        output:
            fraction="<resources>/automatic/{shapes}/eligibility/eligibility_fraction.tif",
        log:
            "<logs>/{shapes}/resample_eligibility.log",
        benchmark:
            "<logs>/{shapes}/resample_eligibility.benchmark.tsv"
        conda:
            "../envs/module.yaml"
        script:
            "../scripts/resample_eligibility.py"
