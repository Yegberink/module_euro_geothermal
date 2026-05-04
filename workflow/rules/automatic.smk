"""Rules for automatic geothermal resource processing."""


rule build_scenario_table:
    """Generate the scenario TSV from config sensitivity settings."""
    output:
        tsv=SCENARIO_TABLE,
    params:
        baseline_scenario=BASELINE_SCENARIO,
        active_sensitivity_parameters=config.get("active_sensitivity_parameters", []),
        sensitivity_parameters=config.get("sensitivity_parameters", {}),
        scenarios=config.get("scenarios", None),
        base_config={k: v for k, v in config.items() if k not in NON_MODEL_KEYS},
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/build_scenario_table.py"


rule build_temperature_voxel:
    input:
        asc=expand(
            TEMPERATURE_GRID_PATTERN,
            depth=sorted(glob_wildcards(TEMPERATURE_GRID_PATTERN).depth),
        ),
    output:
        nc=temp("<temp_voxel>"),
        qc_html="<results>/{shapes}/figures/temp_voxel_depth_slider.html",
    params:
        z_min=0,
        z_max=10000,
        z_step=250,
        vert_interp_kind="linear",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/create_temp_voxel.py"


rule crop_to_speed_up:
    input:
        nc="<temp_voxel>",
        shapes="<shapes>",
    params:
        pad_m=25_000,
    output:
        clipped_temp_voxel="<clipped_temp_voxel>",
        qc_html="<results>/{shapes}/figures/clipped_temp_voxel_depth_slider.html",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/crop_to_speed_up.py"


rule calculate_LCOE:
    input:
        nc="<clipped_temp_voxel>",
        sed="<sediment_thickness>",
        eff="<efficiency_curves>",
        drilling_costs_2017="<baseline_drilling_2017>",
        drilling_costs_2025="<baseline_drilling_2025>",
        scenario_table=SCENARIO_TABLE,
    output:
        **(
            {
                "qc_lcoe": "<results>/{shapes}/scenarios/{scenario}/geothermal_lcoe/figures/lcoe_{cycle}_depth_slider.html",
                "qc_potential": "<results>/{shapes}/scenarios/{scenario}/geothermal_lcoe/figures/potential_{cycle}_depth_slider.html",
            }
            if WRITE_SLIDERS
            else {}
        ),
        cube="<lcoe_cube>",
    params:
        cycle=lambda wc: wc.cycle,
        write_qc_sliders=WRITE_SLIDERS,
        carrier_out=lambda wc: scen_str(wc, "carrier_out", config["carrier_out"]),
        egs_cutoff=lambda wc: scen_float(wc, "EGS_cutoff", config["EGS_cutoff"]),
        sed_nan_policy=lambda wc: scen_str(
            wc, "SED_NAN_POLICY", config["SED_NAN_POLICY"]
        ),
        include_egs=lambda wc: scen_bool(wc, "include_egs", config["include_egs"]),
        temp_turbine_out=lambda wc: scen_temp_turbine_out(wc),
        drilling_cost_curve=lambda wc: scen_str(
            wc, "drilling_cost_curve", config.get("drilling_cost_curve", "thermogis")
        ),
        drilling_cost_curve_definitions=lambda wc, input: drilling_cost_curve_definitions(
            input
        ),
        max_depth=lambda wc: scen_float(wc, "max_depth", config["max_depth"]),
        stimulation_cost_per_doublet=lambda wc: scen_float(
            wc,
            "stimulation_cost_per_doublet",
            config.get("stimulation_cost_per_doublet"),
        ),
        pump_cost=lambda wc: scen_float(wc, "pump_cost", config["pump_cost"]),
        lifetime=lambda wc: scen_float(wc, "lifetime", config["lifetime"]),
        capacity_factor=lambda wc: scen_float(
            wc, "capacity_factor", config["capacity_factor"]
        ),
        fixed_OM_rate=lambda wc: scen_float(
            wc, "fixed_OM_rate", config["fixed_OM_rate"]
        ),
        parasitic_fraction=lambda wc: scen_float(
            wc, "parasitic_fraction", config["parasitic_fraction"]
        ),
        well_spacing=lambda wc: scen_float(wc, "well_spacing", config["well_spacing"]),
        flowrate_egs=lambda wc: scen_float(wc, "flowrate_egs", config["flowrate_egs"]),
        flowrate_conventional=lambda wc: scen_float(
            wc, "flowrate_conventional", config["flowrate_conventional"]
        ),
        orc_money_per_kw=lambda wc: scen_float(
            wc, "orc_money_per_kw", config["orc_money_per_kw"]
        ),
        flash_money_per_kw=lambda wc: scen_float(
            wc, "flash_money_per_kw", config["flash_money_per_kw"]
        ),
        kalina_money_per_kw=lambda wc: scen_float(
            wc, "kalina_money_per_kw", config["kalina_money_per_kw"]
        ),
        min_dt=lambda wc: scen_float(wc, "min_dt", config["min_dt"]),
        qc_stride=config.get("qc_stride", 1),
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/calculate_LCOE.py"


rule make_2d_base:
    input:
        cubes=lambda wc: expand(
            LCOE_CUBE_PATTERN,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        **(
            {
                "overview_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_overview.png",
                "scatter_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/lcoe_vs_potential_scatter.png",
                "avg_potential_depth_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/average_potential_per_cell_by_depth.png",
                "cumulative_curve_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_lcoe.png",
                "cumulative_depth_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_depth.png",
                "cumulative_temperature_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/cumulative_potential_vs_temperature.png",
                "temperature_lcoe_potential_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_temperature_lcoe_heatmap.png",
                "temperature_lcoe_scatter_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_vs_lcoe_scatter.png",
                "temperature_lcoe_isopotential_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_isopotential_contours.png",
                "temperature_lcoe_potential_scatter_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/temperature_lcoe_potential_scatter.png",
                "potential_overview_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/potential_overview.png",
                "depth_overview_fig": "<results>/{shapes}/scenarios/{scenario}/2d_potential_cost/figures/depth_at_min_overview.png",
            }
            if METHOD_2D == "LCOE_minimization"
            else {}
        ),
        lcoe_tifs=(
            _cycle_pathvar_outputs("<lcoe_2d>")
            if METHOD_2D == "LCOE_minimization"
            else _depth_category_pathvar_outputs("<lcoe_2d_depth_category>")
        ),
        pot_tifs=(
            _cycle_pathvar_outputs("<potential_2d>")
            if METHOD_2D == "LCOE_minimization"
            else _depth_category_pathvar_outputs("<potential_2d_depth_category>")
        ),
    params:
        method_2d_aggregation=METHOD_2D,
        depth_categories=config.get("depth_categories"),
        depthcat_labels=DEPTHCAT_LABELS,
        cycles=CYCLES,
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario=BASELINE_SCENARIO,
    script:
        "../scripts/make_2d.py"


rule make_2d_scenario:
    input:
        cubes=lambda wc: expand(
            LCOE_CUBE_PATTERN,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        lcoe_tifs=(
            _cycle_pathvar_outputs("<lcoe_2d>")
            if METHOD_2D == "LCOE_minimization"
            else _depth_category_pathvar_outputs("<lcoe_2d_depth_category>")
        ),
        pot_tifs=(
            _cycle_pathvar_outputs("<potential_2d>")
            if METHOD_2D == "LCOE_minimization"
            else _depth_category_pathvar_outputs("<potential_2d_depth_category>")
        ),
    params:
        method_2d_aggregation=METHOD_2D,
        depth_categories=config.get("depth_categories"),
        depthcat_labels=DEPTHCAT_LABELS,
        cycles=CYCLES,
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario="(?!base$).+",
    script:
        "../scripts/make_2d.py"


rule aggregate_to_regions:
    """Aggregate 2D LCOE and potential rasters to user-defined regions for each scenario."""
    input:
        lcoe_tifs=lambda wc: _2d_lcoe_tifs(wc.shapes, wc.scenario),
        pot_tifs=lambda wc: _2d_pot_tifs(wc.shapes, wc.scenario),
        shapes="<shapes>",
    output:
        geoparquet="<geothermal_regions>",
        figdir=directory(
            "<results>/{shapes}/scenarios/{scenario}/region_aggregation/figures"
        ),
    params:
        cycles=CYCLES,
        method_region_aggregation=METHOD_REGION,
        n_quantiles=N_QUANTILES,
        fixed_bins=FIXED_BINS,
        lcoe_cutoff=LCOE_CUTOFF,
    conda:
        "../envs/module_geothermal.yaml"
    wildcard_constraints:
        scenario=BASELINE_SCENARIO,
    script:
        "../scripts/aggregate_to_regions.py"


rule scenario_summary_fig:
    input:
        base_lcoe=lambda wc: expand(
            LCOE_2D_PATTERN,
            shapes=wc.shapes,
            baseline=BASELINE_SCENARIO,
            scenario=BASELINE_SCENARIO,
            cycle=CYCLES,
        ),
        scen_lcoe=lambda wc: expand(
            LCOE_2D_PATTERN,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
        base_pot=lambda wc: expand(
            POTENTIAL_2D_PATTERN,
            shapes=wc.shapes,
            baseline=BASELINE_SCENARIO,
            scenario=BASELINE_SCENARIO,
            cycle=CYCLES,
        ),
        scen_pot=lambda wc: expand(
            POTENTIAL_2D_PATTERN,
            shapes=wc.shapes,
            scenario=wc.scenario,
            cycle=CYCLES,
        ),
    output:
        fig="<results>/{shapes}/scenarios/{scenario}/sensitivity/summary.png",
        metrics="<results>/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv",
        boxplot_data="<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
    params:
        baseline=BASELINE_SCENARIO,
        cycles=CYCLES,
        sens_tech=config.get("sens_tech", "orc"),
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/sensitivity_summary.py"


rule sensitivity_boxplot_per_scenario:
    input:
        scenario_table=SCENARIO_TABLE,
        boxplot_data="<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
    output:
        base_fig="<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_base_dual_axis.png",
        lcoe_fig="<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_lcoe.png",
        potential_fig="<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_potential.png",
    conda:
        "../envs/module_geothermal.yaml"
    script:
        "../scripts/sensitivity_boxplot_per_scenario.py"


if PERFORM_SENSITIVITY:

    rule sensitivity_vis:
        input:
            scenario_table=SCENARIO_TABLE,
            metrics=lambda wc: expand(
                "<results>/{shapes}/scenarios/{scenario}/sensitivity/metrics.tsv",
                shapes=wc.shapes,
                scenario=SCENARIOS_NO_BASE,
            ),
            boxplot_data=lambda wc: expand(
                "<results>/{shapes}/scenarios/{scenario}/sensitivity/boxplot_data.parquet",
                shapes=wc.shapes,
                scenario=SCENARIOS,
            ),
        output:
            boxplot_lcoe_top6_winner="<results>/{shapes}/scenarios/sensitivity/boxplot_lcoe_top_6_winner.png",
            boxplot_potential_top6_winner="<results>/{shapes}/scenarios/sensitivity/boxplot_potential_top_6_winner.png",
            barplot_switch_rate_top6="<results>/{shapes}/scenarios/sensitivity/barplot_switch_rate_top_6.png",
        params:
            baseline=BASELINE_SCENARIO,
            metric_name="mean_abs_pct_change_sens",
            top_n=6,
            include_all=False,
        conda:
            "../envs/module_geothermal.yaml"
        script:
            "../scripts/sensitivity_vis.py"
