"""Build every applicable figure and table for the native Snakemake report."""


rule prepare_report:
    input:
        overview=(rules.make_2d.output.overview_fig if MAKE_2D_FIGURE_OUTPUTS else []),
        cumulative=(
            rules.make_2d.output.technology_cumulative_four_panel_fig
            if MAKE_2D_FIGURE_OUTPUTS
            else []
        ),
        depth_cost=rules.depth_cost_breakdown_four_panel.output.fig,
        regional=(
            rules.regional_potential_overview.output.fig
            if CUTOFF_LABEL is not None
            else []
        ),
        summary=rules.overall_potential.output.summary,
    output:
        manifest="<report_inputs>",
    log:
        "<logs>/{shapes}/prepare_report.log",
    message:
        "Prepare figures and tables for the geothermal report for {wildcards.shapes}."
    shell:
        "printf '%s\n' {input:q} > {output.manifest:q} 2> {log:q}"
