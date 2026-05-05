"""Write the scenario TSV used by downstream Snakemake rules."""

from scenario_config import build_scenario_rows, write_scenario_table

if "snakemake" in dir():
    rows = build_scenario_rows(snakemake.params.config)  # noqa: F821
    write_scenario_table(rows, snakemake.output.tsv)  # noqa: F821
