"""Write the scenario TSV used by downstream Snakemake rules."""

from scenario_config import build_scenario_rows, write_scenario_table

if "snakemake" in dir():
    p = snakemake.params  # noqa: F821
    cfg = {
        **p.base_config,
        "baseline_scenario": p.baseline_scenario,
        "active_sensitivity_parameters": p.active_sensitivity_parameters,
        "sensitivity_parameters": p.sensitivity_parameters,
    }
    if p.scenarios is not None:
        cfg["scenarios"] = p.scenarios

    rows = build_scenario_rows(cfg)
    write_scenario_table(rows, snakemake.output.tsv)  # noqa: F821
