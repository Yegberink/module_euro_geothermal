"""Scenario expansion helpers for the geothermal Snakemake module."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

METADATA_KEYS = (
    "sensitivity_parameter",
    "sensitivity_base_value",
    "sensitivity_scenario_value",
    "sensitivity_change_label",
    "sensitivity_pct_change",
    "sensitivity_sort_magnitude",
)

NON_MODEL_KEYS = {
    "perform_sensitivity",
    "baseline_scenario",
    "scenarios",
    "active_sensitivity_parameters",
    "sensitivity_parameters",
    "method_2d_aggregation",
    "method_region_aggregation",
    "depth_categories",
    "n_quantiles",
    "fixed_bins",
    "lcoe_cutoff",
    "sens_tech",
    "which_cycles",
    "drilling_cost_curve_definitions",
}


def model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return model parameters after removing workflow-control keys."""
    return {k: v for k, v in cfg.items() if k not in NON_MODEL_KEYS}


def flatten_base_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested model config values into one scenario-table row."""
    flat = {}
    for key, value in cfg.items():
        if key == "temp_turbine_out":
            for cycle, temp in value.items():
                flat[f"temp_turbine_out_{cycle}"] = temp
        else:
            flat[key] = value
    return flat


def build_scenario_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build scenario rows from explicit scenarios or sensitivity settings."""
    base_row = flatten_base_config(model_config(cfg))

    if "scenarios" in cfg:
        return _build_explicit_scenarios(
            base_row=base_row,
            scenarios=cfg["scenarios"],
            baseline_scenario=cfg["baseline_scenario"],
        )

    return _build_sensitivity_scenarios(
        base_row=base_row,
        active_parameters=cfg["active_sensitivity_parameters"],
        parameter_definitions=cfg["sensitivity_parameters"],
        baseline_scenario=cfg["baseline_scenario"],
    )


def write_scenario_table(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write scenario rows as a tab-separated table."""
    output_path = Path(output_path)
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_tsv_value(row.get(key)) for key in fieldnames})


def _build_explicit_scenarios(
    base_row: dict[str, Any],
    scenarios: list[dict[str, Any]] | dict[str, dict[str, Any]],
    baseline_scenario: str,
) -> list[dict[str, Any]]:
    rows = [{"scenario": baseline_scenario, **base_row, **_metadata(None)}]

    if isinstance(scenarios, dict):
        rows.extend(
            {
                "scenario": name,
                **base_row,
                **overrides,
                **_metadata(None),
            }
            for name, overrides in scenarios.items()
            if name != baseline_scenario
        )
        return rows

    rows.extend(
        {
            "scenario": scenario["scenario"],
            **base_row,
            **{k: v for k, v in scenario.items() if k != "scenario"},
            **_metadata(None),
        }
        for scenario in scenarios
        if scenario["scenario"] != baseline_scenario
    )
    return rows


def _build_sensitivity_scenarios(
    base_row: dict[str, Any],
    active_parameters: list[str],
    parameter_definitions: dict[str, Any],
    baseline_scenario: str,
) -> list[dict[str, Any]]:
    rows = [{"scenario": baseline_scenario, **base_row, **_metadata(None)}]

    for param in active_parameters:
        definition = parameter_definitions[param]
        base_value = base_row[param]

        if isinstance(definition, bool):
            flipped = not base_value
            updated_row = {**base_row, param: flipped}

            if param == "include_egs" and flipped is False:
                updated_row["max_depth"] = min(
                    updated_row["max_depth"],
                    updated_row["EGS_cutoff"],
                )

            rows.append(
                {
                    "scenario": f"{baseline_scenario}_{param}_{flipped}",
                    **updated_row,
                    **_metadata(
                        param=param,
                        base_value=base_value,
                        scenario_value=flipped,
                        sort_magnitude=1.0,
                    ),
                }
            )

        elif isinstance(definition, list):
            for target in definition:
                if target == base_value:
                    continue
                rows.append(
                    {
                        "scenario": f"{baseline_scenario}_{param}_{target}",
                        **base_row,
                        param: target,
                        **_metadata(
                            param=param,
                            base_value=base_value,
                            scenario_value=target,
                            sort_magnitude=1.0,
                        ),
                    }
                )

        else:
            rows.extend(
                _numeric_scenarios(
                    base_row=base_row,
                    param=param,
                    definition=definition,
                    baseline_scenario=baseline_scenario,
                )
            )

    return rows


def _numeric_scenarios(
    base_row: dict[str, Any],
    param: str,
    definition: dict[str, Any],
    baseline_scenario: str,
) -> list[dict[str, Any]]:
    base_value = float(base_row[param])
    candidates = []
    if "max_dev_pct" in definition:
        candidates.extend(
            (base_value * (1.0 + delta), delta * 100.0)
            for delta in _build_deltas(definition["max_dev_pct"], definition["steps"])
        )
    candidates.extend(
        (float(value), _pct_change(base_value, value))
        for value in definition.get("absolute_values", [])
    )

    return [
        {
            "scenario": f"{baseline_scenario}_{param}{_format_delta_tag(pct_change / 100.0)}",
            **base_row,
            param: new_value,
            **_metadata(
                param=param,
                base_value=base_value,
                scenario_value=new_value,
                pct_change=pct_change,
                sort_magnitude=abs(pct_change),
            ),
        }
        for new_value, pct_change in candidates
        if new_value != base_value
    ]


def _metadata(
    param: str | None,
    base_value: Any = None,
    scenario_value: Any = None,
    pct_change: float | None = None,
    sort_magnitude: float | None = None,
) -> dict[str, Any]:
    if param is None:
        return {key: None for key in METADATA_KEYS}

    return {
        "sensitivity_parameter": param,
        "sensitivity_base_value": _format_scalar(base_value),
        "sensitivity_scenario_value": _format_scalar(scenario_value),
        "sensitivity_change_label": f"{_format_scalar(base_value)} -> {_format_scalar(scenario_value)}",
        "sensitivity_pct_change": pct_change,
        "sensitivity_sort_magnitude": sort_magnitude,
    }


def _build_deltas(max_dev_pct: float, steps: int) -> list[float]:
    step = (max_dev_pct / 100.0) / steps
    return [-(i * step) for i in range(steps, 0, -1)] + [
        +(i * step) for i in range(1, steps + 1)
    ]


def _format_delta_tag(delta: float) -> str:
    pct = delta * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_str = str(int(round(pct)))
    else:
        pct_str = f"{pct:.6g}".rstrip("0").rstrip(".").replace(".", "_")
    return f"m{pct_str}" if pct < 0 else f"p{pct_str}"


def _pct_change(base_value: float, scenario_value: float) -> float:
    return ((float(scenario_value) / base_value) - 1.0) * 100.0


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names = ["scenario"]
    for row in rows:
        for key in row:
            if key != "scenario" and key not in names:
                names.append(key)
    return names


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"
    return str(value)


def _format_tsv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return value
