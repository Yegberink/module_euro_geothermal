"""Scenario expansion helpers for the geothermal Snakemake module."""

from __future__ import annotations

import csv
import math
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
    "target_shapes",
    "perform_sensitivity",
    "baseline_scenario",
    "scenario_table",
    "scenarios",
    "active_sensitivity_parameters",
    "sensitivity_parameters",
    "write_qc_sliders",
    "method_2d_aggregation",
    "method_region_aggregation",
    "depth_categories",
    "n_quantiles",
    "fixed_bins",
    "lcoe_cutoff",
    "qc_stride",
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
        if key == "temp_turbine_out" and isinstance(value, dict):
            for cycle, temp in value.items():
                flat[f"temp_turbine_out_{cycle}"] = float(temp)
        else:
            flat[key] = value
    return flat


def build_scenario_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build scenario rows from explicit scenarios or sensitivity settings."""
    baseline = str(cfg.get("baseline_scenario", "base"))
    base_row = flatten_base_config(model_config(cfg))

    if cfg.get("scenarios") is not None:
        return _build_explicit_scenarios(
            base_row=base_row,
            scenarios=cfg["scenarios"],
            baseline_scenario=baseline,
        )

    return _build_sensitivity_scenarios(
        base_row=base_row,
        active_parameters=cfg.get("active_sensitivity_parameters", []),
        parameter_definitions=cfg.get("sensitivity_parameters", {}),
        baseline_scenario=baseline,
    )


def scenario_mapping(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return scenario rows keyed by scenario name."""
    return {str(row["scenario"]): {k: _none_if_nan(v) for k, v in row.items()} for row in rows}


def write_scenario_table(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write scenario rows as a tab-separated table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    rows_by_name = {
        baseline_scenario: {
            "scenario": baseline_scenario,
            **base_row,
            **_metadata(None),
        }
    }

    for name, overrides in _iter_explicit_scenarios(scenarios):
        row = {**base_row, **overrides}
        row["scenario"] = name
        for key in METADATA_KEYS:
            row.setdefault(key, None)
        if name != baseline_scenario and not row.get("sensitivity_parameter"):
            row.update(_infer_metadata(base_row, overrides))
        rows_by_name[name] = row

    ordered = []
    if baseline_scenario in rows_by_name:
        ordered.append(rows_by_name.pop(baseline_scenario))
    ordered.extend(rows_by_name.values())
    return ordered


def _iter_explicit_scenarios(
    scenarios: list[dict[str, Any]] | dict[str, dict[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(scenarios, dict):
        return [(str(name), dict(values or {})) for name, values in scenarios.items()]

    items = []
    for scenario in scenarios:
        values = dict(scenario)
        name = values.pop("scenario", values.pop("name", None))
        if name is None:
            msg = "Each explicit scenario must define a 'scenario' or 'name' field."
            raise ValueError(msg)
        items.append((str(name), values))
    return items


def _infer_metadata(base_row: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    changed = [
        key for key, value in overrides.items()
        if key not in METADATA_KEYS and key != "scenario" and base_row.get(key) != value
    ]
    if len(changed) != 1:
        return _metadata(None)

    param = changed[0]
    base_value = base_row.get(param)
    scenario_value = overrides[param]
    pct_change = _pct_change(base_value, scenario_value)
    return _metadata(
        param=param,
        base_value=base_value,
        scenario_value=scenario_value,
        pct_change=pct_change,
        sort_magnitude=abs(pct_change) if pct_change is not None else 1.0,
    )


def _build_sensitivity_scenarios(
    base_row: dict[str, Any],
    active_parameters: list[str],
    parameter_definitions: dict[str, Any],
    baseline_scenario: str,
) -> list[dict[str, Any]]:
    rows = [{"scenario": baseline_scenario, **base_row, **_metadata(None)}]
    existing_names = {baseline_scenario}

    for param in active_parameters:
        definition = parameter_definitions[param]
        base_value = base_row.get(param)

        if isinstance(base_value, bool) or isinstance(definition, bool):
            base_bool = _parse_bool(base_row.get(param, definition))
            flipped = not base_bool
            updated_row = {**base_row, param: flipped}

            if param == "include_egs" and flipped is False:
                egs_cutoff = updated_row.get("EGS_cutoff")
                max_depth = updated_row.get("max_depth")
                updated_row["max_depth"] = min(float(max_depth), float(egs_cutoff))

            name = _make_unique(existing_names, f"{baseline_scenario}_{param}_{flipped}")
            rows.append(
                {
                    "scenario": name,
                    **updated_row,
                    **_metadata(
                        param=param,
                        base_value=base_bool,
                        scenario_value=flipped,
                        sort_magnitude=1.0,
                    ),
                }
            )
            existing_names.add(name)

        elif isinstance(definition, (list, tuple)):
            base_category = str(base_value)
            for target in [str(value) for value in definition]:
                if target == base_category:
                    continue
                name = _make_unique(existing_names, f"{baseline_scenario}_{param}_{target}")
                rows.append(
                    {
                        "scenario": name,
                        **base_row,
                        param: target,
                        **_metadata(
                            param=param,
                            base_value=base_category,
                            scenario_value=target,
                            sort_magnitude=1.0,
                        ),
                    }
                )
                existing_names.add(name)

        else:
            rows.extend(
                _numeric_scenarios(
                    base_row=base_row,
                    param=param,
                    definition=definition,
                    baseline_scenario=baseline_scenario,
                    existing_names=existing_names,
                )
            )

    return rows


def _numeric_scenarios(
    base_row: dict[str, Any],
    param: str,
    definition: dict[str, Any],
    baseline_scenario: str,
    existing_names: set[str],
) -> list[dict[str, Any]]:
    base_value = float(base_row.get(param))
    candidates = []

    if ("max_dev_pct" in definition) or ("steps" in definition):
        if base_value != 0.0:
            max_dev_pct = float(definition["max_dev_pct"])
            steps = int(definition["steps"])
            for delta in _build_deltas(max_dev_pct, steps):
                candidates.append((base_value * (1.0 + delta), delta * 100.0))

    for raw_value in definition.get("absolute_values", []):
        new_value = float(raw_value)
        candidates.append((new_value, _pct_change(base_value, new_value)))

    rows = []
    seen_values = []
    for new_value, pct_change in candidates:
        if abs(new_value - base_value) < 1e-12:
            continue
        if any(abs(new_value - value) < 1e-12 for value in seen_values):
            continue
        seen_values.append(new_value)

        tag = (
            _format_delta_tag(pct_change / 100.0)
            if pct_change is not None
            else f"v{_format_scalar(new_value).replace('.', '_')}"
        )
        name = _make_unique(existing_names, f"{baseline_scenario}_{param}{tag}")
        rows.append(
            {
                "scenario": name,
                **base_row,
                param: new_value,
                **_metadata(
                    param=param,
                    base_value=base_value,
                    scenario_value=new_value,
                    pct_change=pct_change,
                    sort_magnitude=(
                        abs(pct_change) if pct_change is not None else abs(new_value - base_value)
                    ),
                ),
            }
        )
        existing_names.add(name)
    return rows


def _metadata(
    param: str | None,
    base_value: Any = None,
    scenario_value: Any = None,
    pct_change: float | None = None,
    sort_magnitude: float | None = None,
) -> dict[str, Any]:
    if param is None:
        return {
            "sensitivity_parameter": None,
            "sensitivity_base_value": None,
            "sensitivity_scenario_value": None,
            "sensitivity_change_label": None,
            "sensitivity_pct_change": None,
            "sensitivity_sort_magnitude": None,
        }

    return {
        "sensitivity_parameter": param,
        "sensitivity_base_value": _format_scalar(base_value),
        "sensitivity_scenario_value": _format_scalar(scenario_value),
        "sensitivity_change_label": f"{_format_scalar(base_value)} -> {_format_scalar(scenario_value)}",
        "sensitivity_pct_change": pct_change,
        "sensitivity_sort_magnitude": sort_magnitude,
    }


def _build_deltas(max_dev_pct: float, steps: int) -> list[float]:
    pct_range = max_dev_pct / 100.0
    step = pct_range / steps
    neg = [-(i * step) for i in range(steps, 0, -1)]
    pos = [+(i * step) for i in range(1, steps + 1)]
    return neg + pos


def _format_delta_tag(delta: float) -> str:
    pct = delta * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_str = str(int(round(pct)))
    else:
        pct_str = f"{pct:.6g}".rstrip("0").rstrip(".").replace(".", "_")
    return f"m{pct_str}" if pct < 0 else f"p{pct_str}"


def _make_unique(existing: set[str], name: str) -> str:
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {
            "true": True,
            "t": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "f": False,
            "0": False,
            "no": False,
            "n": False,
        }[value.strip().lower()]
    return bool(value)


def _pct_change(base_value: Any, scenario_value: Any) -> float | None:
    try:
        base = float(base_value)
        scenario = float(scenario_value)
    except (TypeError, ValueError):
        return None
    if base == 0.0:
        return None
    return ((scenario / base) - 1.0) * 100.0


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


def _none_if_nan(value: Any) -> Any:
    if value in ("", "None", "none", "null", "NULL"):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
