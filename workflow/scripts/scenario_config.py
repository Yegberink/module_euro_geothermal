"""Scenario expansion helpers for the geothermal Snakemake module."""

from __future__ import annotations

import csv
import re
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

SCENARIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SENSITIVITY_PARAMETERS = frozenset(
    {
        "max_depth",
        "EGS_cutoff",
        "drilling_cost_curve",
        "stimulation_cost_per_doublet",
        "pump_cost",
        "lifetime",
        "capacity_factor",
        "fixed_OM_rate",
        "parasitic_fraction",
        "orc_money_per_kw",
        "flash_money_per_kw",
        "kalina_money_per_kw",
        "temp_turbine_out_ORC",
        "temp_turbine_out_Flash",
        "temp_turbine_out_Kalina",
        "include_egs",
        "SED_NAN_POLICY",
        "min_dt",
        "well_spacing",
        "flowrate_egs",
        "flowrate_conventional",
    }
)


def validate_module_config(cfg: dict[str, Any]) -> None:
    """Validate configuration relationships not expressible in JSON Schema."""
    _validate_scenario_name(cfg["baseline_scenario"])

    if cfg["carrier_out"] != "power":
        raise ValueError("Only carrier_out='power' is supported.")

    curve = cfg["drilling_cost_curve"]
    if curve not in cfg["drilling_cost_curve_definitions"]:
        raise ValueError(f"drilling_cost_curve {curve!r} has no matching definition.")

    _validate_depth_categories(cfg.get("depth_categories", []))
    _validate_fixed_bins(cfg.get("fixed_bins", []))
    _validate_sensitivity_config(cfg)
    _validate_explicit_scenarios(cfg)

    scenario_names = [row["scenario"] for row in build_scenario_rows(cfg)]
    for scenario_name in scenario_names:
        _validate_scenario_name(scenario_name)
    if len(scenario_names) != len(set(scenario_names)):
        raise ValueError("Expanded scenario names must be unique.")


def _validate_scenario_name(name: object) -> None:
    """Require scenario names that are safe to embed in file paths."""
    if not isinstance(name, str) or not SCENARIO_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Invalid scenario name {name!r}; use letters, digits, '_', '-', or '.'."
        )


def _validate_depth_categories(categories: list[list[float]]) -> None:
    """Require ordered, non-overlapping depth intervals within the source model."""
    previous_stop = 0.0
    for category in categories:
        start, stop = map(float, category)
        if start < 0 or stop > 10_000 or start >= stop:
            raise ValueError(
                f"Invalid depth category {category!r}; expected 0 <= start < stop <= 10000."
            )
        if start < previous_stop:
            raise ValueError("depth_categories must be ordered and non-overlapping.")
        previous_stop = stop


def _validate_fixed_bins(bins: list[str]) -> None:
    """Require ordered, non-overlapping LCOE intervals."""
    previous_stop = 0.0
    for value in bins:
        start_text, stop_text = value.split("-", maxsplit=1)
        start, stop = float(start_text), float(stop_text)
        if start < 0 or start >= stop:
            raise ValueError(f"Invalid fixed LCOE bin {value!r}.")
        if start < previous_stop:
            raise ValueError("fixed_bins must be ordered and non-overlapping.")
        previous_stop = stop


def _validate_sensitivity_config(cfg: dict[str, Any]) -> None:
    """Validate sensitivity names, definitions, and candidate values."""
    active = cfg.get("active_sensitivity_parameters", [])
    definitions = cfg.get("sensitivity_parameters", {})
    unknown = (set(active) | set(definitions)) - SENSITIVITY_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported sensitivity parameters: {sorted(unknown)}")

    missing = set(active) - set(definitions)
    if missing:
        raise ValueError(f"Missing sensitivity definitions: {sorted(missing)}")

    base_row = flatten_base_config(model_config(cfg))
    for parameter in active:
        definition = definitions[parameter]
        base_value = base_row[parameter]
        if isinstance(base_value, bool):
            if not isinstance(definition, bool):
                raise ValueError(
                    f"Sensitivity {parameter!r} must use a boolean definition."
                )
        elif isinstance(base_value, str):
            if not isinstance(definition, list) or not definition:
                raise ValueError(
                    f"Sensitivity {parameter!r} must list alternative values."
                )
            for value in definition:
                _validate_parameter_value(parameter, value, cfg)
        else:
            if not isinstance(definition, dict):
                raise ValueError(
                    f"Sensitivity {parameter!r} must use a numeric definition."
                )
            if not ({"max_dev_pct", "absolute_values"} & set(definition)):
                raise ValueError(
                    f"Sensitivity {parameter!r} needs max_dev_pct or absolute_values."
                )
            if "max_dev_pct" in definition:
                if "steps" not in definition:
                    raise ValueError(
                        f"Sensitivity {parameter!r} needs steps with max_dev_pct."
                    )
                for delta in _build_deltas(
                    definition["max_dev_pct"], definition["steps"]
                ):
                    _validate_parameter_value(
                        parameter, float(base_value) * (1.0 + delta), cfg
                    )
            for value in definition.get("absolute_values", []):
                _validate_parameter_value(parameter, value, cfg)


def _validate_explicit_scenarios(cfg: dict[str, Any]) -> None:
    """Validate names and overrides in optional explicit scenarios."""
    if "scenarios" not in cfg:
        return

    baseline = cfg["baseline_scenario"]
    scenarios = cfg["scenarios"]
    if isinstance(scenarios, dict):
        entries = list(scenarios.items())
    else:
        entries = [
            (
                item["scenario"],
                {key: value for key, value in item.items() if key != "scenario"},
            )
            for item in scenarios
        ]

    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("Explicit scenario names must be unique.")

    for name, overrides in entries:
        _validate_scenario_name(name)
        flattened = _flatten_scenario_overrides(overrides)
        unknown = set(flattened) - SENSITIVITY_PARAMETERS
        if unknown:
            raise ValueError(
                f"Scenario {name!r} has unsupported overrides: {sorted(unknown)}"
            )
        if name == baseline and flattened:
            raise ValueError(
                "The baseline scenario entry must be empty; configure baseline values at the top level."
            )
        for parameter, value in flattened.items():
            _validate_parameter_value(parameter, value, cfg)


def _flatten_scenario_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Flatten per-cycle turbine temperatures in explicit scenario overrides."""
    flattened = dict(overrides)
    temperatures = flattened.pop("temp_turbine_out", {})
    for cycle, value in temperatures.items():
        flattened[f"temp_turbine_out_{cycle}"] = value
    return flattened


def _validate_parameter_value(parameter: str, value: Any, cfg: dict[str, Any]) -> None:
    """Validate a sensitivity or explicit-scenario model value."""
    if parameter == "include_egs":
        if not isinstance(value, bool):
            raise ValueError("include_egs overrides must be boolean.")
        return
    if parameter == "SED_NAN_POLICY":
        if value not in {"always_egs", "depth_only"}:
            raise ValueError(f"Unsupported SED_NAN_POLICY value {value!r}.")
        return
    if parameter == "drilling_cost_curve":
        if value not in cfg["drilling_cost_curve_definitions"]:
            raise ValueError(f"Unknown drilling cost curve {value!r}.")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{parameter} overrides must be numeric.")

    numeric = float(value)
    if parameter == "max_depth":
        if not 250 <= numeric <= 10_000 or numeric % 250:
            raise ValueError(
                "max_depth must be a multiple of 250 between 250 and 10000 m."
            )
    elif parameter == "EGS_cutoff":
        if not 0 <= numeric <= 10_000:
            raise ValueError("EGS_cutoff must be between 0 and 10000 m.")
    elif parameter in {"capacity_factor", "fixed_OM_rate", "parasitic_fraction"}:
        if not 0 <= numeric <= 1:
            raise ValueError(f"{parameter} must be between 0 and 1.")
    elif parameter.startswith("temp_turbine_out_"):
        maximum = {
            "temp_turbine_out_ORC": 320,
            "temp_turbine_out_Flash": 485,
            "temp_turbine_out_Kalina": 180,
        }[parameter]
        if not 0 <= numeric <= maximum:
            raise ValueError(f"{parameter} must be between 0 and {maximum} degrees C.")
    elif parameter in {
        "stimulation_cost_per_doublet",
        "pump_cost",
        "orc_money_per_kw",
        "flash_money_per_kw",
        "kalina_money_per_kw",
        "min_dt",
    }:
        if numeric < 0:
            raise ValueError(f"{parameter} must be zero or greater.")
    elif numeric <= 0:
        raise ValueError(f"{parameter} must be greater than zero.")


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
            writer.writerow(
                {key: _format_tsv_value(row.get(key)) for key in fieldnames}
            )


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
                **_flatten_scenario_overrides(overrides),
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
            **_flatten_scenario_overrides(
                {k: v for k, v in scenario.items() if k != "scenario"}
            ),
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
                    updated_row["max_depth"], updated_row["EGS_cutoff"]
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
