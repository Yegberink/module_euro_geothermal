"""
build_scenario_table.py
-----------------------
Snakemake script that generates a scenario TSV for sensitivity analysis.

Reads parameters from snakemake.params and writes a TSV where each row
is one scenario (one parameter changed from the base row).

Invoked by the `build_scenario_table` rule in automatic.smk.
Also safely importable by automatic.smk at parse time (the snakemake
entry-point block at the bottom is skipped when the module is imported).
"""

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np


# ============================================================
# Helpers
# ============================================================

def _format_delta_tag(delta: float) -> str:
    """Convert a fractional delta to a compact filename-safe tag."""
    pct = delta * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_str = str(int(round(pct)))
    else:
        pct_str = f"{pct:.6g}".rstrip("0").rstrip(".").replace(".", "_")
    return f"m{pct_str}" if pct < 0 else f"p{pct_str}"


def _make_unique(existing: set[str], name: str) -> str:
    """Append a counter suffix if name already exists."""
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse {x!r} as bool")


def _build_deltas(max_dev_pct: float, steps: int) -> list[float]:
    """
    Build symmetric list of fractional deltas.
    E.g. steps=2, max_dev_pct=20 -> [-0.2, -0.1, 0.1, 0.2]
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    pct_range = max_dev_pct / 100.0
    step = pct_range / steps
    neg = [-(k * step) for k in range(steps, 0, -1)]
    pos = [+(k * step) for k in range(1, steps + 1)]
    return neg + pos


def _flatten_base_config(cfg: dict) -> dict:
    """
    Flatten config into a single-level dict suitable for a TSV row.
    Handles the nested temp_turbine_out dict specifically.
    """
    flat = {}
    for k, v in cfg.items():
        if k == "temp_turbine_out" and isinstance(v, dict):
            for cycle, temp in v.items():
                flat[f"temp_turbine_out_{cycle}"] = float(temp)
        else:
            flat[k] = v
    return flat

def _format_scalar_for_label(value: Any) -> str:
    """Return a compact human-readable representation for sensitivity labels."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}"
    return str(value)


def _make_metadata(
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
            "sensitivity_pct_change": np.nan,
            "sensitivity_sort_magnitude": np.nan,
        }

    return {
        "sensitivity_parameter": param,
        "sensitivity_base_value": _format_scalar_for_label(base_value),
        "sensitivity_scenario_value": _format_scalar_for_label(scenario_value),
        "sensitivity_change_label": (
            f"{_format_scalar_for_label(base_value)} -> {_format_scalar_for_label(scenario_value)}"
        ),
        "sensitivity_pct_change": np.nan if pct_change is None else float(pct_change),
        "sensitivity_sort_magnitude": np.nan if sort_magnitude is None else float(sort_magnitude),
    }

# ============================================================
# Core logic
# ============================================================

def build_scenario_table(
    base_row: dict,
    active_parameters: list[str],
    parameter_definitions: dict,
    baseline_scenario: str,
) -> pd.DataFrame:
    """
    Generate a scenario DataFrame with one row per (parameter, delta) combination.

    Parameters
    ----------
    base_row:
        Flat dict of all base parameter values (numeric, bool, categorical).
    active_parameters:
        List of parameter names to vary (subset of parameter_definitions keys).
    parameter_definitions:
        Dict of parameter name -> definition. For numeric params:
            {"max_dev_pct": float, "steps": int}
        For boolean/categorical params:
            True  (just a flag to include them)
    baseline_scenario:
        Name of the base scenario row.
    """
    rows = [{"scenario": baseline_scenario, **base_row, **_make_metadata(None)}]
    existing_names: set[str] = {baseline_scenario}

    for param in active_parameters:
        if param not in parameter_definitions:
            raise ValueError(
                f"Parameter {param!r} listed in active_sensitivity_parameters "
                f"but not defined in sensitivity_parameters."
            )
        defn = parameter_definitions[param]

        base_val = base_row.get(param)

        # ── Boolean parameter ──────────────────────────────────────────────
        if isinstance(base_val, bool) or isinstance(defn, bool):
            base_bool = _parse_bool(base_row.get(param, defn))
            if not isinstance(base_val, bool) and isinstance(defn, bool):
                base_val = base_bool

            flipped = not base_bool

            updated_row = {**base_row, param: flipped}

            if param == "include_egs" and flipped is False:
                egs_cutoff = updated_row.get("EGS_cutoff")
                max_depth = updated_row.get("max_depth")

                if egs_cutoff is None or max_depth is None:
                    raise ValueError(
                        "include_egs sensitivity requires both 'EGS_cutoff' and 'max_depth' "
                        "to be present in the base config row."
                    )

                updated_row["max_depth"] = min(float(max_depth), float(egs_cutoff))

            name = _make_unique(existing_names, f"{baseline_scenario}_{param}_{flipped}")
            rows.append(
                {
                    "scenario": name,
                    **updated_row,
                    **_make_metadata(
                        param=param,
                        base_value=base_bool,
                        scenario_value=flipped,
                        sort_magnitude=1.0,
                    ),
                }
            )
            existing_names.add(name)

        # ── Categorical parameter ──────────────────────────────────────────
        elif isinstance(defn, (list, tuple)):
            allowed = [str(v) for v in defn]
            base_cat = str(base_val)
            for target in allowed:
                if target == base_cat:
                    continue
                name = _make_unique(existing_names, f"{baseline_scenario}_{param}_{target}")
                rows.append(
                    {
                        "scenario": name,
                        **base_row,
                        param: target,
                        **_make_metadata(
                            param=param,
                            base_value=base_cat,
                            scenario_value=target,
                            sort_magnitude=1.0,
                        ),
                    }
                )
                existing_names.add(name)

        # Legacy categorical placeholder support
        elif param == "SED_NAN_POLICY":
            allowed = ["always_egs", "depth_only"]
            base_cat = str(base_row.get("SED_NAN_POLICY", "always_egs"))
            for target in allowed:
                if target == base_cat:
                    continue
                name = _make_unique(existing_names, f"{baseline_scenario}_SED_NAN_POLICY_{target}")
                rows.append(
                    {
                        "scenario": name,
                        **base_row,
                        "SED_NAN_POLICY": target,
                        **_make_metadata(
                            param="SED_NAN_POLICY",
                            base_value=base_cat,
                            scenario_value=target,
                            sort_magnitude=1.0,
                        ),
                    }
                )
                existing_names.add(name)

        # ── Numeric parameter ──────────────────────────────────────────────
        else:
            if not isinstance(defn, dict):
                raise ValueError(
                    f"Expected a dict definition for numeric parameter {param!r}, got {defn!r}."
                )

            if base_val is None:
                raise ValueError(
                    f"Parameter {param!r} not found in base config row. "
                    f"Available keys: {list(base_row.keys())}"
                )

            v0 = float(base_val)
            candidates: list[tuple[float, float | None]] = []
            # each entry is (new_value, pct_change)

            has_relative = ("max_dev_pct" in defn) or ("steps" in defn)
            if has_relative:
                if "max_dev_pct" not in defn or "steps" not in defn:
                    raise ValueError(
                        f"Numeric parameter {param!r} must define both "
                        f"'max_dev_pct' and 'steps' for relative sweeps."
                    )

                if v0 == 0.0:
                    print(f"⚠️  Skipping relative sweep for {param!r}: base value is 0.")
                else:
                    max_dev_pct = float(defn["max_dev_pct"])
                    steps = int(defn["steps"])
                    deltas = _build_deltas(max_dev_pct, steps)
                    for delta in deltas:
                        v_new = v0 * (1.0 + delta)
                        candidates.append((v_new, delta * 100.0))

            if "absolute_values" in defn:
                abs_vals = defn["absolute_values"]
                if not isinstance(abs_vals, (list, tuple)):
                    raise ValueError(
                        f"'absolute_values' for {param!r} must be a list, got {type(abs_vals).__name__}."
                    )

                for raw_v in abs_vals:
                    v_new = float(raw_v)
                    pct_change = None if v0 == 0.0 else ((v_new / v0) - 1.0) * 100.0
                    candidates.append((v_new, pct_change))

            if not candidates:
                raise ValueError(
                    f"Numeric parameter {param!r} defines neither a relative sweep "
                    f"('max_dev_pct'/'steps') nor 'absolute_values'."
                )

            seen_values: list[float] = []
            for v_new, pct_change in candidates:
                if abs(v_new - v0) < 1e-12:
                    continue
                if any(abs(v_new - x) < 1e-12 for x in seen_values):
                    continue
                seen_values.append(v_new)

                tag = (
                    _format_delta_tag(pct_change / 100.0)
                    if pct_change is not None
                    else f"v{_format_scalar_for_label(v_new).replace('.', '_')}"
                )
                name = _make_unique(existing_names, f"{baseline_scenario}_{param}{tag}")
                rows.append(
                    {
                        "scenario": name,
                        **base_row,
                        param: v_new,
                        **_make_metadata(
                            param=param,
                            base_value=v0,
                            scenario_value=v_new,
                            pct_change=pct_change,
                            sort_magnitude=(
                                abs(pct_change) if pct_change is not None else abs(v_new - v0)
                            ),
                        ),
                    }
                )
                existing_names.add(name)
    return pd.DataFrame(rows)



# ============================================================
# Snakemake entry point
# Only executed when Snakemake runs this file as a job script.
# Skipped entirely when automatic.smk imports from this module
# at parse time, since `snakemake` is not in scope then.
# ============================================================

if "snakemake" in dir():
    p = snakemake.params  # noqa: F821

    base_row = _flatten_base_config(p.base_config)

    df = build_scenario_table(
        base_row=base_row,
        active_parameters=p.active_sensitivity_parameters,
        parameter_definitions=p.sensitivity_parameters,
        baseline_scenario=p.baseline_scenario,
    )

    output_path = snakemake.output.tsv  # noqa: F821
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"✓ Wrote {len(df)} scenarios ({len(df) - 1} sensitivity) to {output_path}")
