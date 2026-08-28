"""Fast configuration and dependency checks that do not execute Snakemake."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import gregor
import jsonschema
import pytest
import yaml

MODULE_ROOT = Path(__file__).parent.parent


def _load_scenario_config() -> ModuleType:
    """Load scenario helpers without importing or executing Snakemake."""
    path = MODULE_ROOT / "workflow/scripts/scenario_config.py"
    spec = importlib.util.spec_from_file_location("scenario_config_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scenario helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCENARIO_CONFIG = _load_scenario_config()


@pytest.fixture
def example_config() -> dict:
    """Return an independent copy of the published example configuration."""
    with (MODULE_ROOT / "config/config.yaml").open() as config_file:
        return yaml.safe_load(config_file)


def test_published_yaml_files_parse() -> None:
    """Parse the public interface and both rule-time environments as YAML."""
    for relative_path in (
        "INTERFACE.yaml",
        "workflow/envs/module_geothermal.yaml",
        "workflow/envs/shell.yaml",
        "workflow/internal/settings.yaml",
    ):
        with (MODULE_ROOT / relative_path).open() as yaml_file:
            assert yaml.safe_load(yaml_file)


def test_example_config_passes_schema_and_relationship_validation(
    example_config: dict,
) -> None:
    """Validate the example with JSON Schema and module-level relationships."""
    with (MODULE_ROOT / "workflow/internal/config.schema.yaml").open() as schema_file:
        schema = yaml.safe_load(schema_file)
    jsonschema.Draft202012Validator(schema).validate(example_config)
    SCENARIO_CONFIG.validate_module_config(example_config)


def test_scenario_expansion_is_unique_and_contains_baseline(
    example_config: dict,
) -> None:
    """Expand the configured sensitivity sweep into path-safe unique names."""
    rows = SCENARIO_CONFIG.build_scenario_rows(example_config)
    names = [row["scenario"] for row in rows]
    assert names[0] == example_config["baseline_scenario"]
    assert len(names) == len(set(names))
    assert all(SCENARIO_CONFIG.SCENARIO_NAME_PATTERN.fullmatch(name) for name in names)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"baseline_scenario": "../unsafe"}, "Invalid scenario name"),
        ({"carrier_out": "heat"}, "Only carrier_out='power'"),
        ({"drilling_cost_curve": "missing"}, "has no matching definition"),
        ({"depth_categories": [[0, 5000], [4000, 6000]]}, "non-overlapping"),
        ({"fixed_bins": ["50-100", "0-25"]}, "non-overlapping"),
        (
            {
                "active_sensitivity_parameters": ["unsupported"],
                "sensitivity_parameters": {"unsupported": True},
            },
            "Unsupported sensitivity parameters",
        ),
        (
            {
                "active_sensitivity_parameters": ["flowrate_egs"],
                "sensitivity_parameters": {"flowrate_egs": {"max_dev_pct": 30}},
            },
            "needs steps with max_dev_pct",
        ),
        (
            {
                "scenarios": [
                    {"scenario": "duplicate"},
                    {"scenario": "duplicate", "lifetime": 40},
                ]
            },
            "must be unique",
        ),
        (
            {
                "active_sensitivity_parameters": ["max_depth"],
                "sensitivity_parameters": {
                    "max_depth": {"absolute_values": [3000, 3000]}
                },
            },
            "Expanded scenario names must be unique",
        ),
        (
            {"scenarios": {"base": {}, "bad_override": {"unknown": 1}}},
            "unsupported overrides",
        ),
        (
            {
                "scenarios": {
                    "base": {},
                    "invalid_orc_temperature": {"temp_turbine_out": {"ORC": 400}},
                }
            },
            "temp_turbine_out_ORC must be between 0 and 320",
        ),
    ],
)
def test_invalid_configuration_relationships_are_rejected(
    example_config: dict, change: dict, message: str
) -> None:
    """Reject invalid relationships that JSON Schema cannot fully express."""
    invalid = copy.deepcopy(example_config)
    invalid.update(change)
    with pytest.raises(ValueError, match=message):
        SCENARIO_CONFIG.validate_module_config(invalid)


def test_gregor_raster_aggregation_api_is_available() -> None:
    """Keep the Conda Gregor API used by regional aggregation under test."""
    assert callable(gregor.aggregate.aggregate_raster_to_polygon)
