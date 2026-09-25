"""Fast configuration and dependency checks that do not execute Snakemake."""

from __future__ import annotations

import copy
import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest
import yaml

MODULE_ROOT = Path(__file__).parent.parent


def _load_model_config() -> ModuleType:
    """Load configuration helpers without importing or executing Snakemake."""
    path = MODULE_ROOT / "workflow/scripts/model_config.py"
    spec = importlib.util.spec_from_file_location("model_config_for_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load configuration helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_CONFIG = _load_model_config()


@pytest.fixture
def example_config() -> dict:
    """Return an independent copy of the published example configuration."""
    with (MODULE_ROOT / "config/config.yaml").open() as config_file:
        return yaml.safe_load(config_file)


def test_published_yaml_files_parse() -> None:
    """Parse the public interface and rule-time environment as YAML."""
    for relative_path in (
        "INTERFACE.yaml",
        "workflow/envs/module.yaml",
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
    MODEL_CONFIG.validate_module_config(example_config)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"carrier_out": "invalid"}, "Unsupported carrier_out"),
        ({"carrier_out": "heat", "heat_reinjection_temperature": 90}, "below production"),
        ({"drilling_cost_curve": "missing"}, "has no matching definition"),
        ({"depth_categories": [[0, 5000], [4000, 6000]]}, "non-overlapping"),
        ({"fixed_bins": ["50-100", "0-25"]}, "non-overlapping"),
    ],
)
def test_invalid_configuration_relationships_are_rejected(
    example_config: dict, change: dict, message: str
) -> None:
    """Reject invalid relationships that JSON Schema cannot fully express."""
    invalid = copy.deepcopy(example_config)
    invalid.update(change)
    with pytest.raises(ValueError, match=message):
        MODEL_CONFIG.validate_module_config(invalid)


def test_gregor_is_declared_as_a_module_dependency() -> None:
    """Keep Gregor in the isolated rule-time environment."""
    with (MODULE_ROOT / "pixi.toml").open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    assert "gregor" in manifest["feature"]["module"]["dependencies"]


def test_carrier_parameter_tracking_is_independent(example_config: dict) -> None:
    """Allocation controls do not invalidate costs, nor heat mode power cubes."""
    heat = MODEL_CONFIG.heat_model_parameters(example_config)
    power = MODEL_CONFIG.model_parameters(example_config)
    example_config.update(carrier_out="heat_and_power", heat_network_buffer_km=20)
    assert MODEL_CONFIG.heat_model_parameters(example_config) == heat
    assert MODEL_CONFIG.model_parameters(example_config) == power
    example_config["heat_production_temperature"] = 90
    assert MODEL_CONFIG.heat_model_parameters(example_config) != heat
    assert MODEL_CONFIG.model_parameters(example_config) == power


@pytest.mark.parametrize(
    "key",
    [
        "baseline_scenario",
        "perform_sensitivity",
        "scenarios",
        "sens_tech",
        "active_sensitivity_parameters",
        "sensitivity_parameters",
    ],
)
def test_removed_controls_are_rejected(example_config: dict, key: str) -> None:
    """Reject obsolete controls rather than silently ignoring requested runs."""
    schema = yaml.safe_load(
        (MODULE_ROOT / "workflow/internal/config.schema.yaml").read_text()
    )
    example_config[key] = None
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        jsonschema.Draft202012Validator(schema).validate(example_config)
