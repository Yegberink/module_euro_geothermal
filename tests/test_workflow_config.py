"""Isolated workflow checks without downloads or full model execution."""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SNAKEMAKE = str(Path(sys.executable).with_name("snakemake"))
pytestmark = pytest.mark.workflow


def prepare_workflow(folder, config, targets):
    """Import the actual module with isolated file locations."""
    (folder / "config.yaml").write_text(yaml.safe_dump(config))
    (folder / "Snakefile").write_text(f"""
configfile: "config.yaml"
module geothermal:
    snakefile: {str(ROOT / "workflow/Snakefile")!r}
    config: config
    pathvars:
        resources="resources",
        results="results",
        logs="logs",
use rule * from geothermal as geothermal_*
rule provide_shapes:
    output: "resources/user/{{shapes}}/shapes.parquet"
    shell: "touch {{output:q}}"
rule provide_networks:
    output: "resources/user/{{shapes}}/areas.parquet"
    shell: "touch {{output:q}}"
rule all:
    default_target: True
    input: {targets!r}
""")


def snakemake(folder, *args):
    """Run Snakemake and include full diagnostics on failure."""
    result = subprocess.run(
        [SNAKEMAKE, "--cores", "1", *args],
        cwd=folder,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


@pytest.mark.parametrize("method_2d", ["LCOE_minimization", "depth_categories"])
@pytest.mark.parametrize("method_region", ["cutoff", "fixed_bins", "quantiles"])
def test_single_run_targets(tmp_path, method_2d, method_region):
    """Resolve every retained output for all supported aggregation combinations."""
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["method_2d_aggregation"] = method_2d
    config["method_region_aggregation"] = method_region
    cycles = {
        "cutoff": ["orc"],
        "fixed_bins": ["flash", "kalina"],
        "quantiles": ["orc", "flash", "kalina"],
    }[method_region]
    config["which_cycles"] = cycles
    targets = []
    for shapes in ["test", "second"]:
        targets.extend(
            [
                f"results/{shapes}/geothermal_regions.parquet",
                f"results/{shapes}/overall_potential_summary.tsv",
                f"resources/automatic/{shapes}/figures/depth_cost_breakdown_four_panel.png",
            ]
        )
        if method_2d == "LCOE_minimization":
            targets.extend(
                [
                    f"results/{shapes}/lcoe_overview.png",
                    f"resources/automatic/{shapes}/figures/lcoe_potential_cumulative_four_panel.png",
                ]
            )
            if method_region == "cutoff":
                targets.append(
                    f"results/{shapes}/figures/potential_overview_all_depths_cutoff_75.png"
                )
    prepare_workflow(tmp_path, config, targets)
    output = snakemake(tmp_path, "--dry-run")
    calculations = re.findall(
        r"Calculate (orc|flash|kalina) geothermal potential", output
    )
    assert sorted(calculations) == sorted(cycles * 2)
    assert "scenario" not in output.lower()
    assert "sensitivity" not in output.lower()
    assert "geothermal_overall_potential" in output
    assert "geothermal_depth_cost_breakdown_four_panel" in output


def test_model_config_change_triggers_recalculation(tmp_path):
    """Record a completed cube and verify scalar and nested config rerun tracking."""
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    target = "resources/automatic/test/documents/orc_lcoe_cube.nc"
    prepare_workflow(tmp_path, config, [target])
    for name in [
        "resources/automatic/test/generic/clipped_temp_voxel.nc",
        "resources/automatic/GST1_sediment_thickness.nc",
        target,
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    snakemake(
        tmp_path,
        target,
        "--touch",
        "--forceall",
        "--allowed-rules",
        "geothermal_calculate_LCOE",
    )
    unchanged = snakemake(
        tmp_path, target, "--dry-run", "--allowed-rules", "geothermal_calculate_LCOE"
    )
    assert "Nothing to be done" in unchanged
    for key in ["lifetime", "temp_turbine_out"]:
        changed = yaml.safe_load(yaml.safe_dump(config))
        if key == "lifetime":
            changed[key] = 40
        else:
            changed[key]["ORC"] = 80
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(changed))
        output = snakemake(
            tmp_path,
            target,
            "--dry-run",
            "--allowed-rules",
            "geothermal_calculate_LCOE",
        )
        assert "Params have changed" in output
        assert "Calculate orc geothermal potential" in output


@pytest.mark.parametrize("carrier", ["heat", "heat_and_power"])
@pytest.mark.parametrize("method_2d", ["LCOE_minimization", "depth_categories"])
@pytest.mark.parametrize("method_region", ["cutoff", "fixed_bins", "quantiles"])
def test_heat_workflow_targets(tmp_path, carrier, method_2d, method_region):
    """Resolve regional, network and allocation targets with the required branches."""
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config.update(
        carrier_out=carrier,
        method_2d_aggregation=method_2d,
        method_region_aggregation=method_region,
    )
    prepare_workflow(
        tmp_path,
        config,
        [
            "results/test/geothermal_regions.parquet",
            "results/test/geothermal_heat_networks.parquet",
            "results/test/geothermal_heat_allocations.parquet",
        ],
    )
    output = snakemake(tmp_path, "--dry-run")
    assert "geothermal_calculate_LCOH" in output
    assert "geothermal_allocate_heat" in output
    assert ("geothermal_calculate_LCOE" in output) == (carrier == "heat_and_power")


def test_heat_without_power_configuration(tmp_path):
    """Heat does not require cycle settings or electricity plant parameters."""
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["carrier_out"] = "heat"
    for key in [
        "which_cycles",
        "temp_turbine_out",
        "capacity_factor",
        "parasitic_fraction",
        "orc_money_per_kw",
        "flash_money_per_kw",
        "kalina_money_per_kw",
        "min_dt",
        "method_2d_aggregation",
        "method_region_aggregation",
    ]:
        config.pop(key)
    prepare_workflow(tmp_path, config, ["results/test/geothermal_regions.parquet"])
    output = snakemake(tmp_path, "--dry-run")
    assert "geothermal_calculate_LCOH" in output
    assert "geothermal_calculate_LCOE" not in output


def test_heat_model_changes_trigger_only_relevant_jobs(tmp_path):
    """Heat economics rerun for temperatures, but not demand-allocation controls."""
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["carrier_out"] = "heat"
    targets = [
        f"resources/automatic/test/documents/heat_{kind}.tif"
        for kind in ["lcoh", "potential", "depth"]
    ]
    prepare_workflow(tmp_path, config, targets)
    for name in [
        "resources/automatic/temperature_voxel.nc",
        "resources/automatic/GST1_sediment_thickness.nc",
        "resources/automatic/test/generic/heat_extent.json",
        *targets,
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    snakemake(
        tmp_path,
        *targets,
        "--touch",
        "--forceall",
        "--allowed-rules",
        "geothermal_calculate_LCOH",
    )
    changed = dict(config, heat_network_buffer_km=15)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(changed))
    output = snakemake(
        tmp_path, *targets, "--dry-run", "--allowed-rules", "geothermal_calculate_LCOH"
    )
    assert "Nothing to be done" in output
    changed["heat_production_temperature"] = 90
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(changed))
    output = snakemake(
        tmp_path, *targets, "--dry-run", "--allowed-rules", "geothermal_calculate_LCOH"
    )
    assert "Params have changed" in output
