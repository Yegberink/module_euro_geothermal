"""Optional full workflow, report, and rule-graph tests over three shape sets.

These tests use the power integration configuration, without external heat-network
or eligibility inputs. Run explicitly; the integration CI task does not include them.
"""

import subprocess
from pathlib import Path
from urllib.request import urlretrieve

import pytest
import yaml

URL = "https://zenodo.org/records/20765043/files/{shapes}.parquet?download=1"
SHAPES = ("EUROPE_S_C1_ADM1", "EUROPE_V_C4_NUTS1", "EUROPE_L_C34_ADM1")
POWER_RESULTS = ("geothermal_regions", "overall_potential_summary")


def fetch_file(directory: Path, shapes: str) -> Path:
    """Fetch a shape input if it is not already available."""
    path = directory / f"resources/user/{shapes}/shapes.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(URL.format(shapes=shapes), path)
    return path


@pytest.fixture
def interface(module_path) -> dict:
    """Read the public output paths."""
    return yaml.safe_load((module_path / "INTERFACE.yaml").read_text())


@pytest.mark.parametrize("shapes", SHAPES)
def test_full_run(module_path: Path, interface: dict, shapes: str, tmp_path):
    """Build all power outputs, an HTML report, and a PNG rule graph."""
    fetch_file(module_path, shapes)
    config = yaml.safe_load(
        (module_path / "tests/integration/test_config.yaml").read_text()
    )["module_geothermal"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result_paths = [
        interface["pathvars"]["results"][name]["default"]
        .replace("<results>", "results")
        .format(shapes=shapes)
        for name in POWER_RESULTS
    ]
    result_paths.append(
        f"resources/automatic/{shapes}/figures/depth_cost_breakdown_four_panel.png"
    )
    command = ["snakemake", "--use-conda", "--cores", "4", *result_paths]
    command += ["--configfile", str(config_path)]
    subprocess.run(
        [*command, "--forceall", "--rerun-incomplete"], check=True, cwd=module_path
    )
    subprocess.run(
        [*command, "--report", f"results/{shapes}/report.html"],
        check=True,
        cwd=module_path,
    )
    graph = subprocess.run(
        [*command, "--rulegraph"], check=True, cwd=module_path, capture_output=True
    )
    subprocess.run(
        ["dot", "-Tpng", "-o", f"results/{shapes}/rulegraph.png"],
        input=graph.stdout,
        check=True,
        cwd=module_path,
    )
