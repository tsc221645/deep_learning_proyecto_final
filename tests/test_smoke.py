from pathlib import Path

import yaml


def test_package_imports():
    import predictguard

    assert predictguard.__name__ == "predictguard"


def test_project_config_loads():
    config_path = Path(__file__).parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["subset"] == "FD001"
    assert config["seed"] == 42
