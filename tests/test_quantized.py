import json
from types import SimpleNamespace

from songe.backends import quantized
from songe.backends.quantized import missing_packages


def checkpoint(tmp_path, quant=None):
    config = {"model_type": "qwen3"} | ({"quantization_config": quant} if quant else {})
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_a_quantized_checkpoint_names_the_package_it_needs(tmp_path, monkeypatch):
    """Live (#257): FP8 and NVFP4 builds failed at calibration, after 11 to
    23 minutes of practice, for want of kernels / compressed-tensors."""
    monkeypatch.setattr(quantized.importlib.util, "find_spec", lambda name: None)
    assert missing_packages(checkpoint(tmp_path, {"quant_method": "fp8"})) == ["kernels>=0.18.2,<0.19"]
    assert missing_packages(checkpoint(tmp_path, {"quant_method": "compressed-tensors"})) == [
        "compressed-tensors>=0.15.0"]
    # Unquantized, other methods (Inkling's packed experts) and unreadable
    # configs need nothing from here.
    assert missing_packages(checkpoint(tmp_path)) == []
    assert missing_packages(checkpoint(tmp_path, {"quant_method": "modelopt"})) == []
    assert missing_packages(tmp_path / "missing") == []


def test_an_installed_package_is_not_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(quantized.importlib.util, "find_spec", lambda name: SimpleNamespace(name=name))
    assert missing_packages(checkpoint(tmp_path, {"quant_method": "fp8"})) == []
