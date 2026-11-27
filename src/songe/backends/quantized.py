"""Packages transformers needs to load a quantized checkpoint for training.

Songe's install does not bring them. Without one, a Deep Dream ran its
practice (11 to 23 minutes, live) and only then failed at calibration with
"calibration failed · songe doctor" (#257). Checked before practice instead.
"""

import importlib.util
import json
from pathlib import Path

# quant_method in config.json -> (module to import, what to install). The
# versions are the ones transformers 5.24 asks for.
PACKAGES = {
    "fp8": ("kernels", "kernels>=0.18.2,<0.19"),
    "compressed-tensors": ("compressed_tensors", "compressed-tensors>=0.15.0"),
}


def missing_packages(model_dir):
    """What to install before this checkpoint can train, or [] (also when
    its config cannot be read: the trainer then says why itself)."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    from .packed_experts import REPACKED_TYPES

    if isinstance(config, dict) and config.get("model_type") in REPACKED_TYPES:
        return []  # its FP8 experts are decoded by Songe itself (packed_experts)
    quant = config.get("quantization_config") if isinstance(config, dict) else None
    method = quant.get("quant_method") if isinstance(quant, dict) else None
    if method not in PACKAGES:
        return []
    module, requirement = PACKAGES[method]
    return [] if importlib.util.find_spec(module) else [requirement]
