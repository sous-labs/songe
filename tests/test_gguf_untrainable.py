import sys
from types import SimpleNamespace

import pytest

from songe import wording
from songe.models import ModelRef, trainable_base
from tests.crossplatform_helpers import fake_platform, write_gguf

pytest.importorskip("gguf")


def test_an_architecture_nothing_can_train_says_so_plainly(tmp_path, monkeypatch):
    # transformers 5.24 reads neither qwen35 (Qwen3.8) nor inkling2_loop
    # (Inkling 2) from GGUF; the Dream used to end in "did not finish".
    platform = fake_platform(tmp_path / "data")
    source = write_gguf(tmp_path / "Inkling-2-Mini-UD-Q4_K_M.gguf", architecture="inkling2_loop")
    ref = ModelRef("llamacpp", "i", "gguf", source, "inkling2_loop")

    def load_gguf_checkpoint(path, return_tensors=False):
        assert not return_tensors
        raise ValueError("GGUF model with architecture inkling2_loop is not supported yet.")

    reader = SimpleNamespace(load_gguf_checkpoint=load_gguf_checkpoint)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(modeling_gguf_pytorch_utils=reader))
    monkeypatch.setitem(sys.modules, "transformers.modeling_gguf_pytorch_utils", reader)
    calls = []
    with pytest.raises(ValueError) as stopped:
        trainable_base(ref, "torch", platform, runner=lambda *a, **k: calls.append(a))
    assert str(stopped.value) == wording.gguf_untrainable("inkling2_loop")
    assert calls == []  # nothing dequantized

    problem = wording.model_problem({"support": "unsupported", "reason": str(stopped.value)})
    assert problem == "Songe cannot train inkling2_loop from a GGUF file · it keeps teaching it with notes."


def test_a_damaged_gguf_is_left_to_the_converter(tmp_path, monkeypatch):
    platform = fake_platform(tmp_path / "data")
    source = write_gguf(tmp_path / "q.gguf")
    ref = ModelRef("llamacpp", "q", "gguf", source, "llama")

    def load_gguf_checkpoint(path, return_tensors=False):
        raise ValueError("GGUF magic invalid")

    reader = SimpleNamespace(load_gguf_checkpoint=load_gguf_checkpoint)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(modeling_gguf_pytorch_utils=reader))
    monkeypatch.setitem(sys.modules, "transformers.modeling_gguf_pytorch_utils", reader)
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=2**40))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError("conversion failed · songe doctor")

    with pytest.raises(RuntimeError, match="conversion failed"):
        trainable_base(ref, "torch", platform, runner=runner)
    assert len(calls) == 1
