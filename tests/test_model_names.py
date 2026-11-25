from songe.models.names import public_name


def test_a_hugging_face_cache_model_is_named_after_the_model():
    """An MLX model loaded from the Hugging Face cache was offered to the
    agent as songe/0e42af58449718de7931ee04f28191fbe6c43a56, the snapshot's
    commit hash (reported by Jack in a test on 11-24)."""
    snapshot = "0e42af58449718de7931ee04f28191fbe6c43a56"
    mac = (
        "/Users/someone/.cache/huggingface/hub/"
        f"models--Qwen--Qwen3-4B-Instruct-2507-MLX-8bit/snapshots/{snapshot}"
    )
    assert public_name(mac) == "songe/Qwen3-4B-Instruct-2507-MLX-8bit"
    assert public_name(mac + "/") == "songe/Qwen3-4B-Instruct-2507-MLX-8bit"
    windows = (
        "C:\\Users\\someone\\.cache\\huggingface\\hub\\"
        f"models--mlx-community--Qwen3-4B\\snapshots\\{snapshot}"
    )
    assert public_name(windows) == "songe/Qwen3-4B"
    # Other names are unchanged.
    assert public_name("qwen3:1.7b") == "songe/qwen3-1.7b"
    assert public_name("/models/Qwen3-8B") == "songe/Qwen3-8B"
    assert public_name("lmstudio-community/Qwen3-8B-GGUF") == "songe/Qwen3-8B-GGUF"
    assert public_name("songe/kept") == "songe/kept"
