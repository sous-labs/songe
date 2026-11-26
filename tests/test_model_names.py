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


def test_a_hugging_face_cache_model_without_an_org_and_a_folder_called_snapshots():
    """A repository without an org (models--gpt2) is named after the model
    too; a folder that is only called snapshots is not a cache snapshot."""
    commit = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    mac = f"/Users/someone/.cache/huggingface/hub/models--gpt2/snapshots/{commit}"
    assert public_name(mac) == "songe/gpt2"
    windows = mac.replace("/Users/someone", "C:\\Users\\someone").replace("/", "\\")
    assert public_name(windows) == "songe/gpt2"
    assert public_name("/Users/someone/models/snapshots") == "songe/snapshots"
    assert public_name("/Users/someone/backups/snapshots/Qwen3-8B") == "songe/Qwen3-8B"
