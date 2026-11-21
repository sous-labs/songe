import json
import plistlib

from songe import agents, launchd
from songe.agents.opencode import configure


def test_backups_idempotence_and_restore(tmp_path, config):
    home = tmp_path / "home"
    oc = home / ".config/opencode/opencode.json"
    pi = home / ".pi/agent/models.json"
    oc.parent.mkdir(parents=True)
    pi.parent.mkdir(parents=True)
    original = b'{"model":"local/original", "provider":{"existing":{}}}\n'
    oc.write_bytes(original)
    root = tmp_path / "songe"
    root.mkdir()
    changes = agents.proposed(config, home)
    assert len(changes) == 2
    assert all("127.0.0.1:7613" in c[3] for c in changes)
    agents.apply(changes, root)
    assert not agents.proposed(config, home)
    assert (
        json.loads(pi.read_text(encoding="utf-8"))["providers"]["songe"]["api"]
        == "openai-completions"
    )
    assert agents.restore(root) == []
    assert oc.read_bytes() == original
    assert not pi.exists()


def test_user_changes_are_preserved_on_restore(tmp_path, config):
    home = tmp_path / "home"
    path = home / ".pi/agent/models.json"
    path.parent.mkdir(parents=True)
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    path.write_text('{"user":"new changes"}', encoding="utf-8")
    # The user's own later change stays, and is no reason to keep Songe's data.
    assert agents.restore(root) == []
    assert "new changes" in path.read_text(encoding="utf-8")


def test_restore_never_leaves_the_agent_on_a_songe_model(tmp_path, config):
    import json

    home = tmp_path / "home"
    path = home / ".config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"model": "lmstudio/mine"}', encoding="utf-8")
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home, default=True), root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"] = "songe/another-model"  # picked another Songe model later
    path.write_text(json.dumps(data), encoding="utf-8")
    assert agents.restore(root) == []
    assert json.loads(path.read_text(encoding="utf-8")) == {"model": "lmstudio/mine"}


def test_opencode_schema_versions(config):
    assert (
        configure({}, config)["provider"]["songe"]["npm"] == "@ai-sdk/openai-compatible"
    )
    assert (
        configure({}, config, v2=True)["providers"]["songe"]["package"]
        == "@opencode/ai/providers/openai-compatible"
    )
    assert configure({}, config, default=True)["model"] == "songe/songe/test"


def test_plist_explicit_target_no_launchctl(tmp_path, config, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("launchctl must not run")

    monkeypatch.setattr(launchd.subprocess, "run", forbidden)
    target = tmp_path / "LaunchAgents/songe.plist"
    launchd.install(config, tmp_path, target, activate=False)
    data = plistlib.loads(target.read_bytes())
    assert data["KeepAlive"]
    assert data["ProgramArguments"][-1] == "serve"
    assert data["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"
    launchd.remove(target, activate=False)
    assert not target.exists()


def test_opencode_default_is_the_chosen_model_not_the_first_listed(config):
    from songe.agents.opencode import configure

    config.exposed_models = ["LFM2.5-24B-A2B-MLX-4bit", "Qwen3.5-4B-MLX-4bit"]
    config.public_model = "songe/Qwen3.5-4B-MLX-4bit"
    data = configure({}, config, default=True, v2=True)
    assert data["model"].endswith("Qwen3.5-4B-MLX-4bit")
    assert "LFM2.5" not in data["model"]


def test_local_default_routes_through_songe_and_cloud_stays(config):
    from songe.agents.opencode import configure

    config.exposed_models = ["Qwen3.5-4B-MLX-4bit", "LFM3-1.7B-MLX-4bit"]
    config.public_model = "songe/Qwen3.5-4B-MLX-4bit"
    # A local default moves to the Songe version of the same model (live:
    # init made a model the machine could not train the default instead).
    qwen = configure({"model": "lmstudio/qwen3.5-4b-mlx", "small_model": "lmstudio/lfm3-1.7b"}, config)
    assert qwen["model"] == "songe/songe/Qwen3.5-4B-MLX-4bit"
    assert qwen["small_model"] == "lmstudio/lfm3-1.7b"  # titles are untouched
    assert configure({"model": "lmstudio/lfm3-1.7b"}, config)["model"] == "songe/songe/LFM3-1.7B-MLX-4bit"
    # A local model Songe does not offer, or no default at all, is left as it is.
    assert configure({"model": "lmstudio/bonsai-27b"}, config)["model"] == "lmstudio/bonsai-27b"
    assert "model" not in configure({}, config)
    # A cloud default is the user's choice.
    cloud = configure({"model": "anthropic/claude-sonnet"}, config)
    assert cloud["model"] == "anthropic/claude-sonnet"
    # A localhost provider counts as local; --keep-default leaves it.
    custom = {
        "model": "mine/qwen3.5-4b",
        "provider": {"mine": {"options": {"baseURL": "http://127.0.0.1:8080/v1"}}},
    }
    assert configure(custom, config)["model"] == "songe/songe/Qwen3.5-4B-MLX-4bit"
    assert configure(custom, config, default=False)["model"] == "mine/qwen3.5-4b"


def test_songe_entries_refresh_limits_and_keep_user_fields(config):
    from songe.agents.opencode import configure

    config.context_window = 262144
    name = configure({}, config, default=False)
    key = next(iter(name["provider"]["songe"]["models"]))
    stale = {
        "provider": {
            "songe": {
                "models": {
                    key: {"name": key, "limit": {"context": 8192}, "note": "mine"}
                }
            }
        }
    }
    entry = configure(stale, config, default=False)["provider"]["songe"]["models"][key]
    assert entry["limit"]["context"] == 262144 and entry["note"] == "mine"


def test_agents_follow_the_servers_context_per_model(config, tmp_path, monkeypatch):
    import json

    from songe import agents
    from songe.agents.opencode import configure

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    config.exposed_models = ["Qwen3.5-4B-MLX-4bit"]
    name = next(
        iter(configure({}, config, default=False)["provider"]["songe"]["models"])
    )
    path = tmp_path / "config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(configure({"model": "anthropic/x"}, config, default=False)),
        encoding="utf-8",
    )
    # LM Studio serves this model with 32768; only Songe's limits change.
    config.served_contexts = {name: 32768}
    assert agents.refresh_limits(config, home=tmp_path) == [path]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provider"]["songe"]["models"][name]["limit"]["context"] == 32768
    assert data["model"] == "anthropic/x"
    assert agents.refresh_limits(config, home=tmp_path) == []  # nothing new


def test_context_refresh_stays_songes_edit_for_uninstall(config, tmp_path, monkeypatch):
    import json

    from songe import agents
    from songe.agents.opencode import configure

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    config.exposed_models = ["Qwen3.5-4B-MLX-4bit"]
    root = tmp_path / "data"
    root.mkdir()
    path = tmp_path / "config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    original = b'{"model": "anthropic/x"}\n'
    path.write_bytes(original)
    installed = configure(json.loads(original), config, default=False)
    agents.apply([(path, original, json.dumps(installed).encode(), "")], root)
    name = next(iter(installed["provider"]["songe"]["models"]))
    config.served_contexts = {name: 32768}
    assert agents.refresh_limits(config, home=tmp_path, root=root) == [path]
    # Uninstall restores the user's file exactly; no false conflict.
    assert agents.restore(root) == []
    assert path.read_bytes() == original


def test_commented_opencode_config_never_crashes_the_service(
    config, tmp_path, monkeypatch
):
    """opencode accepts comments; Songe must not crash-loop or rewrite them."""
    import json

    import pytest

    from songe import agents
    from songe.agents.opencode import configure

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))
    config.exposed_models = ["Qwen3.5-4B-MLX-4bit"]
    root = tmp_path / "data"
    root.mkdir()
    path = tmp_path / "config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    original = b'{"model": "anthropic/x"}\n'
    path.write_bytes(original)
    installed = configure(json.loads(original), config, default=False)
    agents.apply([(path, original, json.dumps(installed).encode(), "")], root)
    commented = b"// mine\n" + json.dumps(installed).encode()
    path.write_bytes(commented)
    agents.migrate(config, root)  # the service's startup path
    config.served_contexts = {"songe/Qwen3.5-4B-MLX-4bit": 32768}
    assert agents.refresh_limits(config, home=tmp_path, root=root) == []
    assert agents.restore(root) == [str(path)]
    assert path.read_bytes() == commented
    with pytest.raises(agents.NotPlainJSON, match="comments"):
        agents.read_json(path)


def test_routing_says_which_agents_still_need_a_songe_model(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    assert agents.routing(home) == {}  # neither agent installed
    oc = home / ".config/opencode/opencode.json"
    oc.parent.mkdir(parents=True)
    oc.write_text('{"model": "anthropic/claude"}', encoding="utf-8")
    (home / ".pi/agent").mkdir(parents=True)
    assert agents.routing(home) == {"opencode": False, "pi": False}
    oc.write_text('{"model": "songe/songe/qwen"}', encoding="utf-8")
    (home / ".pi/agent/settings.json").write_text(
        '{"defaultProvider": "songe", "defaultModel": "qwen"}', encoding="utf-8"
    )
    assert agents.routing(home) == {"opencode": True, "pi": True}


def test_init_ends_with_the_step_left(config, monkeypatch):
    from songe import cli

    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": True, "pi": False})
    assert cli.next_step(config) == "One step left · pick a songe/… model in pi with /model"
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": True})
    assert cli.next_step(config).startswith("Use your agent as usual")
    monkeypatch.setattr(agents, "routing", lambda home=None: {})
    assert "127.0.0.1:7613/v1" in cli.next_step(config)


def test_init_plan_names_agents_it_did_not_find(config, monkeypatch):
    from songe import cli

    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": False})
    assert cli.missing_agents(config) == [
        ("Agent", "pi not found · run songe init again after installing it")
    ]
    monkeypatch.setattr(agents, "routing", lambda home=None: {})
    rows = cli.missing_agents(config)
    assert rows[-1] == ("Other", "any OpenAI-compatible tool · http://127.0.0.1:7613/v1")
