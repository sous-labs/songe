import json
import plistlib
from pathlib import Path

import pytest

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


def test_pi_defaults_session_affinity_for_songe_models(config):
    from songe.agents.pi import configure

    new_model = configure({}, config)["providers"]["songe"]["models"][0]
    assert new_model["compat"]["sendSessionAffinityHeaders"] is True

    config_data = {
        "providers": {
            "songe": {
                "models": [
                    {"id": config.public_model, "compat": {"supportsStore": False}},
                    {"id": "user-model"},
                ]
            }
        }
    }
    managed, other = configure(config_data, config)["providers"]["songe"]["models"]
    assert managed["compat"] == {
        "supportsStore": False,
        "sendSessionAffinityHeaders": True,
    }
    assert other == {"id": "user-model"}
    config_data["providers"]["songe"]["models"][0]["compat"][
        "sendSessionAffinityHeaders"
    ] = False
    managed = configure(config_data, config)["providers"]["songe"]["models"][0]
    assert managed["compat"]["sendSessionAffinityHeaders"] is False


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
    assert cli.next_step(config) == "One step left · pick a Songe model in pi with /model"
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": True})
    assert cli.next_step(config).startswith("Use your agent as usual")
    monkeypatch.setattr(agents, "routing", lambda home=None: {})
    assert "127.0.0.1:7613/v1" in cli.next_step(config)


def test_init_plan_names_agents_it_did_not_find(config, monkeypatch):
    from songe import cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": False})
    assert cli.missing_agents(config) == [
        ("Agent", "pi not found · run songe init again after installing it")
    ]
    monkeypatch.setattr(agents, "routing", lambda home=None: {})
    rows = cli.missing_agents(config)
    assert rows[-1] == ("Other", "any OpenAI-compatible tool · http://127.0.0.1:7613/v1")


def test_init_plan_says_an_installed_agent_needs_one_start(config, monkeypatch):
    # #70: pi on PATH but never started has no ~/.pi/agent yet; "not found
    # · install it" asked the user to install what they already had.
    from songe import cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": False})
    assert cli.missing_agents(config) == [
        ("Agent", "pi not set up yet · start it once, then run songe init again")
    ]


def test_a_new_backup_never_overwrites_a_kept_one(tmp_path):
    # After a restore that keeps one entry (a conflict), the manifest is
    # smaller than the backups folder: numbering by its size reused the kept
    # entry's backup and lost that agent's original configuration.
    root = tmp_path / "songe"
    (root / "backups").mkdir(parents=True)
    kept = root / "backups" / "1.json"
    kept.write_bytes(b'{"original": "pi"}')
    (root / "agent-backups.json").write_text(
        json.dumps(
            {
                str(tmp_path / "pi.json"): {
                    "backup": str(kept),
                    "written": "{}",
                    "edits": [],
                }
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "opencode.json"
    original = b'{"original": "opencode"}'
    path.write_bytes(original)
    agents.apply([(path, original, b'{"model": "songe/x"}', "")], root)
    assert kept.read_bytes() == b'{"original": "pi"}'
    manifest = json.loads((root / "agent-backups.json").read_text(encoding="utf-8"))
    assert Path(manifest[str(path)]["backup"]).read_bytes() == original


def installed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    (home / ".config/opencode").mkdir(parents=True)
    (home / ".pi/agent").mkdir(parents=True)
    root = tmp_path / "songe"
    root.mkdir()
    return home, root


def test_restore_undoes_every_agent_when_a_backup_is_gone(tmp_path, config, monkeypatch):
    # Uninstall must not stop at the first missing backup and leave the other
    # agents pointed at a stopped service.
    home, root = installed(tmp_path, monkeypatch)
    oc = home / ".config/opencode/opencode.json"
    pi = home / ".pi/agent/models.json"
    oc.write_text('{"theme": "dark"}', encoding="utf-8")
    pi.write_text('{"providers": {"mine": {}}}', encoding="utf-8")
    agents.apply(agents.proposed(config, home), root)
    for backup in (root / "backups").iterdir():
        backup.unlink()
    assert agents.restore(root) == []
    assert json.loads(oc.read_text(encoding="utf-8")) == {"theme": "dark"}
    assert json.loads(pi.read_text(encoding="utf-8")) == {"providers": {"mine": {}}}


def test_empty_or_bom_opencode_config_is_left_alone(tmp_path, config, monkeypatch, capsys):
    home, _ = installed(tmp_path, monkeypatch)
    oc = home / ".config/opencode/opencode.json"
    for raw in (b"", b'\xef\xbb\xbf{"theme": "dark"}'):
        oc.write_bytes(raw)
        assert [c[0] for c in agents.proposed(config, home)] == [home / ".pi/agent/models.json"]
        assert f"Skipped {oc}" in capsys.readouterr().err
        assert oc.read_bytes() == raw


def test_routing_reads_a_plain_opencode_jsonc(tmp_path, monkeypatch):
    home, _ = installed(tmp_path, monkeypatch)
    (home / ".config/opencode/opencode.jsonc").write_text(
        '{"model": "songe/songe/qwen"}', encoding="utf-8"
    )
    assert agents.routing(home)["opencode"] is True


def test_restore_after_user_edits_keeps_their_fields_byte_for_byte(
    tmp_path, config, monkeypatch
):
    # What uninstall writes back, pinned for opencode and pi after the user
    # edited both files since init (non-ASCII stays escaped, as before).
    home, root = installed(tmp_path, monkeypatch)
    oc = home / ".config/opencode/opencode.json"
    pi = home / ".pi/agent/models.json"
    oc.write_text(
        '{"model": "lmstudio/qwen", "theme": "café",'
        ' "provider": {"lmstudio": {"options": {"baseURL": "http://127.0.0.1:1234/v1"}}}}',
        encoding="utf-8",
    )
    pi.write_text('{"providers": {"mine": {"baseUrl": "http://x/v1"}}}', encoding="utf-8")
    agents.apply(agents.proposed(config, home), root)
    data = json.loads(oc.read_text(encoding="utf-8"))
    data["keybinds"] = {"leader": "ctrl+x"}
    data["provider"]["lmstudio"]["name"] = "Mon LM Studio"
    oc.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    data = json.loads(pi.read_text(encoding="utf-8"))
    data["providers"]["other"] = {"baseUrl": "http://y/v1"}
    pi.write_text(json.dumps(data), encoding="utf-8")
    assert agents.restore(root) == []
    assert oc.read_bytes() == OC_RESTORED
    assert pi.read_bytes() == PI_RESTORED


OC_RESTORED = (
    b'{\n  "model": "lmstudio/qwen",\n  "theme": "caf\\u00e9",\n  "provider": {\n'
    b'    "lmstudio": {\n      "options": {\n        "baseURL": "http://127.0.0.1:1234/v1"\n'
    b'      },\n      "name": "Mon LM Studio"\n    }\n  },\n  "keybinds": {\n'
    b'    "leader": "ctrl+x"\n  }\n}\n'
)
PI_RESTORED = (
    b'{\n  "providers": {\n    "mine": {\n      "baseUrl": "http://x/v1"\n    },\n'
    b'    "other": {\n      "baseUrl": "http://y/v1"\n    }\n  }\n}\n'
)


def test_an_opencode_file_of_another_shape_binds_nothing(tmp_path, monkeypatch):
    from songe.agents.opencode import OpenCode

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".local/state"))
    folder = OpenCode().folder(tmp_path)
    folder.mkdir(parents=True)
    for text in ("[]", '{"model": 5}', '{"model": "x/y", "provider": {"x": "str"}}'):
        (folder / "opencode.json").write_text(text, encoding="utf-8")
        assert OpenCode().binding(tmp_path) == ("", "")


def aider_home(tmp_path):
    home = tmp_path / "home"
    (home / ".aider").mkdir(parents=True)  # Aider has run once
    return home


def test_aider_gets_named_songe_models_and_restore_undoes_them(tmp_path, config, monkeypatch):
    # #64: Aider reaches Songe through per-model entries; OPENAI_API_BASE
    # would also send the user's real OpenAI models to Songe.
    import yaml

    monkeypatch.delenv("AIDER_MODEL", raising=False)
    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    original = b"- name: gpt-4o\n  edit_format: diff\n"
    settings.write_bytes(original)
    root = tmp_path / "songe"
    root.mkdir()
    changes = agents.proposed(config, home)
    assert sorted(p.name for p, *_ in changes) == [
        ".aider.model.metadata.json",
        ".aider.model.settings.yml",
    ]
    agents.apply(changes, root)
    name = "openai/" + config.public_model
    entries = {e["name"]: e for e in yaml.safe_load(settings.read_text(encoding="utf-8"))}
    assert entries["gpt-4o"] == {"name": "gpt-4o", "edit_format": "diff"}
    assert entries[name]["extra_params"] == {
        "api_base": "http://127.0.0.1:7613/v1",
        "api_key": "local",
    }
    metadata = json.loads((home / ".aider.model.metadata.json").read_text(encoding="utf-8"))
    assert metadata[name]["max_input_tokens"] == 8192
    assert metadata[name]["litellm_provider"] == "openai"
    assert not agents.proposed(config, home)  # idempotent
    assert agents.routing(home) == {"aider": False}
    assert agents.restore(root) == []
    assert settings.read_bytes() == original
    assert not (home / ".aider.model.metadata.json").exists()


def test_aider_entries_follow_the_servers_context(tmp_path, config):
    home = aider_home(tmp_path)
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    config.served_contexts = {config.public_model: 32768}
    assert len(agents.refresh_limits(config, home, root=root)) == 1
    metadata = json.loads((home / ".aider.model.metadata.json").read_text(encoding="utf-8"))
    assert metadata["openai/" + config.public_model]["max_input_tokens"] == 32768
    assert agents.restore(root) == []


def test_commented_aider_settings_leave_all_of_aider_alone_with_notes(
    tmp_path, config, capsys
):
    # Metadata without the settings' api_base would send openai/songe/...
    # to api.openai.com with the user's key: Aider gets both files or none.
    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    settings.write_text("# my models\n- name: gpt-4o\n", encoding="utf-8")
    assert agents.proposed(config, home) == []
    note = capsys.readouterr().err
    assert f"Skipped {settings} · it has comments" in note
    assert f"Skipped {home / '.aider.model.metadata.json'} · {settings.name} is skipped" in note
    assert "api_base: http://127.0.0.1:7613/v1" in note and "max_input_tokens" in note


def test_doctor_and_plan_name_aider_once(tmp_path, config, monkeypatch, capsys):
    from songe import cli

    home = aider_home(tmp_path)
    root = tmp_path / "songe"
    root.mkdir()
    changes = agents.proposed(config, home)
    assert {cli.agent_name(p) for p, *_ in changes} == {"aider"}
    agents.apply(changes, root)
    settings = home / ".aider.model.settings.yml"
    adapter = agents.adapter_for(settings)
    assert adapter.status(settings, agents.load(settings), config) == "configured"
    assert adapter.pick == "--model openai/songe/…"
    assert "aider" not in agents.KNOWN  # never listed as "not found"


def test_restore_reads_each_file_with_the_codec_it_was_written_with(
    tmp_path, config, monkeypatch
):
    # The manifest records the codec, so a later Songe that no longer knows
    # the file (or maps its name differently) still undoes it exactly.
    from pathlib import Path

    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    original = b"- name: gpt-4o\n  edit_format: diff\n"
    settings.write_bytes(original)
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    manifest = json.loads((root / "agent-backups.json").read_text(encoding="utf-8"))
    assert {Path(k).name: v["codec"] for k, v in manifest.items()} == {
        ".aider.model.settings.yml": "yaml-list:name",
        ".aider.model.metadata.json": "json",
    }
    monkeypatch.setattr(agents, "ADAPTERS", agents.ADAPTERS[:2])  # no Aider
    assert agents.restore(root) == []
    assert settings.read_bytes() == original


@pytest.mark.parametrize(
    "raw,why",
    [
        (b"- name: gpt-4o\n- name: gpt-4o\n  edit_format: diff\n", "two entries named gpt-4o"),
        (b"- name: caf\xe9\n", "it is not UTF-8"),
        (b"{}\n", "not a list of entries"),
        (b"0\n", "not a list of entries"),
        (b"false\n", "not a list of entries"),
    ],
)
def test_aider_settings_songe_cannot_rewrite_are_left_alone(tmp_path, config, capsys, raw, why):
    # Repeated names would lose one entry for good; non-UTF-8 crashed init;
    # `{}`, `0` and `false` were read as an empty list and replaced.
    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    settings.write_bytes(raw)
    assert agents.proposed(config, home) == []
    assert why in capsys.readouterr().err
    assert settings.read_bytes() == raw


def test_non_utf8_json_config_is_skipped_not_a_crash(tmp_path, config, monkeypatch, capsys):
    home, _ = installed(tmp_path, monkeypatch)
    oc = home / ".config/opencode/opencode.json"
    oc.write_bytes(b'{"theme": "caf\xe9"}')
    assert [c[0] for c in agents.proposed(config, home)] == [home / ".pi/agent/models.json"]
    assert f"Skipped {oc} · it is not UTF-8" in capsys.readouterr().err
    assert agents.refresh_limits(config, home) == []


def test_yaml_user_edit_then_uninstall_keeps_their_entries(tmp_path, config):
    import yaml

    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    settings.write_bytes(b"- name: gpt-4o\n  edit_format: diff\n")
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    entries = yaml.safe_load(settings.read_text(encoding="utf-8"))
    entries[0]["edit_format"] = "whole"
    entries.append({"name": "o3", "reasoning_effort": "high"})
    settings.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
    assert agents.restore(root) == []
    # Songe's entry is gone; the user's edits stay, and the file stays YAML.
    assert settings.read_bytes() == (
        b"- name: gpt-4o\n  edit_format: whole\n- name: o3\n  reasoning_effort: high\n"
    )


def test_unknown_codec_is_a_conflict_and_uninstall_carries_on(tmp_path, config):
    # A manifest from a later Songe names a codec this one does not know.
    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    settings.write_bytes(b"- name: gpt-4o\n")
    root = tmp_path / "songe"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    manifest_path = root / "agent-backups.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[str(settings)]["codec"] = "toml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    written = settings.read_bytes()
    assert agents.restore(root) == [str(settings)]
    assert settings.read_bytes() == written  # kept for the later Songe
    assert not (home / ".aider.model.metadata.json").exists()  # the rest is undone


def test_aider_drops_a_renamed_models_entries_unless_the_user_changed_them(
    tmp_path, config
):
    home = aider_home(tmp_path)
    settings = home / ".aider.model.settings.yml"
    metadata = home / ".aider.model.metadata.json"
    root = tmp_path / "songe"
    root.mkdir()
    config.exposed_models = ["songe/test", "songe/other"]
    agents.apply(agents.proposed(config, home), root)
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["openai/songe/other"]["max_output_tokens"] = 1000  # the user's edit
    metadata.write_text(json.dumps(data), encoding="utf-8")
    config.exposed_models = ["songe/renamed"]
    assert len(agents.refresh_limits(config, home, root=root)) == 2
    names = agents.load(settings).keys()
    assert "openai/songe/other" not in names and "openai/songe/renamed" in names
    kept = json.loads(metadata.read_text(encoding="utf-8"))
    assert "openai/songe/other" in kept and "openai/songe/renamed" in kept
    assert agents.restore(root) == []
    assert not settings.exists()


def test_aider_on_songe_reads_its_default_model(tmp_path, monkeypatch):
    home = aider_home(tmp_path)
    monkeypatch.delenv("AIDER_MODEL", raising=False)
    assert agents.routing(home) == {"aider": False}
    (home / ".aider.conf.yml").write_text("model: openai/songe/qwen\n", encoding="utf-8")
    assert agents.routing(home) == {"aider": True}
    monkeypatch.setenv("AIDER_MODEL", "gpt-4o")  # the environment wins
    assert agents.routing(home) == {"aider": False}
