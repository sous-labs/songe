import argparse
import asyncio
import json
import os
import platform as host_platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import webbrowser
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx

from . import NAME, __version__
from .api import LocalClient, interface_data_dir
from .config import Config
from .log import Store
from .platform.locking import acquire as acquire_lock
from .storage import private_dir
from .terminal import console, error, facts, models_table, result, status, step
from .upgrade import catch_up

PROVIDER = NAME


class UsageError(ValueError):
    """Wrong arguments: the fix is the usage line, not diagnostics."""

    def __init__(self, usage):
        super().__init__("Invalid arguments")
        self.usage = usage


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        error(message, f"usage  {NAME} --help")
        raise SystemExit(2)


def parser():
    p = ArgumentParser(
        prog=NAME, description="A local model that learns while you are idle."
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=interface_data_dir(),
        help="Private data directory",
    )
    commands = p.add_subparsers(dest="command")
    init = commands.add_parser("init", help="Connect your agents and model server")
    init.add_argument("--yes", action="store_true", help="Accept without asking")
    init.add_argument("--model", type=Path, help="Model folder or file to start with")
    init.add_argument(
        "--upstream", choices=("lmstudio", "mlx", "ollama", "llamacpp", "vllm")
    )
    init.add_argument("--name")
    init.add_argument("--upstream-url")
    init.add_argument("--ollama-models-dir", type=Path)
    init.add_argument(
        "--make-default", action="store_true", help="Make it the agent's default model"
    )
    init.add_argument(
        "--keep-default", action="store_true", help="Leave the agent's default model"
    )
    init.add_argument("--show-diff", action="store_true", help="Exact config diffs")
    init.add_argument("--launchd-target", type=Path, help=argparse.SUPPRESS)
    commands.add_parser("doctor", help="Check this installation")
    commands.add_parser("tui", help="Terminal dashboard")
    commands.add_parser("serve", help="Run the service (started automatically)")
    undo = commands.add_parser("undo", help="Return a model to its previous version")
    undo.add_argument("model", nargs="?")
    undo.add_argument("--version")
    current_status = commands.add_parser("status", help="What Songe is doing")
    current_status.add_argument(
        "--training",
        nargs="?",
        const="",
        metavar="MODEL",
        help="Show the resolved training settings for a model (default: primary model)",
    )
    models = commands.add_parser(
        "models", help="List models · auto-dream on|off"
    )
    models.add_argument("model", nargs="?")
    models.add_argument("level", nargs="?", choices=("on", "off"))
    for name, text in (
        ("open", "Dashboard in your browser"),
        ("pause", "Pause learning"),
        ("resume", "Resume learning"),
    ):
        commands.add_parser(name, help=text)
    sleep = commands.add_parser("sleep", help="Learn now instead of waiting")
    sleep.add_argument("--now", action="store_true")
    sleep.add_argument("--dry-run", action="store_true")
    forget = commands.add_parser(
        "forget", help="Delete recorded conversations (learned versions stay)"
    )
    forget.add_argument("--yes", action="store_true")
    uninstall = commands.add_parser(
        "uninstall", help="Remove Songe and restore agent settings"
    )
    uninstall.add_argument("--purge", action="store_true")
    uninstall.add_argument("--yes", action="store_true")
    uninstall.add_argument("--launchd-target", type=Path, help=argparse.SUPPRESS)
    return p


def confirm(prompt, yes=False):
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("Terminal required · use --yes")
    default = prompt == "Install?"
    answer = input(prompt + (" (Y/n) " if default else " (y/N) ")).strip().lower()
    return answer in {"y", "yes"} or (default and not answer)


def confirm_install(config, args, root=None):
    if not confirm("Install?", args.yes):
        return False
    config.replace_loaded_model = True
    if root is not None:
        # Init writes nothing before this answer (#72): the data folder and
        # its database start here, private from the first byte.
        Store(private_dir(root))
    return True


def service_manager(args):
    """Whether init can install the background service. Linux without a
    systemd user manager (containers, WSL without systemd) cannot."""
    if getattr(args, "launchd_target", None) is not None:
        return True
    from .platform import current

    return current().service().available()


def service_fact(args):
    return "starts at login" if service_manager(args) else f"none · start it with {NAME} serve"


def detect_agent_binding(home=None):
    """(model, base URL) the user's agent runs now: opencode's, then pi's."""
    from . import agents

    home = home or Path.home()
    for adapter in agents.ADAPTERS:
        model, url = adapter.binding(home)
        if model:
            return model, url
    return "", ""


def trainable_support(config, ref, platform):
    from .models.library import support

    try:
        return support(config, ref, 0, platform)[0]
    except (OSError, ValueError, AttributeError):
        return "unsupported"


def detect_agent_model(home=None):
    return detect_agent_binding(home)[0]


PRIVACY_PANE = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_FilesAndFolders"
)


def open_privacy_settings(platform=sys.platform, run=subprocess.run):
    """macOS never re-asks after a denial; take the user to the one switch."""
    if platform == "darwin" and not getattr(open_privacy_settings, "done", False):
        open_privacy_settings.done = True
        run(["open", PRIVACY_PANE], check=False, capture_output=True)


def initialize(args, root):
    from dataclasses import asdict, replace

    from . import agents
    from .backends import install as dependencies
    from .backends import resolve_kind
    from .models import ModelRef, discovery
    from .models.library import embedding_name, is_embedding
    from .platform import Rooted, current
    from .sleep.fuse import candidate_kind
    from .upstream.lmstudio import find_lms

    config = Config.load(root)
    if config.model and args.upstream and args.upstream != config.upstream:
        raise ValueError(
            "The model server changed · start over: songe uninstall --purge, then songe init"
        )
    preferred, agent_url = detect_agent_binding()
    from urllib.parse import urlparse

    from .api import local_host

    if agent_url and not local_host(urlparse(agent_url).hostname):
        agent_url = ""
    if getattr(args, "upstream_url", None):
        # Every supported server takes the OpenAI API under /v1, and the proxy
        # appends to this URL: Ollama is usually given as http://localhost:11434.
        url = args.upstream_url.rstrip("/")
        args.upstream_url = url if url.endswith("/v1") else url + "/v1"
        replace(config, upstream_url=args.upstream_url).validate()
    platform = Rooted(current(), root)
    kinds = {a.kind for a in platform.accelerators()}
    can_train = bool(kinds & {"metal", "cuda", "rocm", "xpu"})
    defaults = {
        "lmstudio": "http://127.0.0.1:1234/v1",
        "ollama": "http://127.0.0.1:11434/v1",
        "llamacpp": "http://127.0.0.1:8080/v1",
        "mlx": "http://127.0.0.1:1235/v1",
        "vllm": "http://127.0.0.1:8000/v1",
    }
    requested = args.upstream or (config.upstream if config.model else None)
    folder = discovery.lmstudio_downloads()
    config.lms = find_lms()
    if getattr(args, "ollama_models_dir", None):
        config.ollama_models_dir = str(args.ollama_models_dir.resolve())
    refs, endpoints, skipped = [], {}, []
    catalog_refs = []
    agent_upstream = (
        discovery.identify(agent_url) if agent_url and not requested else None
    )
    if config.model and not args.model:
        ref = (
            ModelRef(**(config.model_ref | {"path": Path(config.model_ref["path"])}))
            if config.model_ref
            else ModelRef(
                config.upstream,
                config.base_identifier or config.model,
                "mlx" if resolve_kind(config, platform) == "mlx" else "safetensors",
                Path(config.model),
                "",
            )
        )
        refs = [ref]
        # Models downloaded since the last init are offered to agents too
        # (#72: one was left out without a word); the bound model stays.
        from .models import registry

        try:
            catalog_refs = registry.discover(config)
        except (OSError, ValueError, KeyError, httpx.HTTPError, subprocess.SubprocessError):
            catalog_refs = []
    else:
        for name in [requested] if requested else defaults:
            url = getattr(args, "upstream_url", None) or (
                config.upstream_url
                if config.model and config.upstream == name
                else defaults[name]
            )
            if name == agent_upstream:
                url = agent_url
            try:
                if name == "lmstudio":
                    found = discovery.loaded_lmstudio(
                        discovery.lmstudio(folder=folder), config.lms, folder
                    )
                elif name == "ollama":
                    # Tags made with `ollama create` are only listed by its API.
                    found = discovery.ollama(
                        folder=config.ollama_models_dir or None, url=url
                    )
                else:
                    found = getattr(discovery, name)(url)
                refs += found
                endpoints.update({(r.upstream, r.serving_id): url for r in found})
            except (OSError, ValueError, httpx.HTTPError) as exc:
                skipped.append(f"{name}: {type(exc).__name__}: {exc}")
                if isinstance(exc, PermissionError):
                    open_privacy_settings()
                continue
        catalog_refs = list(refs)
        if args.model:
            selected = args.model.expanduser().resolve()
            refs = [r for r in refs if r.path.resolve() == selected]
            if not refs:
                name = requested or (
                    "lmstudio"
                    if selected.is_relative_to(folder)
                    else ("mlx" if "metal" in kinds else "vllm")
                )
                if selected.is_file():
                    from .models.gguf import read_gguf

                    refs = [
                        ModelRef(
                            name,
                            str(selected),
                            "gguf",
                            selected,
                            read_gguf(selected).architecture,
                        )
                    ]
                else:
                    refs = [discovery.folder_ref(selected, name)]
    skipped += [
        f"{r.upstream}: {r.path} unreadable" for r in refs if not r.path.exists()
    ]
    refs = [r for r in refs if r.path.exists()]
    if not refs:
        # Models may be usable for forwarding even when the server does not
        # expose local weight paths. This is a supported serve-only binding.
        # A server Songe does not recognize gets the generic OpenAI client.
        given = getattr(args, "upstream_url", None)
        if requested:
            candidates = [(requested, given or defaults[requested])]
        elif given:
            candidates = [(discovery.identify(given), given)]
        else:
            candidates = [(agent_upstream, agent_url)] if agent_url else []
            candidates += [(name, defaults[name]) for name in defaults]
        for name, url in candidates:
            try:
                ids = upstream_model_ids(url)
            except (OSError, ValueError, httpx.HTTPError):
                continue
            # An embedding model can never chat: not offered to agents (#72).
            ids = [i for i in ids if not embedding_name(i)]
            if ids:
                label = name or "OpenAI-compatible"
                config.upstream, config.upstream_url = name or "vllm", url
                config.exposed_models = ids
                if not config.model:
                    config.base_identifier = ids[0]
                changes = agents.proposed(
                    config, default=default_choice(args), root=root
                )
                facts(
                    [
                        ("Models", ", ".join(ids)),
                        ("Upstream", f"{label} · {url}"),
                        (
                            "Agents",
                            ", ".join(str(path) for path, *_ in changes) or "none",
                        ),
                        *missing_agents(config),
                        ("Data", str(root)),
                        ("Service", service_fact(args)),
                        ("Replace", "loaded copy when a better version is ready"),
                    ]
                )
                console().print()
                if getattr(args, "show_diff", False):
                    for *_, diff in changes:
                        print(diff)
                if not confirm_install(config, args, root):
                    return
                config.save(root)
                from .models import registry

                existing = Store(root)
                registry.migrate(config, existing)
                for model in ids:
                    registry.register(config, existing, model)
                agents.migrate(config, root, update_agents=False)
                finish_initialization(
                    config, args, root, changes,
                    lambda: agents.proposed(config, default=default_choice(args), root=root),
                )
                return
        detail = "".join(f"\n  {line}" for line in skipped)
        raise ValueError(
            "Local model unavailable · --model <path> --upstream <name>"
            + (f"\nChecked:{detail}" if detail else "")
        )

    # The file the agent's model name really is, when the server can say.
    named = discovery.lmstudio_key_path(
        config.lms, preferred, config.models_dir or folder
    )

    # A first init picks the model auto-dream starts with, so it skips one
    # this machine cannot realistically train (capacity: model size against
    # total memory; issue #73: an 8B on a Mac with little to spare).
    sizing = can_train and not config.model
    sizes = {}

    def oversized(ref):
        """(billions, limit) when ref is too big to train here, else None."""
        if not sizing:
            return None
        if ref.path not in sizes:
            from .capacity import too_big_to_train

            sizes[ref.path] = too_big_to_train(platform, ref, config)
        return sizes[ref.path]

    def preference(ref):
        identifiers = (
            ref.serving_id,
            ref.serving_id.removesuffix(":latest"),
            ref.path.name,
        )
        # A model last run through an earlier Songe keeps its "songe/" name.
        normalized = re.sub(r"[^a-z0-9]", "", preferred.removeprefix("songe/").lower())
        return (
            # A model this machine can train beats one too big for it.
            len(refs) > 1 and oversized(ref) is not None,
            not (named and ref.path.resolve() == named.resolve()),
            not any(
                normalized and normalized in re.sub(r"[^a-z0-9]", "", value.lower())
                for value in identifiers
            ),
            # Otherwise a model that can learn beats one that only serves.
            len(refs) > 1 and trainable_support(config, ref, platform) == "unsupported",
            ref.upstream != agent_upstream,
            ref.upstream,
            # Discovery lists loaded models first: with no agent match, the
            # model the server is running now wins over alphabetical order.
            refs.index(ref),
        )

    ref = min(refs, key=preference)
    if ref.upstream == "lmstudio" and not config.model:
        # A loaded runtime alias identifies an instance, not another model.
        ref = next(
            (r for r in reversed(refs) if r.path.resolve() == ref.path.resolve()),
            ref,
        )
    # Said in the plan: the model passed over for its size, or (none fits)
    # that auto-dream stays off.
    learner, skipped = ref.serving_id, None
    if oversized(ref):
        learner, skipped = None, (ref.serving_id, *oversized(ref))
    else:
        wanted = min(refs, key=lambda r: preference(r)[1:])
        if wanted.path != ref.path and oversized(wanted):
            skipped = (wanted.serving_id, *oversized(wanted))
    catalog_refs = list(
        {
            (r.upstream, r.path.resolve()): r
            for r in [r for r in catalog_refs if r.path.exists()] + refs
            # An embedding model can never chat: it is not offered to agents
            # nor listed as a model (#72).
            if r.path.resolve() == ref.path.resolve() or not is_embedding(r)
        }.values()
    )
    from .models.names import public_name

    # A model already offered under its songe/ name is not added again.
    offered = set(config.exposed_models)
    config.exposed_models = list(
        dict.fromkeys(
            [
                *config.exposed_models,
                *(
                    r.serving_id
                    for r in catalog_refs
                    if r.upstream == ref.upstream
                    and public_name(r.serving_id) not in offered
                ),
            ]
        )
    )
    if config.model:
        config.exposed_models.append(config.public_model)
    if (
        config.model
        and args.model
        and str(ref.path) not in {config.model, config.model_ref.get("path")}
    ):
        if ref.upstream != config.upstream:
            raise ValueError(
                "The model server changed · start over: songe uninstall --purge, then songe init"
            )
        preenable_model(args, root, config, ref, catalog_refs)
        return
    config.upstream = ref.upstream
    selected_url = getattr(args, "upstream_url", None) or (
        config.upstream_url
        if config.model and config.upstream == requested
        else endpoints.get((ref.upstream, ref.serving_id), defaults[ref.upstream])
    )
    if not (config.server_owned and selected_url == config.server_source_url):
        config.upstream_url = selected_url
    config.models_dir = str(folder)
    config.model_ref = asdict(ref) | {"path": str(ref.path)}
    config.base_identifier = ref.serving_id
    name = args.name or (
        config.public_model.partition("/")[2]
        if config.model
        else re.sub(r"[^a-zA-Z0-9._-]+", "-", ref.serving_id.rsplit("/", 1)[-1]).strip(
            "-"
        )
    )
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", name):
        raise ValueError("Model name: letters, digits, ., _, -")
    config.public_model = f"{PROVIDER}/{name}"
    config.artifact_kind = candidate_kind(replace(config, artifact_kind=""), platform)
    # Only an earlier install's history is read before the user accepts;
    # a first init creates nothing until then (#72).
    existing = Store(root) if (root / "experience.sqlite3").exists() else None
    bound = Config.load(root)
    if existing and (existing.version() or existing.nights()) and (
        bound.upstream_url != config.upstream_url
        or bound.public_model != config.public_model
        or bound.upstream != config.upstream
        or (bound.model_ref and bound.model_ref != config.model_ref)
        or (
            args.model
            and bound.model
            and str(ref.path) not in {bound.model, bound.model_ref.get("path")}
        )
    ):
        raise ValueError(
            "The model or its server changed · start over: songe uninstall --purge, then songe init"
        )
    if not ref.path.exists():
        raise ValueError(f"Local model unavailable: {ref.path}")
    commands = dependencies.plan(platform, root)
    if not commands:
        commands = dependencies.conversion_plan(
            ref, config.artifact_kind, platform, root
        )
    if config.upstream in {"llamacpp", "vllm", "mlx"} and not commands:
        from .upstream.supervisor import listening_server, provision_endpoint

        if not config.server_owned:
            server = listening_server(config) if config.upstream != "mlx" else None
            if server:
                config.server_command = server.cmdline()
            if config.upstream == "llamacpp" and not config.server_command:
                config.server_command = [shutil.which("llama-server") or "llama-server"]
            provision_endpoint(config)
    v2 = False
    if shutil.which("opencode"):
        result = subprocess.run(
            ["opencode", "--version"], capture_output=True, text=True, timeout=10
        )
        match = re.search(r"(\d+)\.", result.stdout)
        v2 = bool(match and int(match[1]) >= 2)
    changes = agents.proposed(
        config, default=default_choice(args), opencode_v2=v2, root=root
    )
    labels = {
        "lmstudio": "LM Studio",
        "ollama": "Ollama",
        "llamacpp": "llama.cpp",
        "mlx": "mlx-lm",
        "vllm": "vLLM",
    }
    server = labels.get(config.upstream, config.upstream)
    from .models.names import exposed

    # Every chat model on the server gets a songe/ version; init shows the
    # server and how many, not one model (live: it showed a model the
    # machine could not train, as if it were the only one).
    count = len(exposed(config))
    # Agents group them under the Songe provider (opencode shows
    # songe/songe/<name>), so the words name the provider, not a prefix (#72).
    rows = [("Models", f"{count} from {server}, listed under Songe")]
    shown = set()
    for path, _, result, _ in changes:
        adapter = agents.adapter_for(path)
        if adapter is None or adapter.name in shown:
            continue  # One row per agent, however many files it has.
        shown.add(adapter.name)
        picked = json.loads(result).get("model") if adapter.name == "opencode" else None
        if isinstance(picked, str) and picked.startswith("songe/"):
            # opencode shows it as songe/songe/<name>; the name alone is
            # what the user finds under Songe (#72).
            how = f"default {picked.removeprefix('songe/').removeprefix('songe/')}"
        else:
            how = f"choose one with {adapter.pick}"
        rows.append(("Agent", f"{adapter.name} → Songe models · {how}"))
    rows += missing_agents(config)
    rows += [("Data", str(root)), ("Service", service_fact(args))]
    if not can_train:
        rows.append(("Learning", "records only · no GPU for training"))
    elif skipped:
        from .wording import init_learning_line

        rows.append(("Learning", init_learning_line(learner, skipped)))
    else:
        rows.append(("Learning", "new conversations · starts when idle"))
    if config.server_owned:
        rows.append(("Server", f"{server} · {config.upstream_url}"))
    if commands:
        rows.append(("Download", "training runtime"))
    rows.append(("Replace", "loaded copy when a better version is ready"))
    console().print()
    facts(rows)
    console().print(f"\nremove anytime  {NAME} uninstall\n")
    if getattr(args, "show_diff", False):
        dependencies.display(commands)
        for *_, diff in changes:
            print(diff)
    if not confirm_install(config, args, root):
        print("Changes: none")
        return
    existing = Store(root)
    if commands:
        with step("Downloading runtime", "Runtime installed"):
            dependencies.install(commands, root)
    if commands:
        # Resume init with the private interpreter, keeping the one approval.
        command = [
            str(dependencies.runtime_python(root)),
            "-m",
            "songe",
            "--data-dir",
            str(root),
            "init",
            "--yes",
            "--model",
            str(ref.path),
            "--upstream",
            ref.upstream,
            "--upstream-url",
            config.upstream_url,
            "--name",
            name,
        ]
        if args.make_default:
            command.append("--make-default")
        if getattr(args, "keep_default", False):
            command.append("--keep-default")
        if args.launchd_target:
            command += ["--launchd-target", str(args.launchd_target)]
        if config.ollama_models_dir:
            command += ["--ollama-models-dir", config.ollama_models_dir]
        subprocess.run(command, check=True)
        return
    old_model = config.model
    if old_model and not args.model:
        selected = Path(old_model)
    else:
        selected = ref.path
    config.model = str(selected)
    if selected.is_dir():
        raw = json.loads((selected / "config.json").read_text(encoding="utf-8"))
        config.context_window = min(
            raw.get("text_config", raw).get("max_position_embeddings", 8192), 8192
        )
        config.max_sequence = config.context_window
    if old_model and old_model != config.model:
        raise ValueError(
            "The model's files changed · start over: songe uninstall --purge, then songe init"
        )
    config.save(root)
    if not existing.version("base"):
        existing.add_version(
            {
                "id": "base",
                "identifier": config.base_identifier,
                "artifact": str(ref.path),
                "adapter": None,
                "kind": "base"
                if config.upstream == "vllm"
                else ("gguf-full" if ref.format == "gguf" else "mlx-fused"),
                "created": time.time(),
                "night": 0,
            }
        )
        existing.swap("base", None)
    from .models import registry

    registry.migrate(config, existing)
    for found in catalog_refs:
        if found.upstream == config.upstream:
            registry.register(config, existing, found.serving_id, found)
    for found in catalog_refs:
        if found.upstream == config.upstream and oversized(found):
            # Too big to train here: using it later does not switch learning
            # on by itself (registry.observe); songe models NAME on still does.
            registry.mark_chosen(existing, found.serving_id)
    if oversized(ref):
        registry.set_auto_dream(existing, ref.serving_id, False)
    agents.migrate(config, root, update_agents=False)
    # Recording starts now; the scheduler measures this machine in the first
    # idle gap, under the same memory, battery and traffic rules as training.
    finish_initialization(
        config, args, root, changes,
        lambda: agents.proposed(config, default=default_choice(args), opencode_v2=v2, root=root),
    )


def preenable_model(args, root, config, ref, refs):
    """Adding a base never rebinds the primary learner or its undo history."""
    from dataclasses import replace

    from . import agents
    from .models import registry
    from .models.names import public_name

    proposal = replace(
        config,
        base_identifier=ref.serving_id,
        public_model=public_name(ref.serving_id),
    )
    changes = agents.proposed(proposal, default=default_choice(args), root=root)
    facts([("Model", ref.serving_id), ("Learning", "new only"), ("Data", str(root))])
    console().print("Prior history: retained")
    if getattr(args, "show_diff", False):
        for *_, diff in changes:
            print(diff)
    if not confirm_install(config, args, root):
        return
    config.save(root)
    store = Store(root)
    registry.migrate(config, store)
    for found in refs:
        if found.upstream == config.upstream:
            registry.register(config, store, found.serving_id, found)
    registry.set_learning(store, ref.serving_id, "new")
    finish_initialization(
        config, args, root, changes,
        lambda: agents.proposed(proposal, default=default_choice(args), root=root),
    )


def upstream_model_ids(url):
    with httpx.Client(trust_env=False, timeout=5) as client:
        response = client.get(url.rstrip("/") + "/models")
        response.raise_for_status()
        return [
            m["id"]
            for m in response.json().get("data", [])
            if isinstance(m.get("id"), str)
        ]


def finish_initialization(config, args, root, changes, repropose=None):
    from . import agents, launchd

    # A reinstall after `songe uninstall` (which pauses) learns again.
    Store(private_dir(root)).set("paused", False)
    managed = service_manager(args)
    if managed:
        with step("Starting service", "Service running · starts at login"):
            launchd.install(
                config, root, args.launchd_target, activate=args.launchd_target is None
            )
            if args.launchd_target is None:
                deadline = time.monotonic() + config.upstream_timeout
                while True:
                    if LocalClient(root, config).snapshot()["service_running"]:
                        break
                    if time.monotonic() >= deadline:
                        # Do not leave a service that restarts every ten seconds.
                        launchd.remove(None, activate=True, root=root)
                        raise RuntimeError(
                            f"Service did not start · is port {config.port} in use? "
                            f"· agent settings unchanged · see {root / 'songe.log'}"
                        )
                    time.sleep(config.guard_max_interval)
    # The newly started service writes agent settings itself (its own
    # entries, context lengths), so a reviewed file may have changed since
    # (live: "Agent configuration changed since review" right after "Service
    # running"). The same reviewed change is proposed again on top of what
    # the file holds now; a file nobody touched keeps its reviewed proposal.
    if repropose is not None:
        moved = {
            path for path, original, *_ in changes
            if (path.read_bytes() if path.exists() else None) != original
        }
        if moved:
            fresh = {change[0]: change for change in repropose()}
            changes = [fresh.get(change[0], change) if change[0] in moved else change for change in changes]
    # The newly started service can already have migrated its owned entries.
    changes = [
        change
        for change in changes
        if not change[0].exists()
        or agents.load(change[0]) != agents.load(change[0], change[2])
    ]
    names = sorted({agent_name(path) for path, *_ in changes})
    with step(
        "Connecting " + (", ".join(names) or "agents"),
        "Connected " + ", ".join(names) if names else "Agent settings unchanged",
    ):
        agents.apply(changes, root)
        agents.migrate(config, root)
    result(f"Ready · http://127.0.0.1:{config.port}/v1")
    console().print()
    console().print(next_step(config))
    facts(
        ([] if managed else [(f"{NAME} serve", "start Songe · no service manager here")])
        + [
            (f"{NAME} open", "dashboard"),
            (f"{NAME} status", "what it is doing"),
            (f"{NAME} pause", "stop learning"),
            (f"{NAME} uninstall", "remove everything"),
        ]
    )


def missing_agents(config):
    """Plan rows for agents Songe could set up but did not find."""
    from . import agents

    installed = agents.routing()
    rows = [
        (
            "Agent",
            # Agents are found by the settings folder they make on first start,
            # so one installed but never started is not set up yet.
            f"{agent} not set up yet · start it once, then run songe init again"
            if shutil.which(agent)
            else f"{agent} not found · run songe init again after installing it",
        )
        for agent in agents.KNOWN
        if agent not in installed
    ]
    if not installed:
        rows.append(("Other", f"any OpenAI-compatible tool · http://127.0.0.1:{config.port}/v1"))
    return rows


def next_step(config):
    """The last line of init: what is left before Songe sees any work."""
    from . import agents

    routed = agents.routing()
    left = [agent for agent, on_songe in routed.items() if not on_songe]
    if routed and not left:
        return "Use your agent as usual · learning starts when idle"
    if not routed:
        return (
            f"One step left · point your agent at http://127.0.0.1:{config.port}/v1"
            " and pick a Songe model"
        )
    picks = {a.name: a.pick for a in agents.ADAPTERS}
    return "One step left · pick a Songe model " + ", ".join(
        f"in {a} with {picks[a]}" for a in left
    )


def agent_name(path):
    """The harness a configuration file Songe edits belongs to."""
    from . import agents

    adapter = agents.adapter_for(path)
    if adapter:
        return adapter.name
    return "opencode" if "opencode" in str(path) else "pi"


def require_stopped_worker(root):
    lease = (root / "worker.lock").open("a+")
    deadline = time.monotonic() + 6
    while True:
        try:
            acquire_lock(lease)
            return lease
        except BlockingIOError:
            if time.monotonic() >= deadline:
                lease.close()
                raise RuntimeError("Worker busy · data preserved") from None
            time.sleep(0.1)


@contextmanager
def require_stopped(root, timeout=6):
    with ExitStack() as leases:
        for name in ("serve", "worker", "experience"):
            lease = leases.enter_context((root / f"{name}.lock").open("a+"))
            deadline = time.monotonic() + timeout
            while True:
                try:
                    acquire_lock(lease)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{name} running · data preserved") from None
                    time.sleep(0.1)
        yield


def default_choice(args):
    """--make-default forces, --keep-default leaves it, otherwise automatic."""
    if getattr(args, "make_default", False):
        return True
    if getattr(args, "keep_default", False):
        return False
    return None


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.data_dir.expanduser().resolve()
    try:
        if args.command == "status" and args.training is not None:
            print_training_status(root, args.training)
            return 0
        config = Config.load(root)
        command = args.command
        local = LocalClient(root, config)
        if command not in {"init", "uninstall", "serve"} and catch_up(root, config):
            # Already printed once by catch_up; don't repeat it in the status.
            os.environ["SONGE_NOTICE_SHOWN"] = "1"
        if command == "init":
            from .storage import foreign_entries

            if not (root / "config.toml").exists() and foreign_entries(root):
                raise ValueError(
                    f"Data directory has other files · {root}"
                    " · choose an empty or new --data-dir"
                )
        # Init creates its data only once the user accepts (#72).
        if command in {"serve", "uninstall"} or (
            command == "sleep" and args.dry_run
        ):
            store = Store(private_dir(root))
        if command == "init":
            initialize(args, root)
        elif command == "serve":
            if not config.model and not config.exposed_models:
                raise ValueError("Uninitialized · songe init")

            with (root / "serve.lock").open("a+") as probe:
                try:
                    acquire_lock(probe)
                except BlockingIOError:
                    print(
                        f"{NAME} running"
                        f" · http://127.0.0.1:{config.port}\n"
                        f"status  {NAME} status · {NAME} open"
                    )
                    return 0
            from . import agents

            agents.migrate(config, root)
            import uvicorn

            from .proxy import create_app
            from .schedule import ModelScheduler
            from .telemetry import configure
            from .upstream import create

            configure(root)
            if hasattr(signal, "SIGUSR2"):
                import faulthandler

                # `kill -USR2 <pid>` writes every thread's stack for diagnosis.
                faulthandler.register(
                    signal.SIGUSR2,
                    open(root / "stacks.txt", "a", encoding="utf-8"),
                    all_threads=True,
                )
            upstream = create(config, root)
            scheduler = ModelScheduler(config, store, upstream)
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(config, store, scheduler=scheduler),
                    host="127.0.0.1",
                    port=config.port,
                    access_log=False,
                    log_level="warning",
                )
            )
            scheduler.shutdown = lambda: setattr(server, "should_exit", True)
            server.run()
            if getattr(scheduler, "upgrade_exit", False):
                # Non-zero so systemd (Restart=on-failure) restarts it too.
                return 3
        elif command == "doctor":
            if doctor(config, root, local):
                return 1
        elif command == "tui":
            from .tui import run

            run(root)
        elif command in {"pause", "resume"}:
            local.control(command)
            result("Learning paused" if command == "pause" else "Learning resumed")
        elif command == "open":
            if not local.running():
                error("Songe is stopped", f"run  {NAME} serve")
                return 1
            url = f"http://127.0.0.1:{config.port}/"
            webbrowser.open(url)
            result(f"Dashboard · {url}")
        elif command == "undo":
            local.control(
                "undo",
                **({"model": args.model} if args.model else {}),
                **({"version": args.version} if args.version else {}),
            )
            result(
                "Previous version restored" + (f" · {args.model}" if args.model else "")
            )
        elif command == "models":
            if args.model and args.level:
                # What it learns from is Songe's decision; this only says
                # whether Dreams start on their own.
                local.control("auto_dream", model=args.model, on=args.level == "on")
                result(f"{args.model} · auto-dream {args.level}")
                console().print()
            elif args.model or args.level:
                raise UsageError(f"{NAME} models <model> on|off")
            data = local.snapshot()
            print_models(data.get("models", []), data=data)
        elif command == "sleep":
            if args.dry_run:
                from .sleep.worker import isolated

                print(json.dumps(isolated(config, store, dry_run=True), indent=2))
            else:
                # --now is the default; the flag stays for older scripts.
                reply = local.control("sleep_now") or {}
                # Never a request that silently does nothing: what it waits on.
                waiting = reply.get("waiting") if isinstance(reply, dict) else None
                result(
                    f"Learning requested · not started yet · {waiting} · {NAME} status"
                    if waiting
                    else f"Learning requested · {NAME} status"
                )
        elif command == "forget":
            if confirm("Delete experience and examples?", args.yes):
                local.control("forget", confirm=True)
                result(
                    f"Conversations deleted · learning paused · {NAME} resume to continue"
                )
        elif command == "uninstall":
            from . import agents, launchd

            store.set("paused", True)
            launchd.remove(
                args.launchd_target, activate=args.launchd_target is None, root=root
            )
            local.control("shutdown")
            purged = False
            # A streaming answer or a version being applied can take a while
            # to end; six seconds left the service removed but agents wired.
            with require_stopped(root, timeout=30):
                conflicts = agents.restore(root)
                if args.purge and confirm(
                    "Delete Songe-created models and all Songe data?", args.yes
                ):
                    from .upstream.cleanup import purge

                    asyncio.run(purge(config, store))
                    from .artifacts.storage import purge_roots

                    purge_roots(store)
                    with store.connect() as db:
                        versions = [
                            dict(r) for r in db.execute("SELECT * FROM versions")
                        ]
                    for version in versions:
                        artifact = Path(version["artifact"])
                        from .artifacts.runtime import remove_owned

                        remove_owned(artifact, store.get("installation_id"))
                    # Windows cannot delete open lease files. Remove the
                    # config/data while locked, then close the empty leases.
                    # An agent setting you changed since init is left as it
                    # is; only the backups of the originals stay behind.
                    from .storage import owned_entry

                    keep = {"serve.lock", "worker.lock", "experience.lock"} | (
                        {"backups", "agent-backups.json"} if conflicts else set()
                    )
                    # This process's own open database handles go first.
                    from .log import close_connections

                    close_connections()
                    for path in root.iterdir():
                        if owned_entry(path.name) and path.name not in keep:
                            shutil.rmtree(path) if path.is_dir() else path.unlink()
                    purged = True
            kept = []
            if purged:
                from .storage import foreign_entries

                for name in ("serve", "worker", "experience"):
                    (root / f"{name}.lock").unlink(missing_ok=True)
                kept = foreign_entries(root) or (["backups"] if conflicts else [])
                if not kept:
                    root.rmdir()
            result("Service removed")
            result(
                "Agent configurations restored"
                if not conflicts
                else "Agent configurations restored · conflicts preserved"
            )
            if purged:
                result("Songe data removed")
                if kept:
                    facts(
                        [
                            (
                                "Kept",
                                (
                                    "backups of your agent settings"
                                    if conflicts
                                    else "files not created by Songe"
                                )
                                + f" · {root}",
                            )
                        ]
                    )
            if conflicts:
                facts(
                    [
                        ("Preserved", ", ".join(conflicts)),
                        ("Originals", "retained in backups"),
                    ]
                )
            result("Songe uninstalled")
            if not purged:
                facts(
                    [
                        ("Data kept", str(root)),
                        ("Delete", f"{NAME} uninstall --purge"),
                    ]
                )
        else:
            print_status(local.snapshot())
    except (
        OSError,
        sqlite3.Error,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        httpx.HTTPError,
    ) as exc:
        error(exc, f"usage  {exc.usage}" if isinstance(exc, UsageError) else None)
        return 1
    return 0


def print_models(models, *, data=None):
    models_table(models, data=data)


def print_status(data):
    status(data)


def print_training_status(root, model):
    from .api import ReadOnlyStore
    from .config import TRAINING_KEYS
    from .models.registry import entries, model_config_with_keys
    from .terminal import training_settings

    config, installation_keys = Config.load_with_keys(root)
    store = ReadOnlyStore(root)
    registered = [row["id"] for row in entries(store)]
    model_keys, ignored = (), ()
    if registered:
        primary = store.get("primary_model")
        model = model or (primary if primary in registered else registered[0])
        config, model_keys, ignored = model_config_with_keys(config, store, model)
    elif model:
        raise ValueError(f"Unknown model: {model}")

    sources = (
        dict.fromkeys(TRAINING_KEYS, "default")
        | dict.fromkeys(installation_keys, "installation")
        | dict.fromkeys(model_keys, "per-model")
    )
    training_settings(
        config.public_model if config.model_key or config.model else None,
        ((key, getattr(config, key), sources[key]) for key in TRAINING_KEYS),
        ignored,
    )


def measured_hardware():
    from .platform import current

    device = current()
    memory = device.memory()
    accelerators = device.accelerators()
    return (
        device.name,
        memory.total,
        memory.available,
        ", ".join(
            f"{a.kind} {a.name}"
            # Apple GPUs share system memory, already shown on the Memory line.
            + (
                ""
                if a.kind == "metal" or not a.memory_total
                else f" · {a.memory_free / 1e9:.1f} of {a.memory_total / 1e9:.1f} GB free"
            )
            for a in accelerators
        )
        or "none · CPU serving",
    )


def last_errors(root):
    """The recorded learning error and retry count, per model when scoped."""
    from .api import ReadOnlyStore
    from .wording import stamp

    try:
        with ReadOnlyStore(root).connect() as db:
            rows = db.execute(
                "SELECT key, value FROM state WHERE key IN ('child_error', 'failures')"
                " OR key LIKE 'model:%:child_error' ORDER BY key"
            ).fetchall()
    except sqlite3.Error:
        return []
    lines = []
    for key, raw in rows:
        value = json.loads(raw)
        if not value:
            continue
        model = key.removeprefix("model:").rpartition(":")[0]
        if key.endswith("child_error"):
            text = str(value).removesuffix(" · songe doctor")
            lines.append(f"Last error: {text}" + (f" · {model}" if model else ""))
        elif isinstance(value, dict) and value.get("count"):
            at = value.get("at")
            lines.append(
                f"Failures: {value['count']}"
                + (f" · last {stamp(at)}" if isinstance(at, (int, float)) else "")
            )
    return lines


def doctor(config, root, local):
    from . import agents

    problems = []
    try:
        name, total, available, accelerator = measured_hardware()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        name, total, available, accelerator = (
            sys.platform,
            None,
            None,
            f"probe failed: {exc}",
        )
    print(
        f"Platform: {name}, {host_platform.machine()}, Python {host_platform.python_version()}; "
        + (
            "supported"
            if name in {"darwin", "linux", "win32", "windows"}
            else "unsupported"
        )
    )
    print(f"Accelerator: {accelerator}")
    print(
        f"Memory: {available / 1e9:.1f} of {total / 1e9:.1f} GB free"
        if available is not None and total is not None
        else "Memory: —"
    )
    tools = [
        f"{name}={path}"
        for name in ("lms", "ollama", "llama-server", "mlx_lm", "vllm")
        if (path := shutil.which(name))
    ]
    # The user's own server, not one Songe runs only while it needs it.
    server = config.server_source_url or config.upstream_url
    try:
        with httpx.Client(trust_env=False, timeout=1) as client:
            response = client.get(server.rstrip("/") + "/models")
            response.raise_for_status()
            models = response.json()["data"]
            if not isinstance(models, list):
                raise ValueError("Model listing: list required")
        endpoint = f"{len(models)} models at {server}"
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        endpoint = f"{server} unavailable · {type(exc).__name__}"
        problems.append("upstream")
    print(
        f"Upstreams: {len(tools)} executables"
        + (f" ({', '.join(tools)})" if tools else "")
        + f"; {endpoint}"
    )
    configured, invalid = [], []
    try:
        for kind, path in agents.targets():
            adapter = agents.adapter_for(path)
            try:
                found = adapter.status(path, agents.load(path, path.read_bytes()), config)
            except (OSError, ValueError, AttributeError):
                invalid.append(f"{kind}: unreadable configuration")
                continue
            if found == "configured":
                configured.append(kind)
            elif found == "missing":
                invalid.append(f"{kind}: no {NAME} provider")
            elif found == "differs":
                invalid.append(f"{kind}: provider URL differs")
    except ValueError as exc:
        invalid.append(str(exc))
    problems += invalid
    print(
        f"Agents: {len(configured)} configured"
        + (f" ({', '.join(configured)})" if configured else f"; {NAME} init")
        + (f"; {', '.join(invalid)}" if invalid else "")
    )
    try:
        data = local.snapshot()
        print(
            f"Service: running on 127.0.0.1:{config.port}, {data['state']}"
            if data["service_running"]
            else f"Service: stopped · 127.0.0.1:{config.port} · {NAME} serve"
        )
        if not data["service_running"]:
            problems.append("service")
        if data.get("upgrade_notice") and not os.environ.get("SONGE_NOTICE_SHOWN"):
            print(data["upgrade_notice"])
    except (OSError, sqlite3.Error, ValueError, RuntimeError, httpx.HTTPError) as exc:
        print(f"Service: unhealthy on 127.0.0.1:{config.port} ({exc})")
        problems.append("service")
    for line in last_errors(root):
        print(line)
    print(f"Log: {root / 'songe.log'}")
    path = Path(
        config.models_dir
        if config.upstream == "lmstudio" and config.models_dir
        else config.ollama_models_dir
        if config.upstream == "ollama" and config.ollama_models_dir
        else config.model_ref.get("path") or config.model or root
    )
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        disk = shutil.disk_usage(path)
        model = Path(config.model) if config.model else None
        artifact_bytes = (
            None
            if model is None or not model.exists()
            # A GGUF model is one file; rglob finds nothing under a file.
            else model.stat().st_size
            if model.is_file()
            else sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
        )
        size = (
            f"; model {artifact_bytes / 1e9:.1f} GB"
            if artifact_bytes is not None
            else f"; model path missing · {NAME} init"
            if config.model
            else f"; uninitialized · {NAME} init"
        )
        print(
            f"Disk: {disk.free / 1e9:.1f} of {disk.total / 1e9:.1f} GB free at {path}{size}"
        )
    except OSError as exc:
        print(f"Disk: unavailable · {path} · {exc}")
    # Recordings grow in the data folder, which can be on another disk than
    # the models (an external drive); show its free space too.
    try:
        from .artifacts.storage import volume

        if volume(root) != volume(path):
            data = shutil.disk_usage(root if root.exists() else volume(root))
            print(
                f"Data disk: {data.free / 1e9:.1f} of {data.total / 1e9:.1f} GB free at {root}"
            )
    except OSError as exc:
        print(f"Data disk: unavailable · {root} · {exc}")
    line = recordings_line(config, root)
    if line:
        print(line)
    return problems


def recordings_line(config, root):
    """How much the recorded conversations use, against their limit
    (recordings_limit_gb); read without changing the store."""
    from .api import ReadOnlyStore
    from .recordings import limit_bytes, sizes
    from .wording import recordings_line as line

    if not (root / "experience.sqlite3").exists():
        return None
    try:
        used = sizes(ReadOnlyStore(root))[0]
    except sqlite3.Error:
        return None
    return line(used, limit_bytes(config))
