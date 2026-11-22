import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.upstream.lmstudio import LMStudio


@pytest.fixture(autouse=True)
def measured_memory(monkeypatch):
    from songe.platform import Memory
    from songe.upstream import protection

    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [],
        ),
    )


class FakeLMS(LMStudio):
    """Records lms invocations; the index shows a new folder only after a
    delay, as LM Studio does after a fused version is written."""

    def __init__(self, models_dir, appear_after=2, running=None):
        super().__init__(SimpleNamespace(models_dir=str(models_dir)))
        self.calls, self.listings, self.appear_after = [], 0, appear_after
        self.ps = running or []

    async def listing(self):
        return list(self.ps)

    async def request_load(self, key, identifier, context=None):
        self.contexts = getattr(self, "contexts", []) + [context]
        await self.command("load", key, "--identifier", identifier, "-y")
        self.completed(identifier, self.ps[-1]["instanceReference"])

    def sdk(self):
        owner = self

        class Client:
            async def __aenter__(self):
                self.llm = self
                return self

            async def __aexit__(self, *args):
                pass

            async def unload(self, identifier):
                await owner.command("unload", identifier)

        return Client()

    async def command(self, *args):
        self.calls.append(args)
        if args[0] == "load":
            self.ps.append(
                {
                    "identifier": args[3],
                    "path": "songe/granite/night-1"
                    if args[1] == "night-1"
                    else "lmstudio-community/granite-4.2-3b-MLX-4bit",
                    "instanceReference": f"instance-{len(self.calls)}",
                }
            )
        if args[0] == "unload":
            self.ps[:] = [m for m in self.ps if m["identifier"] != args[1]]
        if args[:2] == ("ps", "--json"):
            return json.dumps(self.ps)
        if args[:2] == ("ls", "--json"):
            self.listings += 1
            models = [
                {
                    "modelKey": "granite-4.2-3b-mlx",
                    "path": "lmstudio-community/granite-4.2-3b-MLX-4bit",
                }
            ]
            if self.listings >= self.appear_after:
                models.append({"modelKey": "night-1", "path": "songe/granite/night-1"})
            return json.dumps(models)
        return ""


def test_load_uses_model_key_not_folder_path(tmp_path):
    lms = FakeLMS(tmp_path)
    version = {
        "artifact": str(tmp_path / "songe/granite/night-1"),
        "identifier": "songe/granite/night-1",
    }
    asyncio.run(lms.load(version))
    load = [c for c in lms.calls if c[0] == "load"]
    assert load == [("load", "night-1", "--identifier", "songe/granite/night-1", "-y")]
    assert "songe/granite/night-1" in lms.loaded


def test_load_resolves_existing_base_key(tmp_path):
    lms = FakeLMS(tmp_path, appear_after=99)
    base = {
        "artifact": str(tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"),
        "identifier": "songe/granite/base",
    }
    asyncio.run(lms.load(base))
    assert (
        "load",
        "granite-4.2-3b-mlx",
        "--identifier",
        "songe/granite/base",
        "-y",
    ) in lms.calls


def test_unindexed_folder_fails_closed(tmp_path):
    lms = FakeLMS(tmp_path, appear_after=99)
    with pytest.raises(RuntimeError, match="not indexed"):
        asyncio.run(
            lms.model_key(Path("songe/granite/missing"), timeout=0.05, interval=0.01)
        )
    assert not lms.loaded


def test_artifact_outside_models_dir_is_refused(tmp_path):
    lms = FakeLMS(tmp_path / "models")
    with pytest.raises(ValueError, match="inside downloadsFolder"):
        asyncio.run(
            lms.load({"artifact": str(tmp_path / "elsewhere"), "identifier": "x"})
        )


def test_already_loaded_same_weights_is_reused(tmp_path):
    # Undo returns to a version that is still loaded: LM Studio would refuse
    # a duplicate identifier, so the load must be a no-op.
    lms = FakeLMS(
        tmp_path,
        running=[
            {
                "identifier": "songe/granite/base",
                "path": "lmstudio-community/granite-4.2-3b-MLX-4bit",
            }
        ],
    )
    base = {
        "artifact": str(tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"),
        "identifier": "songe/granite/base",
    }
    asyncio.run(lms.load(base))
    assert not [c for c in lms.calls if c[0] == "load"]
    assert "songe/granite/base" not in lms.loaded


def test_identifier_bound_to_other_weights_fails_closed(tmp_path):
    lms = FakeLMS(
        tmp_path,
        running=[
            {"identifier": "songe/granite/night-1", "path": "songe/granite/night-0"}
        ],
    )
    version = {
        "artifact": str(tmp_path / "songe/granite/night-1"),
        "identifier": "songe/granite/night-1",
    }
    with pytest.raises(RuntimeError, match="already serves"):
        asyncio.run(lms.load(version))
    assert not [c for c in lms.calls if c[0] == "load"]


async def test_parking_keeps_every_unowned_identifier_and_context(tmp_path):
    listing = [
        {"identifier": "user", "path": "pub/model", "memoryBytes": 100},
        {
            "identifier": "songe/local/night-1",
            "path": "songe/local-night-1",
            "memoryBytes": 300,
        },
    ]
    lms = FakeLMS(tmp_path, running=listing)
    assert await lms.loaded_bytes() == 0
    assert await lms.park() == []
    await lms.unload({"identifier": "songe/local/night-1"})
    assert not any(c[0] == "unload" for c in lms.calls)
    assert lms.ps == listing


async def test_legacy_broad_snapshot_does_not_reload_user_models(tmp_path):
    lms = FakeLMS(tmp_path)
    await lms.restore(
        [{"identifier": "user", "key": "key", "path": "pub/model"}], set()
    )
    assert not any(c[0] == "load" for c in lms.calls)


def test_reload_uses_the_context_the_model_last_ran_with(tmp_path):
    from songe.log import Store

    models = tmp_path / "models"
    (models / "pub/Model").mkdir(parents=True)
    upstream = LMStudio(
        SimpleNamespace(models_dir=str(models)), store=Store(tmp_path / "d")
    )
    upstream.store.set("served_context_ids", {"pub/Model": 32768})
    base = {"identifier": "pub/Model", "artifact": str(models / "pub/Model")}
    assert upstream.remembered_context(base) == 32768
    # A learned version has never run; it inherits its original model's.
    learned = {
        "identifier": "songe/Model/v1",
        "artifact": str(models / "songe/Model-v1"),
        "base": str(models / "pub/Model"),
    }
    assert upstream.remembered_context(learned) == 32768
    assert upstream.remembered_context({"identifier": "other"}) is None


def test_only_the_users_own_loads_are_remembered(tmp_path):
    import asyncio

    upstream = LMStudio(SimpleNamespace(models_dir=str(tmp_path)))
    upstream.owned = {"pub/A": {"runtime_instance": "ref-songe"}}

    async def listing():
        return [
            {
                "identifier": "pub/A",
                "path": "pub/A",
                "instanceReference": "ref-songe",
                "contextLength": 8192,
            },
            {
                "identifier": "pub/B",
                "path": "pub/B",
                "instanceReference": "ref-user",
                "contextLength": 65536,
            },
        ]

    upstream.listing = listing
    served = asyncio.run(upstream.served_contexts())
    assert served == {"pub/A": 8192, "pub/B": 65536}
    assert upstream.user_contexts == {"pub/B": 65536}


@pytest.mark.parametrize(
    "supported, remembered, expected",
    [
        # LM Studio's default (8192) cannot hold opencode's prompt (#65).
        (131072, None, 32768),
        (4096, None, 4096),  # Never more than the model supports.
        (131072, 16384, 16384),  # The user's own setting wins.
        (None, None, None),  # Unknown limit: leave it to LM Studio.
    ],
)
def test_first_load_has_room_for_an_agents_prompt(
    tmp_path, supported, remembered, expected
):
    from songe.log import Store

    class Listed(FakeLMS):
        async def command(self, *args):
            out = await super().command(*args)
            if args[:2] == ("ls", "--json") and supported:
                models = json.loads(out)
                for model in models:
                    model["maxContextLength"] = supported
                return json.dumps(models)
            return out

    lms = Listed(tmp_path, appear_after=99)
    lms.store = Store(tmp_path / "d")
    if remembered:
        lms.store.set("served_context_ids", {"songe/granite/base": remembered})
    base = {
        "artifact": str(tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"),
        "identifier": "songe/granite/base",
    }
    asyncio.run(lms.load(base))
    assert lms.contexts == [expected]
