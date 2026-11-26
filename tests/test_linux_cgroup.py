"""Guard accounting on shared Linux GPU/container hosts."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from songe.memory import GuardPolicy, MemoryPressure, admit, guarded_run
from songe.platform.linux import Linux


@pytest.fixture
def controller(tmp_path):
    def make(kind="v2", group="/", mount_root="/"):
        proc, sysfs = tmp_path / "proc", tmp_path / "sys"
        (proc / "self").mkdir(parents=True, exist_ok=True)
        (proc / "pressure").mkdir(exist_ok=True)
        (proc / "meminfo").write_text(
            "MemTotal: 1000000 kB\nMemAvailable: 900000 kB\n"
            "SwapTotal: 1000 kB\nSwapFree: 0 kB\nZswap: 500 kB\n"
        )
        (proc / "pressure/memory").write_text("some avg10=90.00\n")
        (proc / "vmstat").write_text("pswpout 100000\n")
        mount = sysfs / "fs/cgroup" / ("memory" if kind == "v1" else "")
        mount.mkdir(parents=True, exist_ok=True)
        (proc / "self/cgroup").write_text(
            f"2:memory:{group}\n" if kind == "v1" else f"0::{group}\n"
        )
        (proc / "self/mountinfo").write_text(
            f"22 1 0:20 {mount_root} /sys/{mount.relative_to(sysfs)} rw - "
            + ("cgroup cgroup rw,memory\n" if kind == "v1" else "cgroup2 cgroup rw\n")
        )
        names = (
            ("memory.limit_in_bytes", "memory.usage_in_bytes")
            if kind == "v1"
            else ("memory.max", "memory.current")
        )
        (mount / names[0]).write_text("10000000")
        (mount / names[1]).write_text("2000000")
        if kind == "v1":
            (mount / "memory.stat").write_text("total_swap 123\n")
        else:
            (mount / "memory.swap.current").write_text("123")
            (mount / "memory.pressure").write_text("some avg10=0.00\n")
        platform = Linux(proc=proc, sysfs=sysfs)
        platform.accelerators = lambda: []
        return platform, mount, names

    return make


@pytest.mark.parametrize("kind", ["v1", "v2"])
@pytest.mark.parametrize(
    "group,root", [("/", "/"), ("/docker/test", "/docker/test"), ("/docker/test", "/")]
)
def test_container_headroom_and_sensors_are_scoped(controller, kind, group, root):
    platform, _, _ = controller(kind, group, root)
    sample = admit(platform, required_bytes=7000000)
    assert sample.total == 10000000
    assert sample.available == 8000000
    assert sample.pressure == "normal"
    assert sample.swap_used == 123
    assert sample.swap_out is None and sample.compressed is None
    assert sample.psi == (0 if kind == "v2" else None)
    with pytest.raises(MemoryPressure) as caught:
        admit(platform, required_bytes=9000000)
    assert caught.value.reason["rule"] == "admission_prediction"


def test_nested_parent_limit_includes_sibling_usage(controller):
    platform, mount, names = controller(group="/child")
    child = mount / "child"
    child.mkdir()
    (child / names[0]).write_text("max")
    (child / names[1]).write_text("1000000")
    (mount / names[1]).write_text("9500000")
    assert platform.memory().available == 500000
    assert platform.memory().total == 10000000


@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_exhausted_container_is_critical(controller, kind):
    platform, mount, names = controller(kind)
    (mount / names[1]).write_text("10000001")
    with pytest.raises(MemoryPressure):
        admit(platform)


def test_cgroup_psi_warning_allows_guarded_conversion_but_critical_stops(
    controller, monkeypatch
):
    from songe import memory

    monkeypatch.setattr(memory, "process_tree_bytes", lambda *a: 0)
    monkeypatch.setattr(memory, "process_tree_swap_bytes", lambda *a: None)
    platform, mount, _ = controller()
    psi = mount / "memory.pressure"
    psi.write_text("some avg10=0.10\n")
    # Production conversion uses this same guard. Warning is a healthy baseline.
    result = guarded_run(
        [sys.executable, "-c", "print('prepared')"],
        platform=platform,
        policy=GuardPolicy(baseline_seconds=0),
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.stdout.strip() == "prepared"
    psi.write_text("some avg10=1.00\n")
    with pytest.raises(MemoryPressure):
        admit(platform)


def test_unlimited_container_retains_host_capacity(controller):
    platform, mount, _ = controller()
    (mount / "memory.max").write_text("max")
    sample = platform.memory()
    assert (sample.total, sample.available) == (1000000 * 1024, 900000 * 1024)
    assert sample.swap_used == 123 and sample.psi == 0


def test_known_tool_protocol_missing_call_is_invalid_and_evaluable():
    from songe.sleep.evaluate import health
    from songe.sleep.gate import decide

    tokenizer = SimpleNamespace(chat_template="Use <tool_call>JSON</tool_call>")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
            },
        }
    ]
    invalid = health("I cannot find that", tokenizer, tools, True)
    valid = health(
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>',
        tokenizer,
        tools,
        True,
    )
    assert invalid["tool_supported"] == valid["tool_supported"] == 1
    assert invalid["tool_valid"] == 0 and valid["tool_valid"] == 1
    before = [
        dict(id=i, conversation=str(i), kl=1, drift=0, tool_step=True, **invalid)
        for i in range(8)
    ]
    after = [r | dict(kl=0.5) for r in before]
    assert decide(before, after, min_heldout=8, drift_limit=0.1).accepted
    # The fix must still refuse a validity regression in that same protocol.
    before = [r | valid for r in before]
    assert (
        decide(before, after, min_heldout=8, drift_limit=0.1).reason
        == "Tool-call validity dropped"
    )


def test_host_controller_root_has_no_limit_files(controller):
    platform, mount, names = controller(group="/child")
    child = mount / "child"
    child.mkdir()
    for name, value in zip(names, ("10000000", "2000000"), strict=True):
        (child / name).write_text(value)
        (mount / name).unlink()
    assert platform.memory().available == 8000000


def test_cgroup_namespace_relative_parent_path_stays_at_mount(controller):
    platform, mount, _ = controller(group="/../../docker/test")
    assert platform.memory_cgroup()[1] == mount
    assert platform.memory().available == 8000000


@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_missing_leaf_limit_does_not_advertise_host_capacity(controller, kind):
    platform, mount, names = controller(kind, group="/child")
    child = mount / "child"
    child.mkdir()
    (child / names[1]).write_text("2000000")
    with pytest.raises(OSError):
        admit(platform)


def test_unlimited_host_controller_root_has_no_limit(controller):
    platform, mount, names = controller()
    (mount / names[0]).unlink()
    sample = platform.memory()
    assert (sample.total, sample.available) == (1000000 * 1024, 900000 * 1024)


@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_page_cache_of_files_read_once_counts_as_available(controller, kind):
    # A container that downloaded a model: usage is nearly all inactive page cache.
    platform, mount, names = controller(kind)
    (mount / names[1]).write_text("9800000")
    name = "inactive_file" if kind == "v2" else "total_inactive_file"
    (mount / "memory.stat").write_text(f"{name} 9000000\nfile 9500000\n")
    assert platform.memory().available == 9200000
    assert admit(platform, required_bytes=7000000).pressure == "normal"
