import copy
import re
import xml.etree.ElementTree as ET

import pytest
from textual.widgets import Button, DataTable, Static

from songe.tui import SongeApp


def data():
    return {
        "model": "songe/test",
        "state": "awake",
        "reason": "service stopped · songe serve",
        "paused": False,
        "slice": None,
        "nights": [],
        "scoreboard": [],
        "service_running": False,
    }


class Client:
    url = "http://127.0.0.1:7613"

    def __init__(self):
        self.data = data()
        self.actions = []

    def snapshot(self):
        return copy.deepcopy(self.data)

    def control(self, action, **payload):
        self.actions.append((action, payload) if payload else action)
        if action in {"pause", "resume"}:
            self.data["paused"] = action == "pause"
        if action == "model":
            model = next(m for m in self.data["models"] if m["id"] == payload["model"])
            model["learning_level"] = payload["level"]


@pytest.mark.parametrize("size", [(110, 48), (42, 55)])
async def test_screen_offline_keys_and_narrow_layout(monkeypatch, size):
    client = Client()
    opened = []
    monkeypatch.setattr("songe.tui.webbrowser.open", opened.append)
    app = SongeApp(client=client)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert "Songe is stopped" in str(app.query_one("#state", Static).render())
        assert not app.query("#scores, #nights, #versions, #learning-summary")
        await pilot.press("p", "p", "u", "o")
        assert client.actions == ["pause", "resume", "undo"]
        assert opened == [client.url + "/"]
        assert app.screen.has_class("narrow") == (size[0] < 80)
        await pilot.press("q")


@pytest.mark.parametrize("reason", ["Learning", "battery low · checkpoint pending"])
async def test_live_screen_keeps_header_without_global_sections(reason):
    client = Client()
    client.data.update(
        {
            "state": "settling",
            "reason": reason,
            "service_running": True,
            "slice": {"phase": "train", "units_done": 7},
            "nights": [
                {
                    "id": 1,
                    "status": "accepted",
                    "progress": 1,
                    "phase": "swap",
                    "record": {"steps_learned": 32, "reason": "[red]literal[/red]"},
                }
            ],
            "scoreboard": [
                {
                    "day": "2026-11-10",
                    "repeat_error_rate": 0.25,
                    "steps_per_task": 4,
                    "tool_call_valid_rate": 0.9,
                    "general_drift": 0.003,
                }
            ],
        }
    )
    app = SongeApp(client=client)
    async with app.run_test(size=(100, 55)) as pilot:
        await pilot.pause()
        assert not app.query("#scores, #nights, #versions, #night-detail")
        assert ("Learning" if reason == "Learning" else "Waiting for power") in str(
            app.query_one("#state", Static).render()
        )
        assert len(str(app.query_one("#state", Static).render()).splitlines()) == 2
        assert not app.query("#rings")
        # Refresh preserves the model table's focus.
        app.query_one("#models").focus()
        await pilot.pause()
        await app.refresh_data()
        await pilot.pause()
        assert app.focused.id == "models"


async def test_model_switch_undo_and_active_training():
    client = Client()
    client.data["service_running"] = True
    client.data["models"] = [
        {
            "id": "one",
            "enabled": True,
            "status": "learning",
            "reason": "",
            "pending": 10,
            "last_result": "Accepted",
            "learning": {
                "conversations": 2,
                "since": 100,
                "signals": {
                    "tool_error_fixed": 1,
                    "user_correction": 0,
                    "retry_success": 0,
                },
                "topics": ["pytest"],
                "sources": [],
                "last": None,
            },
        },
        {
            "id": "two",
            "enabled": False,
            "status": "off",
            "reason": "",
            "pending": 12,
            "last_result": None,
        },
        {
            "id": "three",
            "enabled": True,
            "status": "unsupported",
            "reason": "Measured memory limit",
            "pending": 12,
            "last_result": None,
        },
    ]
    client.data["training"] = {
        "model": "one",
        "phase": "train",
        "progress": 0.3,
        "data_progress": {
            "phase": "train",
            "completed": 3,
            "total": 10,
            "unit": "samples",
        },
    }
    app = SongeApp(client=client)
    async with app.run_test(size=(110, 55)) as pilot:
        await pilot.pause()
        models = app.query_one("#models", DataTable)
        assert models.row_count == 3
        assert not app.query_one("#model-details").display
        assert not app.query_one("#learning-control").display
        models.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert app.query_one("#model-details").display
        assert "Data         2 conversations" in str(
            app.query_one("#model-details", Static).render()
        )
        assert "Training one" in str(app.query_one("#state", Static).render())
        assert "3/10 samples" in str(app.query_one("#state", Static).render())
        models.focus()
        await pilot.pause()
        models.move_cursor(row=1)
        await pilot.press("enter", "l")
        await pilot.pause()
        # The same rows and quiet defaults as the dashboard.
        assert str(app.query_one("#model-details", Static).render()) == (
            "Now          Nothing running\n"
            "Result       No update yet\n"
            "Data         No conversations yet · pick a songe/… model in your agent\n"
            "Next         First update after your first conversations\n"
            "Versions     Original · In use"
        )
        assert client.actions[-1] == ("model", {"model": "two", "level": "new"})
        assert app.query_one("#learning-new", Button).has_class("selected")
        await pilot.click("#learning-all")
        await pilot.pause()
        assert client.actions[-1] == ("model", {"model": "two", "level": "all"})
        assert app.query_one("#learning-all", Button).has_class("selected")
        await pilot.click("#learning-off")
        await pilot.pause()
        assert client.actions[-1] == ("model", {"model": "two", "level": "off"})
        await pilot.press("u")
        await pilot.pause()
        assert client.actions[-1] == ("undo", {"model": "two"})
        models.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert not app.query_one("#model-details").display
        models.move_cursor(row=0)
        await pilot.pause()
        assert app.query_one("#model-details").display
        await app.refresh_data()
        assert app.query_one("#model-details").display
        await pilot.press("enter")
        assert not app.query_one("#model-details").display


async def test_details_and_training_color_survive_selected_row(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    client = Client()
    client.data.update(
        service_running=True,
        training={"model": "one", "phase": "train"},
        models=[
            {
                "id": "one",
                "name": "Qwen3.5-4B",
                "state": "loaded",
                "format": "mlx",
                "enabled": True,
                "learning_level": "new",
                "support": "pending",
                "reason": "not measured yet",
                "size_bytes": 3.1e9,
                "files": [
                    {
                        "id": "one",
                        "path": "/fixture/qwen",
                        "format": "mlx",
                        "size_bytes": 3.1e9,
                    }
                ],
                "learning": {"conversations": 0},
                "versions": [{"id": "base", "kind": "base", "current": True}],
            }
        ],
    )
    app = SongeApp(client=client)
    async with app.run_test(size=(110, 48)) as pilot:
        await pilot.pause()
        models = app.query_one("#models", DataTable)
        models.focus()
        await pilot.pause()
        await pilot.press("enter")
        details = str(app.query_one("#model-details", Static).render())
        # The model being trained says so; nothing else is known yet.
        assert details.splitlines()[0] == "Now          Training"
        assert "Next         update after more conversations" in details
        assert models.get_row("one")[0].plain == "Qwen3.5-4B"
        assert [column.label.plain for column in models.columns.values()] == [
            "Model",
            "Learning",
            "Status",
        ]
        assert models.get_row("one")[1] == "New"
        assert models.get_row("one")[2].plain == "● Training"
        # Check rendered colors, including the cursor and the default widget theme.
        svg = app.export_screenshot()
        fills = dict(re.findall(r"\.([\w-]+) \{ fill: (#[a-fA-F0-9]{6})", svg))
        doc = ET.fromstring(svg)
        accented = []
        for node in doc.iter("{http://www.w3.org/2000/svg}text"):
            fill = fills.get(node.get("class"))
            if not fill:
                continue
            red, green, blue = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
            if max(red, green, blue) - min(red, green, blue) > 10:
                assert green > red and green > blue
                accented.append("".join(node.itertext()).strip())
        assert accented == ["●", "●"]
        client.data["paused"] = True
        await app.refresh_data()
        assert models.get_row("one")[0].plain == "Qwen3.5-4B"
        assert models.get_row("one")[2].plain == "No conversations yet"
        assert "●" not in app.export_screenshot()


async def test_full_pipeline_panel_updates_and_model_identity(pipeline_data):
    client = Client()
    model = {
        "id": "one",
        "name": "Qwen3.5-4B",
        "format": "mlx",
        "state": "loaded",
        "support": "trainable",
        "enabled": True,
        "learning_level": "new",
        "pending": 12,
        "pipeline": pipeline_data,
        "files": [{"id": "one", "path": "/fixture/qwen", "format": "mlx"}],
        "reason": "[red]literal[/red]",
    }
    client.data.update(models=[model], service_running=True)
    app = SongeApp(client=client)
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.pause()
        models = app.query_one("#models", DataTable)
        models.focus()
        await pilot.pause()
        await pilot.press("enter")
        details = str(app.query_one("#model-details", Static).render())
        labels = [line.split()[0] for line in details.splitlines()]
        assert labels == ["Now", "Result", "Learns", "Data", "Next", "Versions"]
        assert "38 conversations · 6 lessons" in details
        assert "fixed tool errors · your corrections · successful retries" in details
        assert "examples" not in details and "runs" not in details
        assert "Repeat-error rate down 18 percentage points" in details
        assert (
            "v3 · In use · v2 · Available to restore · Original · Available to restore"
            in details
        )
        assert "[red]literal[/red]" not in details
        # Inserting a model preserves the selected model, expansion and focus.
        client.data["models"].insert(0, {"id": "other", "enabled": False})
        await app.refresh_data()
        assert app.selected_model()["id"] == "one"
        assert app.query_one("#model-details").display
        assert app.focused is models
        # A recorded step from a finished run is not live work.
        model["pipeline"]["consolidate"]["step"] = 42
        await app.refresh_data()
        assert "Step 42" not in str(app.query_one("#model-details", Static).render())
        await pilot.press("enter")
        assert not app.query_one("#model-details").display
        await app.refresh_data()
        assert not app.query_one("#model-details").display
