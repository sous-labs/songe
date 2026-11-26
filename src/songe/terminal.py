"""Restrained terminal output, also readable in pipes and captured logs."""

import json
import os
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from . import NAME
from .interface_copy import terse
from .panels import pipeline
from .status import model_training, training_model
from .wording import (
    IGNORED_MODEL_SETTINGS,
    TRAINING_SETTINGS,
    model_level,
    model_status,
    status_lines,
)

ACCENT = "cyan"


def console(*, error=False):
    stream = sys.stderr if error else sys.stdout
    styled = stream.isatty() and "NO_COLOR" not in os.environ
    return Console(
        file=stream,
        force_terminal=styled,
        no_color=not styled,
        color_system="auto" if styled else None,
        highlight=False,
        markup=False,
        # Piped output must retain full ids and paths, independent of $COLUMNS.
        width=None if styled else 10000,
    )


def result(message):
    line = Text("✓ ", style="green")
    line.append(message, style="default")
    console().print(line)


def error(message, hint=None):
    line = " ".join(terse(message).splitlines())
    console(error=True).print(Text(f"✗ {line}", style="red"))
    console(error=True).print(Text(hint or f"check  {NAME} doctor", style="dim"))


def trimmed(output, renderable):
    """Print a table without the spaces Rich pads its last column with to
    the widest cell (they ended every `songe init` line)."""
    for line in output.render_lines(renderable, pad=False):
        text = Text.assemble(*((s.text, s.style) for s in line if not s.control))
        text.rstrip()
        output.print(text, no_wrap=True, crop=False)


def facts(rows):
    table = Table.grid(padding=(0, 2))
    for label, value in rows:
        table.add_row(Text(label, style="dim"), Text(str(value)))
    trimmed(console(), Padding(table, (0, 0, 0, 2), expand=False))


@contextmanager
def step(label, completed):
    output = console()
    if output.is_terminal and "NO_COLOR" not in os.environ:
        with output.status(Text(label), spinner="dots", spinner_style=ACCENT):
            yield
    else:
        yield
    result(completed)


def model_status_text(model, data=None):
    data = data or {}
    text = Text()
    if model_training(model, training_model(data)) and model_level(model) != "off":
        text.append("● ", style="green")
    text.append(model_status(model, pipeline(model, snapshot=data), data))
    return text


def status_text(data):
    text = Text()
    for index, (label, value) in enumerate(status_lines(data)):
        if index:
            text.append("\n")
        text.append(f"{label:<8}")
        if label == "now" and training_model(data):
            text.append("● ", style="green")
        text.append(value)
    return text


def models_table(models, *, data=None):
    data = data or {}
    table = Table(box=None, padding=(0, 2), pad_edge=False, header_style="dim")
    for label in ("Model", "Auto-dream", "Status"):
        table.add_column(label, no_wrap=True)
    for model in models:
        # What it learns from is Songe's decision; only auto-dream is a choice.
        auto = model.get("auto_dream", model_level(model) != "off")
        table.add_row(
            Text(model.get("name") or model["id"]),
            Text("On" if auto else "Off", style="default" if auto else "dim"),
            model_status_text(model, data),
        )
    if not models:
        console().print(
            Text("  Connect your first model · Run songe init", style="dim")
        )
    else:
        trimmed(console(), Padding(table, (0, 0, 0, 2), expand=False))


def status(data):
    output = console()
    if data.get("upgrade_notice") and not os.environ.get("SONGE_NOTICE_SHOWN"):
        output.print(Text(data["upgrade_notice"], style="yellow"))
    output.print(status_text(data))
    from .wording import cloud_line

    cloud = cloud_line((data.get("settings") or {}).get("cloud"))
    if cloud:
        output.print(Text(f"{'cloud':<8}{cloud}"))
    if data.get("recordings_note"):
        output.print(Text(data["recordings_note"], style="yellow"))
    output.print()
    # Details per model live in the dashboard (songe open), not here.
    models_table(data.get("models", []), data=data)


def training_settings(model, rows, ignored):
    output = console()
    title = f"{TRAINING_SETTINGS} · {model}" if model else TRAINING_SETTINGS
    output.print(Text(title))
    table = Table(box=None, padding=(0, 2), pad_edge=False, header_style="dim")
    table.add_column("Setting", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Value")
    for key, value, source in rows:
        table.add_row(
            key, source, Text(json.dumps(value, ensure_ascii=False, sort_keys=True))
        )
    trimmed(output, table)
    if ignored:
        skipped = ", ".join(sorted(ignored))
        output.print(Text(f"{IGNORED_MODEL_SETTINGS} · {skipped}", style="yellow"))
