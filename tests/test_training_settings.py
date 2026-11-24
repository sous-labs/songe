"""The settings a Dream resolves, and the layer each one came from (#124)."""

import json

from songe.config import Config, data_dir
from songe.models import registry


def learned_model(config, store):
    config.calibration_available = True
    registry.migrate(config, store)
    registry.observe(config, store, "learned")
    registry.support(store, "learned", "trainable")
    return registry.find(store, "learned")


def save_row(store, row, **values):
    saved = dict(row["config"], **values)
    with store.connect() as db:
        db.execute("UPDATE models SET config=? WHERE id=?", (json.dumps(saved), row["id"]))


def test_installation_per_model_and_default_are_named(store, config):
    row = learned_model(config, store)
    save_row(store, row, accumulation=8)
    root = data_dir(store.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text("learning_rate = 6e-5\n", encoding="utf-8")

    settings, ignored = registry.training_settings(
        Config.load(store.root), store, "learned"
    )
    sources = {key: source for key, _, source in settings}
    values = {key: value for key, value, _ in settings}

    assert (values["learning_rate"], sources["learning_rate"]) == (
        6e-5,
        "installation",
    )
    assert (values["accumulation"], sources["accumulation"]) == (8, "per model")
    assert (values["vocab_clip"], sources["vocab_clip"]) == (1.0, "default")
    assert ignored == []


def test_a_per_model_key_a_dream_drops_is_reported(store, config):
    row = learned_model(config, store)
    save_row(store, row, vocab_clip=0.0)

    _, ignored = registry.training_settings(Config.load(store.root), store, "learned")

    assert ignored == ["vocab_clip"]
