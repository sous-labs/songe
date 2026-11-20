# Contributing

Thanks for helping. Songe is in alpha; issues and pull requests are welcome.

## Set up

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/sous-labs/songe
cd songe
uv sync
```

Run your checkout as the real `songe` command with `uv tool install --force .`
(then `songe init`). Learning needs Apple Silicon (MLX) or an NVIDIA/AMD GPU
with the `torch` extra; everything else runs anywhere.

## Before you open a pull request

```sh
uv run ruff check src tests scripts .github/ci packaging/homebrew
uv run pytest
```

CI runs the same checks on Ubuntu and Windows (Python 3.11 and 3.12) for every
pull request; `main` only changes through a pull request with passing CI.

- Keep a change to one topic, with a test that fails without it.
- Say what you saw and what changed in the description; for a bug, how to
  reproduce it (`songe --version`, OS, model server, agent).
- User-facing wording lives in `src/songe/wording.py` and
  `src/songe/ui/wording.json`; keep it plain.
- Songe never runs tools from recorded conversations and never sends data off
  the machine; changes must keep it that way.

More detail on how it works: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md),
[docs/CONTRACTS.md](docs/CONTRACTS.md) and [DESIGN.md](DESIGN.md).

## Security

Please report vulnerabilities privately, as described in
[SECURITY.md](SECURITY.md), not in a public issue.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
