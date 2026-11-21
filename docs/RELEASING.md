# Packaging and releases

The package name is `songe`, the source repository is `sous-labs/songe`, and
the public tap is `sous-labs/homebrew-tap` (installed as
`sous-labs/tap/songe`). PyPI includes the CLI release `0.1.0a2`. The original
`0.0.0.dev0` package was a name reservation without the CLI.

## Dependency policy

Python 3.11 or later is required. The default install contains HTTP transport,
the proxy, SQLite store, CLI, bundled web assets, and the Textual dependency for
the TUI workstream. Apple Silicon installs also select MLX and MLX-LM through
`sys_platform == 'darwin' and platform_machine == 'arm64'` markers. `[mlx]`
explicitly requests the same marked dependencies; it does not enable MLX on
other platforms. The MLX 0.32 line matches the existing compatibility code.

`[torch]` adds PyTorch, Transformers, PEFT, Accelerate, and Safetensors.
`[cuda]` adds that stack plus the optional bitsandbytes dependency on
Linux/Windows. `[rocm]` adds the Torch stack without CUDA-specific quantization
packages. No ML framework is selected automatically on Linux or Windows;
Python environment markers cannot identify GPU hardware or installed drivers.
The Torch backend, portable CLI leases, and TUI are implemented. Installing dependencies is not a hardware training validation.

`uv.lock` records development resolution, including optional stacks; normal
`uv sync --locked` installs only base and development dependencies. Optional
frameworks are not downloaded merely because they appear in the lockfile.

## Accelerator installation

Use a virtual environment for pip commands. On Apple Silicon,
`python -m pip install songe` selects MLX automatically.

For NVIDIA on Linux or Windows, use the command from the official
[PyTorch installer selector](https://pytorch.org/get-started/locally/) to
install `torch` with the CUDA wheel index matching the machine's driver and
Python version, then run:

```sh
python -m pip install 'songe[cuda]'
```

For AMD on Linux, select Linux / Pip / the supported ROCm version in the same
selector, install `torch` using its `https://download.pytorch.org/whl/rocm…`
index, then run:

```sh
python -m pip install 'songe[rocm]'
```

For Intel XPU, follow the official
[PyTorch XPU instructions](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
and install `songe[torch]` afterwards. For CPU-only Torch, first install `torch`
with `--index-url https://download.pytorch.org/whl/cpu`, then `songe[torch]`.
Installing Torch first lets pip reuse the selected build when it satisfies
Songe's declared range. Do not use `--upgrade` on the second command unless
intending to replace that build. Wheel availability and driver support are
defined by PyTorch; this project does not pin a CUDA/ROCm index in PyPI metadata.
ROCm/XPU quantization needs the vendor-supported implementation; the standard
bitsandbytes wheel is not evidence that it works there.

Check the selected accelerator with:

```sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.version, 'hip', None))"
```

This checks the installed framework, not a Songe training run. `songe init` discovers the accelerator and installs a missing training stack in a private environment after showing the exact commands. It asks once for dependencies and agent/service changes; `--yes` accepts the plan. Apple Silicon uses its existing MLX dependencies for native MLX models. GGUF import/export additionally needs the CPU converter stack; init displays and provisions that private converter environment when missing. CPU-only installs remain serve-only.

## CI coverage

`.github/workflows/ci.yml` runs on every pull request, and before each
release. On Ubuntu and Windows with Python 3.11 and 3.12, and on an Apple
Silicon `macos-15` runner: Ruff, pytest, `uv build`, then the install smoke
test (`.github/ci/smoke.py`): the built wheel in a clean venv runs `songe init
--yes` against a fake model server in a throwaway home, serves a chat request,
and `songe uninstall --purge` must restore the agent config. A separate
`dashboard` job runs the same with the dashboard loaded in headless Chromium,
failing on any script error. These checks take seconds; pull requests stay a
few minutes. uv caches are managed by
[setup-uv](https://docs.astral.sh/uv/guides/integration/github/).

Only before a release, `release.yml`'s `dream-check` job runs one real Deep
Dream (`.github/ci/release_dream.py`): on a Modal GPU it installs the release
wheel with `[cuda]`, records made-up conversations with corrections for a small
model (Qwen2.5-0.5B-Instruct) and runs the Dream worker to a decision.
Publishing waits for it. It needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as
secrets of the `release-check` environment, which only `v*` tags may use. Run
it by hand with `SONGE_WHEEL=dist/<wheel> uvx modal run
.github/ci/release_dream.py`.

The test-only plugin in `.github/ci/platform_checks.py` reports Metal tests as
skipped when Metal cannot be imported or is unavailable, including sandboxed
Mac sessions. Non-MLX installations skip the model-support test that reads
MLX-LM's installed metadata. Windows uses the same portable worker and mutation leases as Unix. The remaining store permission assertion skips on Windows because POSIX mode `0700` is not a Windows ACL check. Other portable
tests, including HTTP streaming, run on Windows. No pull-request job installs a
CUDA/ROCm training stack or loads real models; only the release Dream check does.

## One-time PyPI Trusted Publishing setup

In GitHub repository settings for `sous-labs/songe`, create an environment named
`pypi` and restrict deployment to release tags `v*`. Configure any desired
reviewers there. No PyPI API token is needed.

The GitHub pending-publisher settings are:

| Field | Value |
| --- | --- |
| PyPI project name | `songe` |
| Owner | `sous-labs` |
| Repository | `songe` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

Because `songe` is already registered, add this as a Trusted Publisher under
the existing project's **Manage → Publishing**, rather than creating another
pending project. If registering on an index where the project does not yet
exist, use these same values in its pending-publisher form. See
[PyPI's existing-project instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

`release.yml` runs the reusable CI workflow before building. A separate build
job verifies that `v<version>`, `pyproject.toml`, and
`songe.__version__` agree, runs `uv build` and `twine check --strict`, and
uploads the sdist and wheel as one artifact. Only the subsequent publish job
gets `id-token: write` and the `pypi` environment. It downloads those artifacts
and uses `pypa/gh-action-pypi-publish` with Trusted Publishing and attestations.
See [the publisher action's instructions](https://github.com/pypa/gh-action-pypi-publish).

## Release procedure

1. Integrate all workstreams and resolve the documented Windows/backend gaps
   appropriate to the release's advertised scope. Run CI on `main`.
2. Set the same stable or preview version in `pyproject.toml` and
   `src/songe/__init__.py`, and run `uv lock`. Tag names use `v`, for example
   `v0.1.0` or `v0.1.0a2`; the Homebrew updater accepts stable versions and
   `a`, `b`, and `rc` previews, ordered using PEP 440.
3. Validate from a clean checkout:

   ```sh
   export UV_CACHE_DIR=/Volumes/disk/sous-build-cache/uv  # this development Mac
   uv sync --locked
   uv run ruff check src tests scripts .github/ci packaging/homebrew
   uv run pytest
   uv build
   uvx twine check --strict dist/*
   ```

4. Commit the version changes, create the matching `v<version>` tag, and push
   the reviewed commit and tag when ready to publish. The tag starts release
   CI, then the build and PyPI publishing jobs. Do not reuse an uploaded version.
5. Check the [PyPI release files](https://pypi.org/project/songe/#files), their
   attestations, and an installation in a fresh environment. A successful CI
   build alone does not verify a live PyPI install or accelerator training.
6. Update the tap as below. The optional notification job runs only after PyPI
   publication. If it fails, rerun that job or dispatch the tap updater manually;
   do not rerun a successful PyPI upload to repair a tap notification.

## Homebrew tap

Keep these source files synchronized with the public tap:

| Source in this repository | Destination in the tap |
| --- | --- |
| `packaging/homebrew/songe.rb` | `Formula/songe.rb` |
| `packaging/homebrew/update_formula.py` | `scripts/update_formula.py` |
| `packaging/homebrew/bump-formula.yml` | `.github/workflows/bump-formula.yml` |

The checked-in formula pins the published CLI wheel and every dependency
wheel with a SHA256. The updater requires Python 3.11+ and uv; its inline
script metadata supplies `packaging` automatically. To generate a release in
the tap checkout:

```sh
uv run --script scripts/update_formula.py 0.1.0a2 --formula Formula/songe.rb
```

The updater fetches that exact version's PyPI JSON, rejects yanked or
unexpected artifacts, verifies the universal wheel's SHA256, package metadata,
and `songe` console entry point, and rejects version downgrades. Previews are
allowed. It then resolves the published wheel's dependencies for CPython 3.12
on macOS ARM64/Intel and Linux ARM64/x86_64. It does not use this repository's
pyproject or development lockfile. All dependencies must have compatible
wheels; a missing platform wheel fails the update before writing the formula.
Resolution excludes uploads newer than the release wheel, making repeat
updates reproducible when index metadata is unchanged.

macOS wheels target macOS 14 or newer (including the MLX stack on Apple
Silicon); Linux wheels target glibc 2.28 or newer. These minimums are explicit
in the resolver and wheel selector. Universal resources are shared; native
wheels and platform-specific dependencies appear in conditional resource
blocks. Linux and Intel wheel resolution is checked during every update;
an actual installation must still be tested on those hosts before claiming
runtime validation there.

Homebrew downloads and verifies every resource before the build sandbox.
Inside the sandbox the formula creates an isolated virtualenv without pip,
loads the pinned pure Python `installer` wheel through `PYTHONPATH`, and
installs the already downloaded wheels. `installer` performs wheel extraction,
script generation, and RECORD installation without resolving dependencies,
accessing the network, or asking for the macOS version. This avoids the pip
truststore import and uv interpreter-inspection failures seen when the
sandbox makes `platform.mac_ver()` empty. Neither pip nor uv is used during
`brew install`, and users need no trust override or environment variables.
See [installer's CLI documentation](https://installer.pypa.io/en/stable/cli/installer/).

After reviewing and pushing a generated formula, verify it with:

```sh
brew install --build-from-source sous-labs/tap/songe
songe --version
brew test sous-labs/tap/songe
brew uninstall songe
```

For local verification before pushing, copy the generated formula into an
existing local tap checkout used by Homebrew, or use a temporary local tap.
Remove any temporary tap afterwards. `brew test` runs only `songe --version`
with a temporary data directory. Neither installation nor testing runs
`songe init`, configures agents, or starts/stops a service.

For automated bumps, allow the tap workflow's `GITHUB_TOKEN` to write contents
(and ensure branch rules permit its formula commits). In `sous-labs/songe`,
configure a fine-grained token or GitHub App token as `HOMEBREW_TAP_TOKEN`,
scoped to **Contents: write** on `sous-labs/homebrew-tap` for repository
dispatch, and set the repository variable `HOMEBREW_TAP_ENABLED=true`.
The default source-repository `GITHUB_TOKEN` cannot dispatch to another private
repository. After each successful publication, `release.yml` sends a
`songe-release` dispatch; the tap workflow uses uv and the updater to verify
the published CLI wheel and commit the complete formula with dependency hashes.
The tap itself needs no PyPI secret or access to the private source repository.

To recover a failed notification:

```sh
gh workflow run bump-formula.yml --repo sous-labs/homebrew-tap -f version=0.1.0a2
```
