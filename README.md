[![Songe: a bed on the moon, and the command pip install songe](https://souslabs.ai/assets/songe-share-v1.png)](https://souslabs.ai/songe)

# Songe

A local model that learns from how you use it.

Use your model as usual through opencode, pi or any OpenAI-compatible tool. When your agent is quiet, Songe looks at what happened after each of the model's actions — the tool result, the error, your next message — and teaches the model to see it coming. A new version is used only if it passes checks. Nothing leaves your machine.

## Install

With [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/) (Python 3.11 or later):

```sh
uv tool install songe      # or: pipx install songe
songe init
```

macOS with Homebrew (newer Homebrew asks you to trust a tap once):

```sh
brew tap sous-labs/tap
brew trust sous-labs/tap
brew install songe
songe init
```

Plain `pip install songe` works inside a virtual environment; outside one,
Homebrew's and most Linux distributions' Python refuse it.

`songe init` finds your model server and your agents, shows what it will change, and asks once. Then pick a `songe/…` model in your agent. In opencode, a local default model switches over automatically.

Needs a model served locally by LM Studio, Ollama, llama.cpp, mlx-lm or vLLM. Learning runs on Apple Silicon (MLX); Linux and Windows with NVIDIA or AMD are supported but less tested. Machines without a GPU still serve and record.

## Use

Use your agent as usual. Songe learns in **Dreams**:

- **Quick Dream** (a few minutes): reads your newest conversations and adds
  the preferences you stated or showed ("always use pnpm", "put my name at
  the top of the script") to the model's instructions right away. It shows
  what it added, with a before/after where the answer changes.
- **Deep Dream** (an hour or more): writes practice from those lessons,
  trains a small adapter on it, and checks the result on your conversations
  before using it. If it is not clearly better, nothing changes, and the
  report says why.

With **auto-dream** on, Dreams start when your agent has been quiet for a
while and stop the moment you use it again. Or press **Dream** in the
dashboard. A Dream never just disappears: it keeps going on the page or
says why it stopped.

```sh
songe          # what it is doing
songe open     # dashboard in your browser
songe models   # each model, and whether it dreams on its own
```

To change things:

```sh
songe models MODEL on|off   # auto-dream for one model
songe undo MODEL            # go back one version
songe pause
songe resume
songe sleep                 # start a Deep Dream now
songe doctor                # check the setup
songe forget                # delete recorded conversations (learned versions stay)
```

## How it learns

- **Reading.** Songe reads each conversation for the moments that teach
  something: your corrections and stated rules, retries, tool errors.
- **Practice.** For each lesson your model writes short practice scenes in
  your agent's own tools. Your real conversations are never trained on.
- **Training.** The model as it was, shown the lesson, is the teacher; the
  same model with a small adapter, shown only the scene, is the student
  (on-policy distillation). Earlier skills are rehearsed so they stay.
- **Checking.** At the moments you reacted to, the version in use and the
  new one each answer; a judge compares what they do. The new version is
  used only if it does clearly better, does not change what already worked,
  and its answers stay healthy.

Songe never runs your tools. It only reads what already happened.

## When it runs

Only when your agent is quiet, there is enough free memory, the battery is
not low and the machine is cool. Before the first training Songe measures
how much your computer can learn at once. When you come back it stops at
once, saves its place and continues later.

## What it changes

- Its data folder: `~/Library/Application Support/songe` on macOS, `$XDG_DATA_HOME/songe` on Linux, `%LOCALAPPDATA%\songe` on Windows.
- A `songe/` folder in LM Studio's models folder with the current learned version. Older versions are rebuilt when you return to them. Your own models are never touched.
- One provider entry in opencode and pi, with a backup.
- A background service that starts at login.

No account, no notifications.

## Upgrade and remove

`brew upgrade songe` or `pip install -U songe`: the service moves to the new version by itself in a quiet moment.

`songe uninstall` stops the service and restores your agent settings; add `--purge` to also delete its data and learned versions. Run it before `brew uninstall songe`.

---

Details and development: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Design: [DESIGN.md](DESIGN.md).

## License

[Apache-2.0](LICENSE).
