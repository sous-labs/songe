# Roadmap

Songe is in alpha. This page is where we are headed, not a schedule: it has no
dates and will change as we learn from people using it. Ideas and votes are
welcome as [issues](https://github.com/sous-labs/songe/issues).

## In progress

- **More setups and models.** Verifying Songe end to end across model servers
  (LM Studio, vLLM, Ollama, llama.cpp, mlx-lm), operating systems and model
  families.
- **Lighter training.** Deep Dreams that fit alongside everyday apps, on
  everyday machines.
- **A more robust Dream.** Better lessons from small models, a stronger check
  before a new version is used, and clear, honest reports of what changed.

## Next

- **Reinforcement learning.** Today Songe learns from what already happened
  in your conversations. Next, the model builds its own sandbox from what you
  work on, practises there, and learns from the results: trying, failing and
  improving on its own, not only from feedback.

## Future directions

- **Models that become specialists in how you work.** A local model that
  keeps getting better at your tools, your codebase and your preferences, the
  longer you use it.
- **Local models that work alongside frontier models.** A frontier agent such
  as Claude directs, and your local model does more of the work, learning
  from that collaboration until it can take on more by itself.

## Not planned

- Sending your conversations or learned versions anywhere. Learning stays on
  your machine.
- Running your tools or touching your files. Songe reads what already
  happened; any practice it runs happens in an isolated sandbox.
- A chat interface of its own. Songe works behind the agents you already use.
- Accounts, telemetry or cloud training.

How it works today: [README](README.md) and [DESIGN.md](DESIGN.md).
