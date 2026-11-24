import json
import random
import re
import shutil
from pathlib import Path
from statistics import mean

from ..steps import digest
from .prompts import paired_prompts

XML_FUNCTION = re.compile(r"^\s*<function=([^>\s]+)>(.*)</function>\s*$", re.S)
XML_PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def xml_call(segment, tools):
    """A tool call in the XML form some models write inside <tool_call>
    (LFM3, Qwen3-Coder): <function=name><parameter=key>value</parameter>...
    Values are text; a parameter the schema types as non-string is read as
    JSON. None when the segment is not in this form."""
    found = XML_FUNCTION.match(segment)
    if not found:
        return None
    name, body = found.groups()
    schema = next(
        (
            (t.get("function", {}).get("parameters") or {}).get("properties") or {}
            for t in tools or []
            if t.get("function", {}).get("name") == name
        ),
        {},
    )
    arguments = {}
    for key, value in XML_PARAMETER.findall(body):
        kind = (schema.get(key) or {}).get("type")
        if kind not in (None, "string"):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        arguments[key] = value
    leftover = XML_PARAMETER.sub("", body).strip()
    if leftover:
        return None  # text outside the parameters: not a clean call
    return {"name": name, "arguments": arguments}


def health(text, tokenizer, tools, tool_step, repetition_ngram=4):
    calls = []
    valid = not tool_step
    supported, malformed = True, False
    if tool_step:
        parser = getattr(tokenizer, "tool_parser", None)
        if parser:
            # Use the installed parser and its own family-specific delimiters.
            import importlib

            module = importlib.import_module(parser.__module__)
            start = getattr(module, "tool_call_start", "<tool_call>")
            end = getattr(module, "tool_call_end", "</tool_call>")
            segments = re.findall(
                re.escape(start) + r"(.*?)" + re.escape(end), text, re.S
            )
            for segment in segments:
                try:
                    result = parser(segment, tools)
                    calls.extend(result if isinstance(result, list) else [result])
                except (ValueError, TypeError, KeyError):
                    malformed = True
            malformed |= text.count(start) != len(segments) or text.count(end) != len(
                segments
            )
        if not parser:
            # Recognize the complete protocol, never JSON embedded in prose.
            # Unknown model-specific framing cannot certify tool health.
            start, end = "<tool_call>", "</tool_call>"
            segments = re.findall(start + r"(.*?)" + end, text, re.S)
            template = getattr(tokenizer, "chat_template", "") or ""
            if isinstance(template, dict):
                template = "\n".join(template.values())
            # Support belongs to the tokenizer protocol, not to whether this
            # sample complied. A missing/truncated call in a known protocol is
            # invalid and still participates in the before/after validity gate.
            supported = (
                (start in template and end in template)
                or bool(segments)
                or start in text
                or end in text
            )
            malformed = text.count(start) != len(segments) or text.count(end) != len(
                segments
            )
            if not supported:
                try:
                    value = json.loads(text)
                    segments = [text] if isinstance(value, (dict, list)) else []
                    supported = bool(segments)
                except ValueError:
                    pass
            for segment in segments:
                try:
                    value = json.loads(segment)
                    for call in value if isinstance(value, list) else [value]:
                        fn = call.get("function", call)
                        if "name" in fn and "arguments" in fn:
                            calls.append(fn)
                        else:
                            malformed = True
                except (ValueError, TypeError, AttributeError):
                    xml = xml_call(segment, tools)
                    if xml is None:
                        malformed = True
                    else:
                        calls.append(xml)
        from ..steps import valid_action

        valid = (
            bool(calls)
            and not malformed
            and valid_action(
                {
                    "tool_calls": [
                        c if "function" in c else {"function": c} for c in calls
                    ]
                },
                tools,
            )
        )
    words = text.split()
    phrases = [
        tuple(words[i : i + repetition_ngram])
        for i in range(len(words) - repetition_ngram + 1)
    ]
    repeated = 1 - len(set(phrases)) / len(phrases) if phrases else 0.0
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return {
        "calls": calls,  # parsed tool calls, for before/after examples
        "tool_valid": int(valid),
        "tool_supported": int(supported),
        "empty": int(not visible),
        "repetition": repeated,
    }


def references(base, steps, *, cache=None):
    """The original model's sampled answer for each gate step.

    cache: a ReferenceCache. The answer depends only on the original weights,
    the prompt and the answer length, so later Dreams reuse it instead of
    sampling again (sampling is most of a reference's cost)."""
    if any(s.get("split") not in {"heldout", "anchor", "drift"} for s in steps):
        raise ValueError(
            "Gate references require held-out real data, fixed anchors or the drift set"
        )
    refs = []
    for step in steps:
        getattr(base, "boundary", base.check)()
        prompt, teacher = paired_prompts(
            base.tokenizer, step, base.config, check_length=False
        )
        y = cache.get(prompt, base.config.response_tokens) if cache else None
        if y is None:
            with base.base():
                y, _, _ = base.sample(prompt, seed=7613)
            if cache:
                cache.put(prompt, base.config.response_tokens, y)
        refs.append(
            {
                "id": step["id"],
                "conversation": step["conversation"],
                "prompt": prompt,
                "teacher": teacher,
                "y": y,
                "tools": step.get("tools", []),
                "tool_step": bool(step.get("action", {}).get("tool_calls")),
            }
        )
    return refs


class ReferenceCache:
    """Reference answers kept across Dreams for one original model.

    One folder per base fingerprint; folders for other fingerprints (an
    older base) are removed when this one is first written, so the cache
    never outlives the weights it describes."""

    def __init__(self, root, fingerprint):
        self.root, self.fingerprint = Path(root), fingerprint
        self.folder = self.root / fingerprint[:24]
        self.pruned = False

    def path(self, prompt, length):
        return self.folder / (digest([self.fingerprint, length, list(prompt)]) + ".json")

    def get(self, prompt, length):
        try:
            return json.loads(self.path(prompt, length).read_text(encoding="utf-8"))["y"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, prompt, length, y):
        if not self.pruned:
            self.pruned = True
            if self.root.is_dir():
                for other in self.root.iterdir():
                    if other.is_dir() and other != self.folder:
                        shutil.rmtree(other, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(prompt, length)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"y": [int(t) for t in y]}), encoding="utf-8")
        temporary.replace(path)


class KeptMeasurements:
    """The incumbent's gate measurements kept across Dreams.

    While the version in use, its teacher and the base stay the same, its
    score, drift and answer health at a moment do not change, so the next
    Dream reuses them instead of measuring the incumbent again (a third of
    the gate). fingerprint names everything they depend on; one folder per
    fingerprint, others removed when this one is first written."""

    def __init__(self, root, fingerprint):
        self.root = Path(root)
        self.folder = self.root / digest(fingerprint)[:24]
        self.pruned = False

    def path(self, kind, ref):
        """ref: a gate reference (what it reads) or any JSON identity."""
        if isinstance(ref, dict):
            ref = [list(ref["prompt"]), list(ref.get("teacher") or []), list(ref["y"])]
        return self.folder / f"{kind}-{digest([kind, ref])}.json"

    def get(self, kind, ref):
        try:
            return json.loads(self.path(kind, ref).read_text(encoding="utf-8"))["value"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, kind, ref, value):
        if not self.pruned:
            self.pruned = True
            if self.root.is_dir():
                for other in self.root.iterdir():
                    if other.is_dir() and other != self.folder:
                        shutil.rmtree(other, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.path(kind, ref)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"value": value}), encoding="utf-8")
        temporary.replace(path)


class KeptViews(dict):
    """kl_from's views (the reference's read of each moment, shared by the
    versions of one gate) with the frozen base's reads kept on disk across
    Dreams: the drift moments and bundled prompts are read by the same base
    weights every night. The cycle teacher's reads change with the teacher
    and stay in memory only. Torch tensors (torch.save)."""

    def __init__(self, root, fingerprint):
        super().__init__()
        self.disk = KeptMeasurements(root, fingerprint)

    def _file(self, key):
        return self.disk.path("view", list(key[1:])).with_suffix(".pt")

    def get(self, key, default=None):
        if key in self:
            return self[key]
        if key[0]:  # the cycle teacher's read
            return default
        import torch

        try:
            value = torch.load(self._file(key), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, EOFError):
            return default
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key[0]:
            return
        import torch

        if not self.disk.pruned:
            self.disk.put("marker", ["views"], 1)  # creates the folder, prunes others
        path = self._file(key)
        temporary = path.with_suffix(".tmp")
        torch.save(value, temporary)
        temporary.replace(path)


def kept(cache, kind, ref, measure):
    """cache's value for this moment, else measure() (kept for next time)."""
    if cache is None:
        return measure()
    value = cache.get(kind, ref)
    if value is None:
        value = measure()
        cache.put(kind, ref, value)
    return value


def kl_between(base, candidate, prompt, teacher_prompt, y, *, hindsight=False, views=None, state=None):
    # Held-out hindsight KL uses the same teacher as training (the version
    # the cycle started from); retention drift stays against the original base.
    # views: the reference's reads kept across versions; state: the
    # candidate's read of the prompt, shared with its answer (see kl_from).
    extra = {} if views is None else {"views": views}
    if state is not None:
        extra["state"] = state
    return candidate.kl_from(base, prompt, teacher_prompt, y, hindsight=hindsight, **extra)


def interval(values_by_cluster, confidence, bootstraps=2000):
    """Two-sided percentile bootstrap interval of the mean, resampling
    clusters (conversations), as the gate's noise margin does."""
    means = [mean(v) for v in values_by_cluster.values()]
    rng = random.Random(7613)
    draws = sorted(mean(rng.choices(means, k=len(means))) for _ in range(bootstraps))
    tail = int((1 - confidence) / 2 * bootstraps)
    return draws[tail], draws[bootstraps - 1 - tail]


def drift_between(base, candidate, r, shapes, views=None):
    """The candidate's KL to the base on a prompt the base answered, and in
    shapes (beside it, one entry per prompt) what that mean hides: the base's
    and the candidate's entropy, and how often they agree on the most likely
    token. None for a backend that only measures the KL."""
    measure = getattr(candidate, "measure", None)
    if measure is None:
        shapes.append(None)
        return kl_between(base, candidate, r["prompt"], r["prompt"], r["y"], views=views)
    m = measure(base, r["prompt"], r["prompt"], r["y"], **({} if views is None else {"views": views}))
    shapes.append([m["reference_entropy"], m["entropy"], m["agreement"]])
    return m["kl"]


def sharpness(checkpoint):
    """How the candidate's distributions compare with the base's on the
    "not broken" set that gated (the drift set, else the general prompts):
    mean entropy of each in nats/token, the relative change, and the share of
    tokens where both prefer the same one. A model can grow overconfident
    (entropy falls) with its preferred tokens unchanged, which keeps the mean
    KL small; this shows it. Recorded only: it gates nothing."""
    source = (checkpoint.get("drift_check") or {}).get("source", "general")
    shapes = [s for s in checkpoint.get(
        "drift_set_shapes" if source == "drift" else "drifts_shapes") or [] if s]
    if not shapes:
        return {"base_entropy": None, "entropy": None, "entropy_change": None, "agreement": None}
    base, own = mean(s[0] for s in shapes), mean(s[1] for s in shapes)
    return {
        "base_entropy": base,
        "entropy": own,
        "entropy_change": (own - base) / base if base > 0 else None,
        "agreement": mean(s[2] for s in shapes),
    }


def drift_choice(general, drift_values, drift_refs, config):
    """Which "not broken" measure gates (docs/CONTRACTS.md, Integration
    decisions): the drift set once it has min_drift conversations and its
    interval is narrower than the limit; the fixed anchors until then."""
    info = {"general": general, "source": "general", "drift_set": None}
    if not drift_refs:
        info["reason"] = "no confirmed successes yet"
        return general, info
    clusters = {}
    for r, value in zip(drift_refs, drift_values, strict=True):
        clusters.setdefault(r["conversation"], []).append(value)
    info["drift_set"] = mean(drift_values)
    info["conversations"] = len(clusters)
    if len(clusters) < config.min_drift:
        info["reason"] = f"{len(clusters)} of {config.min_drift} conversations"
        return general, info
    low, high = interval(clusters, config.gate_confidence)
    info["interval"] = [low, high]
    if high - low >= config.general_drift_limit:
        info["reason"] = "interval wider than the drift limit"
        return general, info
    info["source"], info["reason"] = "drift", "enough confirmed successes"
    return info["drift_set"], info


def measure_row(base, candidate, r, views):
    """One held-out moment for one version: the score against the cycle's
    teacher and the health of the version's own answer."""
    # The prompt is read once for the score and the answer.
    state = candidate.prompt_state(r["prompt"]) if hasattr(candidate, "prompt_state") else None
    kl = kl_between(
        base, candidate, r["prompt"], r["teacher"], r["y"], hindsight=True, views=views,
        state=state,
    )
    _, text, _ = candidate.sample(
        r["prompt"], seed=7613, **({} if state is None else {"state": state})
    )
    state = None
    return {
        "kl": kl,
        "tool_step": r["tool_step"],
        **{
            k: v
            for k, v in health(
                text,
                candidate.tokenizer,
                r["tools"],
                r["tool_step"],
                candidate.config.repetition_ngram,
            ).items()
            if k != "calls"  # gate rows keep measurements only
        },
    }


def evaluate(
    base,
    candidate,
    refs,
    general_refs,
    *,
    drift_refs=(),
    checkpoint=None,
    commit=lambda: None,
    identical=False,
    views=None,
    kept_as=None,
    rows=True,
):
    """Gate rows for one version against the frozen base.

    identical: this version is the base itself (the first Dream's
    incumbent), so its drift from the base is zero by definition and is not
    measured; held-out rows still are.

    views: a dict shared by the versions of one gate, so the reference's
    read of each moment (the same for every version) is done once.
    kept_as: a KeptMeasurements for a version measured again on later
    Dreams (the incumbent): moments it has measured are not measured again.
    rows=False: only the "not broken" measure; returns its value (a later
    call with the same checkpoint goes on to the held-out rows)."""
    checkpoint = {} if checkpoint is None else checkpoint

    def kept_drift(kind, r, into):
        """The drift KL at r, its shape appended to into; both kept for the
        incumbent (kept_as) like its other measurements."""
        def measure():
            shape = []
            return [drift_between(base, candidate, r, shape, views), shape[0]]

        value, shape = kept(kept_as, kind, r, measure)
        into.append(shape)
        return value

    # The drift set first: the same measure on real contexts where the base
    # was right (confirmed successes). Never trained on. When it can decide
    # (drift_choice), the bundled prompts are not needed and not measured.
    drift_set = checkpoint.setdefault("drift_set", [])
    # One entry beside each KL (None where it was not measured, as in a
    # check begun before these were kept).
    set_shapes = checkpoint.setdefault("drift_set_shapes", [])
    set_shapes.extend([None] * (len(drift_set) - len(set_shapes)))
    for r in list(drift_refs)[len(drift_set) :]:
        if identical:
            drift_set.append(0.0)
            set_shapes.append(None)
            continue
        getattr(candidate, "boundary", candidate.check)()
        drift_set.append(kept_drift("drift", r, set_shapes))
        commit()
    drifts = checkpoint.setdefault("drifts", [])
    shapes = checkpoint.setdefault("drifts_shapes", [])
    shapes.extend([None] * (len(drifts) - len(shapes)))
    _, decided = drift_choice(None, drift_set, list(drift_refs), candidate.config)
    if decided["source"] != "drift":
        # The bundled prompts, with the same samples for every version: a
        # candidate cannot pass by choosing easier generated sequences.
        if callable(general_refs):
            general_refs = general_refs()  # computed only when needed
        for r in general_refs[len(drifts) :]:
            if identical:
                drifts.append(0.0)
                shapes.append(None)
                continue
            getattr(candidate, "boundary", candidate.check)()
            drifts.append(kept_drift("general", r, shapes))
            commit()
    commit()
    drift, checkpoint["drift_check"] = drift_choice(
        mean(drifts) if drifts else None, drift_set, list(drift_refs), candidate.config
    )
    if not rows:
        return drift
    rows = checkpoint.setdefault("rows", [])
    for r in refs[len(rows) :]:
        getattr(candidate, "boundary", candidate.check)()
        measured = kept(kept_as, "row", r, lambda r=r: measure_row(base, candidate, r, views))
        rows.append({"id": r["id"], "conversation": r["conversation"], "drift": drift, **measured})
        commit()
    return rows
