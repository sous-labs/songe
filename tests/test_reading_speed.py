"""Long agent sessions read in reasonable time: shorter review prompts, and
answers cut off at the length limit asked once more with more room."""

import json

import httpx

from songe.reading import client as reading_client
from songe.reading.extract import REVIEW_FULL, Session


class Recorder:
    def __init__(self):
        self.prompts = []

    async def ask(self, prompt, schema, name, **kwargs):
        self.prompts.append((name, prompt))
        return {"signals": []} if name == "review" else {}


def build_session(results):
    items = [{"kind": "user", "content": "Build the demo."}]
    for n in range(1, results + 1):
        call = {"id": f"c{n}", "name": "bash", "arguments": json.dumps({"command": f"cargo build --step {n}"})}
        items.append({"kind": "action", "step": n, "text": "", "tool_calls": [call], "exchange": n})
        items.append({"kind": "tool_result", "step": n, "call": f"c{n}", "tool": "bash",
                      "content": f"first line of build {n}\n" + "\n".join(f"detail {n}.{k}" for k in range(40))})
    return items


async def test_the_review_shows_the_turns_last_results_in_full():
    model = Recorder()
    session = Session("s", build_session(12), model, "/work")
    await session.run_all([])
    (prompt,) = [p for name, p in model.prompts if name == "review"]
    early, late = 12 - REVIEW_FULL, list(range(12 - REVIEW_FULL + 1, 13))
    for n in range(1, early + 1):  # one line: command and the result's first line
        assert f"first line of build {n}" in prompt and f"detail {n}.1" not in prompt
    for n in late:  # in full (result_view)
        assert f"detail {n}.1" in prompt


async def test_a_cut_off_answer_is_forced_from_the_reasoning_so_far(store):
    sent = []

    def server(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:  # reasoned too long: no JSON yet
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "length", "message": {
                    "content": "", "reasoning_content": "Step 4 broke the build, so the answer is step 4."}}],
                "usage": {"completion_tokens": 6000}})
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"steps": [4]})}}],
            "usage": {"completion_tokens": 12}})

    reader = reading_client.Reader(store, "http://model/v1", "m", server="vllm",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    answer = await reader.ask("which step?", {"type": "object"}, "responsible")
    assert answer == {"steps": [4]} and reader.forced == 1 and reader.tokens_out == 6012
    # On vLLM the reasoning is capped by the server; the answer still follows.
    assert sent[0]["thinking_token_budget"] == reading_client.THINKING_BUDGET[False]
    forced = sent[1]
    # Only the reasoning was cut: the follow-up shows it, asks for the JSON
    # alone, with reasoning off and a small room (nothing thought twice).
    assert "Step 4 broke the build" in forced["messages"][0]["content"]
    assert forced["messages"][0]["content"].endswith("Answer with the JSON only.")
    assert forced["reasoning_effort"] == "none"
    assert forced["chat_template_kwargs"] == {"enable_thinking": False}
    assert forced["max_tokens"] <= reading_client.FORCE_TOKENS
    # The next question starts with the normal room.
    await reader.ask("another question", {"type": "object"}, "review")
    assert sent[-1]["max_tokens"] == 6000
    await reader.close()


async def test_a_moment_still_unreadable_when_forced_is_given_up_and_remembered(store):
    sent = []

    def server(request):
        body = json.loads(request.content)
        sent.append((body["max_tokens"], body["messages"][0]["content"]))
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                      "message": {"content": '{"steps": [1, '}}],
                                         "usage": {"completion_tokens": body["max_tokens"]}})

    reader = reading_client.Reader(store, "http://model/v1", "m",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    answer = await reader.ask("which step?", {"type": "object"}, "responsible")
    # Not a vLLM server: no thinking budget is sent.
    assert answer.get("gave_up") and [m for m, _ in sent] == [6000, reading_client.FORCE_TOKENS]
    # Short reasoning is asked for, and generated tokens are counted.
    assert sent[0][1].endswith(reading_client.BRIEF) and reader.tokens_out == 6000 + reading_client.FORCE_TOKENS
    # The next read does not pay for it again.
    assert await reader.ask("which step?", {"type": "object"}, "responsible") == answer
    assert len(sent) == 2 and reader.cached == 1
    await reader.close()


async def test_a_keep_going_reply_is_not_classified():
    model = Recorder()
    items = build_session(2)
    items += [{"kind": "user", "content": "continue"},
              {"kind": "action", "step": 3, "text": "Done.", "tool_calls": [], "exchange": 3},
              {"kind": "user", "content": "No, you broke the build."}]
    await Session("s", items, model, "/w").run_all([])
    classified = [p for name, p in model.prompts if name == "classify"]
    assert len(classified) == 1 and "you broke the build" in classified[0]


async def test_reader_stats_are_logged_per_stage(store, caplog):
    import logging

    def server(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": json.dumps({"ok": True})}}],
                                         "usage": {"completion_tokens": 100}})

    reader = reading_client.Reader(store, "http://model/v1", "m",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    with caplog.at_level(logging.INFO, logger="songe.reading.client"):
        await reader.ask("first", {"type": "object"}, "review")
        reader.log_stats("read")
        reader.log_stats("practice")  # nothing new since: no line
        await reader.ask("second", {"type": "object"}, "review")
        await reader.close()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reader_stats")]
    assert len(lines) == 2
    assert "stage=read calls=1" in lines[0] and "per_call=100" in lines[0]
    assert "stage=close calls=2" in lines[1]
