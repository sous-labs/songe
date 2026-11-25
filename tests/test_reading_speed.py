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


async def test_a_server_that_refuses_the_thinking_cap_is_asked_without_it(store):
    """Live (vLLM 0.34.0 serving Mistral-7B-Instruct-v0.2 and TinyLlama with
    the server's defaults, so without --reasoning-parser): every reader
    question was answered 400 "thinking_token_budget is set but
    reasoning_config is not configured". The reader took the first refusal for
    "no structured output", asked plainly, was refused again, and no session
    was ever read (20 of 20 stores, an hour each)."""
    sent = []

    def server(request):
        body = json.loads(request.content)
        sent.append(body)
        if "thinking_token_budget" in body:
            return httpx.Response(400, json={"error": {
                "message": "thinking_token_budget is set but reasoning_config is not configured. Please set "
                           "--reasoning-parser and/or --reasoning-config to use thinking_token_budget.",
                "type": "BadRequestError", "param": None, "code": 400}})
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"kind": "about_me"})}}],
            "usage": {"completion_tokens": 8}})

    reading_client.NO_BUDGET.clear()
    transport = httpx.MockTransport(server)
    reader = reading_client.Reader(store, "http://plain/v1", "m", server="vllm",
                                   client=httpx.AsyncClient(transport=transport))
    assert await reader.ask("is this about the user?", {"type": "object"}, "preference") == {"kind": "about_me"}
    # Refused once, asked again without the cap; structured output is kept.
    assert [("thinking_token_budget" in b, "response_format" in b) for b in sent] == [(True, True), (False, True)]
    assert reader.structured
    await reader.ask("another question", {"type": "object"}, "review")
    assert len(sent) == 3 and "thinking_token_budget" not in sent[-1]
    # The other thinking controls are sent as before.
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False} and sent[-1]["reasoning_effort"] == "none"
    await reader.close()
    # A later reading of the same server and model starts without the cap;
    # another server is still asked with it.
    again = reading_client.Reader(store, "http://plain/v1", "m", server="vllm",
                                  client=httpx.AsyncClient(transport=transport))
    await again.ask("a third question", {"type": "object"}, "review")
    assert len(sent) == 4 and "thinking_token_budget" not in sent[-1]
    await again.close()
    other = reading_client.Reader(store, "http://model/v1", "m", server="vllm",
                                  client=httpx.AsyncClient(transport=transport))
    assert other.budget
    await other.close()
    reading_client.NO_BUDGET.clear()


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


async def test_each_server_gets_the_reasoning_controls_it_honours(store):
    sent = {}

    def server_for(name):
        def server(request):
            sent[name] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                          "message": {"content": json.dumps({"ok": True})}}]})
        return server

    for name in ("vllm", "mlx", "llamacpp", "lmstudio", "ollama"):
        reader = reading_client.Reader(store, "http://model/v1", f"m-{name}", server=name,
                                       client=httpx.AsyncClient(transport=httpx.MockTransport(server_for(name))))
        await reader.ask(f"question for {name}", {"type": "object"}, "review")
        await reader.close()
    # Probed: the chat-template switch works on vLLM, mlx-lm and llama.cpp ...
    for name in ("vllm", "mlx", "llamacpp"):
        assert sent[name]["chat_template_kwargs"] == {"enable_thinking": False}
    # ... only vLLM caps reasoning with a budget ...
    assert sent["vllm"]["thinking_token_budget"] == reading_client.THINKING_BUDGET[False]
    assert "thinking_token_budget" not in sent["mlx"] and "thinking_token_budget" not in sent["llamacpp"]
    # ... and LM Studio and Ollama get neither (nothing tried worked there).
    for name in ("lmstudio", "ollama"):
        assert "chat_template_kwargs" not in sent[name] and "thinking_token_budget" not in sent[name]
        assert sent[name]["reasoning_effort"] == "none"


async def test_a_model_with_a_small_window_is_asked_for_no_more_than_it_has(store, monkeypatch):
    monkeypatch.setattr(reading_client, "WINDOW", {}, raising=False)
    sent = []

    def answer(body):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": json.dumps({"ok": True})}}]})

    def server(request):
        # vLLM 0.34.0 serving a model with a 4,096-token window.
        body = json.loads(request.content)
        sent.append((body["max_tokens"], "response_format" in body))
        if body["max_tokens"] > 4096:
            return httpx.Response(400, json={"error": {
                "message": f"max_tokens={body['max_tokens']} cannot be greater than max_model_len=4096. "
                           f"Please request fewer output tokens. (parameter=max_tokens, value={body['max_tokens']})",
                "type": "BadRequestError", "param": "max_tokens", "code": 400}})
        return answer(body)

    def reader_for(url, handler=server):
        return reading_client.Reader(store, url, "m", server="vllm",
                                     client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    reader = reader_for("http://small/v1")
    # Refused for 6,000 tokens, asked again for the 4,096 the model has;
    # the refusal is not taken for "no structured output".
    assert await reader.ask("first question", {"type": "object"}, "review") == {"ok": True}
    assert sent == [(6000, True), (4096, True)] and reader.structured
    # A question meant to think asks for three times as much: lowered before it is sent.
    await reader.ask("second question", {"type": "object"}, "classify")
    assert sent[2:] == [(4096, True)]
    await reader.close()
    # Remembered for this server and model ...
    again = reader_for("http://small/v1")
    await again.ask("third question", {"type": "object"}, "review")
    assert sent[3:] == [(4096, True)]
    await again.close()
    # ... and only for them.
    other = reader_for("http://other/v1")
    await other.ask("fourth question", {"type": "object"}, "review")
    assert sent[4:] == [(6000, True), (4096, True)]
    await other.close()

    # A question longer than the window cannot be answered: the moment is
    # given up and remembered, and the reading goes on (it used to fail there
    # at every read). The refusal is not taken for "no structured output".
    asked = []

    def too_long(request):
        asked.append(json.loads(request.content)["messages"][0]["content"])
        if "long session" in asked[-1] or "cut off" in asked[-1]:
            return httpx.Response(400, json={"error": {
                "message": "This model's maximum context length is 4096 tokens. However, your prompt contains at "
                           "least 4097 input tokens, which exceeds this limit even without any output tokens.",
                "type": "BadRequestError", "param": "input_tokens", "code": 400}})
        if "nearly full" in asked[-1]:
            # Little room left for the answer: it is cut off.
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                          "message": {"content": '{"steps": [1, '}}]})
        return answer(None)

    small = reader_for("http://small/v1", too_long)
    gave_up = await small.ask("a long session", {"type": "object"}, "review")
    assert gave_up.get("gave_up") and len(asked) == 1 and small.structured
    assert await small.ask("a long session", {"type": "object"}, "review") == gave_up and len(asked) == 1
    # Cut off, and the forced question (with the reasoning so far) no longer fits.
    assert (await small.ask("a nearly full window", {"type": "object"}, "review")).get("gave_up")
    assert await small.ask("a short session", {"type": "object"}, "review") == {"ok": True}
    await small.close()


async def test_older_vllm_refusing_max_tokens_is_asked_for_what_the_prompt_leaves(store, monkeypatch):
    # Older vLLM words a max_tokens refusal like a prompt over the window
    # ("maximum context length"). Taken for one, every moment was given up and
    # cached for good; it is asked again for the room the prompt leaves.
    monkeypatch.setattr(reading_client, "WINDOW", {}, raising=False)
    sent = []
    wordings = {
        "new": "'max_tokens' or 'max_completion_tokens' is too large: {n}. This model's maximum context "
               "length is 4096 tokens and your request has 100 input tokens ({n} > 4096 - 100).",
        "old": "This model's maximum context length is 4096 tokens. However, you requested {t} tokens "
               "(100 in the messages, {n} in the completion). Please reduce the length of the messages "
               "or completion.",
    }

    def server_for(wording):
        def server(request):
            body = json.loads(request.content)
            sent.append(body["max_tokens"])
            content = body["messages"][0]["content"]
            if "long session" in content:
                return httpx.Response(400, json={"error": {"message": (
                    "This model's maximum context length is 4096 tokens. However, you requested 5000 "
                    "tokens in the messages, Please reduce the length of the messages."), "code": 400}})
            if body["max_tokens"] > 4096 - 100:
                n = body["max_tokens"]
                return httpx.Response(400, json={"error": {
                    "message": wordings[wording].format(n=n, t=n + 100), "code": 400}})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                          "message": {"content": json.dumps({"ok": True})}}]})
        return server

    for wording in wordings:
        sent.clear()
        reader = reading_client.Reader(store, f"http://{wording}/v1", "m", server="vllm",
                                       client=httpx.AsyncClient(transport=httpx.MockTransport(server_for(wording))))
        assert await reader.ask(f"a session ({wording})", {"type": "object"}, "review") == {"ok": True}
        assert sent[-1] == 3996 and reader.structured
        # A prompt longer than the window is still given up.
        assert (await reader.ask(f"a long session ({wording})", {"type": "object"}, "review")).get("gave_up")
        await reader.close()
