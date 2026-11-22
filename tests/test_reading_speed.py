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


async def test_an_answer_cut_off_at_the_limit_is_asked_again_with_more_room(store):
    sent = []

    def server(request):
        body = json.loads(request.content)
        sent.append(body["max_tokens"])
        if len(sent) == 1:  # thinking ran out of room: broken JSON
            return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                          "message": {"content": '{"parts": [{"kind": "feed'}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": json.dumps({"parts": []})}}]})

    reader = reading_client.Reader(store, "http://model/v1", "m",
                                   client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    answer = await reader.ask("question one", {"type": "object"}, "review")
    assert answer == {"parts": []} and sent == [6000, 12000] and reader.length_retries == 1
    # Later questions start with the room this model needed.
    await reader.ask("question two", {"type": "object"}, "review")
    assert sent[-1] == 12000
    await reader.close()
