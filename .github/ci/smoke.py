"""Install smoke test: the built package, installed and set up like a user's.

    python .github/ci/smoke.py --songe <path to the installed songe command> [--browser]

Runs against a fake OpenAI-compatible model server, in a throwaway home and
data folder, so it never touches the machine's own settings or a running
Songe:

1. `songe init --yes` finds the server and connects opencode (a stand-in
   config). The service file is written to a scratch path, never installed.
2. `songe serve` starts; a chat request through Songe reaches the server.
3. `songe status` and the dashboard answer.
4. With --browser, the dashboard is loaded in headless Chromium (Playwright)
   and must render without script errors.
5. `songe uninstall --purge` restores the agent config byte for byte.

Fast by design (well under a minute); a real Dream is tested only before a
release (release.yml).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "tiny-smoke-model"
PORT = 7699  # not Songe's default, so a Songe already running here is left alone


class FakeServer(BaseHTTPRequestHandler):
    """Just enough of an OpenAI-compatible server for Songe to serve and record."""

    def log_message(self, *args):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self.reply({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            return self.send_error(404)
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ):
                chunk |= {"id": "smoke", "object": "chat.completion.chunk", "model": MODEL}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self.reply({
            "id": "smoke",
            "object": "chat.completion",
            "model": request.get("model", MODEL),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "pong"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })


def step(text):
    print(f"--- {text}", flush=True)


def http(url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def wait_until(check, seconds, what):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except OSError as exc:
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {what}: {last}")


def browser_check(url):
    from playwright.sync_api import sync_playwright

    problems = []
    with sync_playwright() as p:
        # CI downloads Chromium; locally SMOKE_BROWSER_CHANNEL=chrome uses Chrome.
        browser = p.chromium.launch(channel=os.environ.get("SMOKE_BROWSER_CHANNEL") or None)
        page = browser.new_page()
        page.on("pageerror", lambda exc: problems.append(f"page error: {exc}"))
        page.on(
            "console",
            lambda msg: problems.append(f"console {msg.type}: {msg.text}")
            if msg.type == "error" else None,
        )
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(3000)  # a few refreshes of /api/state
        text = page.inner_text("body")
        browser.close()
    if problems:
        raise SystemExit("dashboard errors:\n" + "\n".join(problems))
    if MODEL not in text:
        raise SystemExit(f"dashboard does not show {MODEL}:\n{text[:2000]}")
    print("dashboard rendered without errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songe", required=True, help="the installed songe command")
    parser.add_argument("--browser", action="store_true", help="also load the dashboard")
    args = parser.parse_args()

    fake = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    upstream = f"http://127.0.0.1:{fake.server_port}/v1"

    scratch = Path(tempfile.mkdtemp(prefix="songe-smoke-"))
    home, data = scratch / "home", scratch / "data"
    agent = home / ".config/opencode/opencode.json"
    agent.parent.mkdir(parents=True)
    original = b'{"model": "anthropic/claude-sonnet"}\n'
    agent.write_bytes(original)
    env = {
        k: v for k, v in os.environ.items()
        if k not in {"XDG_CONFIG_HOME", "PI_CODING_AGENT_DIR", "SONGE_DATA_DIR"}
    } | {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "LOCALAPPDATA": str(home / "AppData/Local"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "NO_COLOR": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    base = [args.songe, "--data-dir", str(data)]

    def songe(*command, check=True):
        found = subprocess.run(
            base + list(command), env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=180,
        )
        print(found.stdout + found.stderr, flush=True)
        if check and found.returncode:
            raise SystemExit(f"songe {' '.join(command)} exited {found.returncode}")
        return found

    service_file = scratch / "service-file"
    step("songe --version")
    songe("--version")

    step("songe init")
    out = songe("init", "--yes", "--upstream", "vllm", "--upstream-url", upstream,
                "--launchd-target", str(service_file)).stdout
    assert MODEL in out, "init did not list the server's model"
    assert "One step left" in out and "opencode" in out, "init did not say which step is left"
    config = json.loads(agent.read_text(encoding="utf-8"))
    assert "songe" in config.get("provider", {}), "opencode was not connected"
    assert config["model"] == "anthropic/claude-sonnet", "a cloud default must stay"

    # Move the service off the default port before starting it.
    settings = data / "config.toml"
    text = settings.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("port =")]
    settings.write_text(f"port = {PORT}\n" + "\n".join(lines) + "\n", encoding="utf-8")

    step("songe serve")
    log = open(scratch / "serve.log", "w", encoding="utf-8")
    server = subprocess.Popen(base + ["serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{PORT}"
        wait_until(lambda: http(url + "/api/state")[0] == 200, 60, "the service")

        status, body = http(url + "/v1/models")
        ids = [m["id"] for m in json.loads(body)["data"]]
        print("models:", ids)
        public = next((i for i in ids if MODEL in i), None)
        assert public, f"{MODEL} is not offered through Songe"

        step("chat through Songe")
        status, body = http(url + "/v1/chat/completions", {
            "model": public, "messages": [{"role": "user", "content": "ping"}],
        })
        answer = json.loads(body)["choices"][0]["message"]["content"]
        assert answer == "pong", f"unexpected answer {answer!r}"

        step("songe status")
        songe("status")

        status, body = http(url + "/")
        assert status == 200 and b"<html" in body.lower(), "dashboard page missing"
        if args.browser:
            step("dashboard in a browser")
            browser_check(url + "/")
    except BaseException:
        log.close()
        print((scratch / "serve.log").read_text(encoding="utf-8", errors="replace")[-6000:])
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

    step("songe uninstall --purge")
    songe("uninstall", "--purge", "--yes", "--launchd-target", str(service_file))
    assert agent.read_bytes() == original, "uninstall did not restore opencode's config"
    print("smoke test passed")
    fake.shutdown()
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
