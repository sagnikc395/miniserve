"""Terminal chat UI for a running miniserve server.

This is a client, in its own process: the server owns the model, the KV cache
and the device, this process only speaks HTTP to it over
`/v1/chat/completions` with `stream=true`. Point it at a server that is
already up, or pass `--spawn` to start one as a child process and shut it down
again on exit.

    python tui.py                                   # talk to 127.0.0.1:8000
    python tui.py --spawn --model Qwen/Qwen3-0.6B   # bring the server along
    python tui.py --spawn --serve-args='--device cpu --max-num-seqs 4'

Keys: enter sends, escape stops the running generation, ctrl+r starts a fresh
conversation, ctrl+c quits.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    import httpx
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "the tui needs httpx; install it with `uv sync` (dev group)"
    ) from exc

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll
    from textual.widgets import Footer, Header, Input, Static
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "the tui needs textual; install it with `uv sync` (dev group)"
    ) from exc


DEFAULT_SYSTEM = "You are miniserve, a concise and helpful assistant."


@dataclass
class ChatOptions:
    base_url: str = "http://127.0.0.1:8000"
    model: str | None = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    system: str = DEFAULT_SYSTEM


@dataclass
class ServerProcess:
    """A `miniserve serve` child process owned by this TUI."""

    host: str
    port: int
    serve_args: list[str] = field(default_factory=list)
    proc: subprocess.Popen | None = None
    log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self.log_path = Path(
            os.environ.get("TMPDIR", "/tmp"), f"miniserve-serve-{os.getpid()}.log"
        )
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "serve",
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.serve_args,
        ]
        print(f"[tui] starting server: {shlex.join(cmd)}", file=sys.stderr)
        print(f"[tui] server log: {self.log_path}", file=sys.stderr)
        log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_healthy(self, timeout: float = 600.0) -> dict:
        """Poll /health until the model is loaded, or raise."""
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(
                    f"the server exited with code {self.proc.returncode}; "
                    f"see {self.log_path}"
                )
            try:
                reply = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if reply.status_code == 200:
                    return reply.json()
                last = f"HTTP {reply.status_code}"
            except httpx.HTTPError as exc:
                last = type(exc).__name__
            time.sleep(0.5)
        where = f"; see {self.log_path}" if self.log_path else ""
        raise SystemExit(f"the server never became healthy ({last}){where}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        print("[tui] stopping server", file=sys.stderr)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()


class Message(Static):
    """One turn in the transcript."""

    LABELS = {"user": "you", "assistant": "miniserve", "error": "error"}

    def __init__(self, role: str, text: str = "") -> None:
        self.role = role
        self.text = text
        super().__init__(self._content(), markup=False, classes=f"message {role}")

    def _content(self) -> str:
        body = self.text or ("…" if self.role == "assistant" else "")
        return f"{self.LABELS[self.role]}\n{body}"

    def _refresh_text(self) -> None:
        self.update(self._content())

    def append(self, delta: str) -> None:
        self.text += delta
        self._refresh_text()

    def set_text(self, text: str) -> None:
        self.text = text
        self._refresh_text()


class MiniserveTUI(App):
    """A chat client for miniserve, streaming over the OpenAI-compatible API."""

    CSS = """
    #transcript { padding: 1 2; }
    .message { margin-bottom: 1; }
    .message.user { color: $accent; }
    .message.assistant { color: $text; }
    .message.error { color: $error; }
    #statusbar { height: 1; padding: 0 2; color: $text-muted; }
    #prompt { dock: bottom; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "quit"),
        ("escape", "stop", "stop generating"),
        ("ctrl+r", "reset", "new conversation"),
    ]

    def __init__(self, options: ChatOptions, server: ServerProcess | None = None):
        super().__init__()
        self.options = options
        self.server = server
        self.history: list[dict[str, str]] = []
        self.client: httpx.AsyncClient | None = None
        self.status_text = ""
        self._generating = False

    # layout -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Static("starting…", id="statusbar")
        yield Input(placeholder="ask miniserve…", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "miniserve"
        model = self.options.model or "default model"
        self.sub_title = f"{model} @ {self.options.base_url}"
        self.client = httpx.AsyncClient(
            base_url=self.options.base_url, timeout=httpx.Timeout(600.0, connect=10.0)
        )
        self.query_one("#prompt", Input).focus()
        self.status("ready")

    async def on_unmount(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    # helpers ----------------------------------------------------------------

    def status(self, text: str) -> None:
        self.status_text = text
        self.query_one("#statusbar", Static).update(text)

    def add_message(self, role: str, text: str = "") -> Message:
        message = Message(role, text)
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(message)
        transcript.scroll_end(animate=False)
        return message

    # actions ----------------------------------------------------------------

    @on(Input.Submitted, "#prompt")
    def submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        if self._generating:
            self.status("still generating — press escape to stop")
            return
        event.input.value = ""
        self.add_message("user", prompt)
        self.history.append({"role": "user", "content": prompt})
        self.generate()

    def action_stop(self) -> None:
        if not self._generating:
            return
        self.workers.cancel_group(self, "generate")
        self.status("stopped")
        self._generating = False

    def action_reset(self) -> None:
        self.action_stop()
        self.history.clear()
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.status("new conversation")

    # streaming --------------------------------------------------------------

    @work(group="generate", exclusive=True)
    async def generate(self) -> None:
        assert self.client is not None
        bubble = self.add_message("assistant")
        transcript = self.query_one("#transcript", VerticalScroll)
        self._generating = True
        self.status("generating…")

        messages = [{"role": "system", "content": self.options.system}, *self.history]
        payload = {
            "messages": messages,
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
            "top_p": self.options.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.options.model:
            payload["model"] = self.options.model

        started = time.monotonic()
        ttft: float | None = None
        chunks = 0
        usage: dict | None = None

        try:
            async with self.client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as reply:
                if reply.status_code != 200:
                    body = (await reply.aread()).decode("utf-8", "replace")
                    raise RuntimeError(
                        f"HTTP {reply.status_code}: {body.strip()[:500]}"
                    )

                async for line in reply.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"].get("message", data))
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = (choice.get("delta") or {}).get("content") or ""
                        if not delta:
                            continue
                        if ttft is None:
                            ttft = time.monotonic() - started
                        chunks += 1
                        bubble.append(delta)
                        transcript.scroll_end(animate=False)
        except Exception as exc:  # network, protocol or server-side failure
            self._generating = False
            if not bubble.text:
                bubble.set_text("(no output)")
            self.add_message("error", str(exc))
            self.status("failed")
            return

        self._generating = False
        if bubble.text:
            self.history.append({"role": "assistant", "content": bubble.text})
        self.status(self._summary(started, ttft, chunks, usage))

    @staticmethod
    def _summary(
        started: float, ttft: float | None, chunks: int, usage: dict | None
    ) -> str:
        elapsed = time.monotonic() - started
        tokens = usage["completion_tokens"] if usage else chunks
        parts = [f"{tokens} tok in {elapsed:.1f}s"]
        if elapsed > 0:
            parts.append(f"{tokens / elapsed:.1f} tok/s")
        if ttft is not None:
            parts.append(f"ttft {ttft * 1000:.0f} ms")
        if usage:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            parts.append(f"prompt {usage['prompt_tokens']} ({cached} cached)")
        return " · ".join(parts)


def run_tui(
    *,
    base_url: str = "http://127.0.0.1:8000",
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    system: str = DEFAULT_SYSTEM,
    spawn: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    serve_args: list[str] | None = None,
) -> None:
    """Start the TUI, optionally spawning a server process for it first."""
    server: ServerProcess | None = None
    if spawn:
        server = ServerProcess(host=host, port=port, serve_args=list(serve_args or []))
        server.start()
        print("[tui] waiting for the model to load…", file=sys.stderr)
        health = server.wait_healthy()
        print(f"[tui] server ready: {health.get('model')}", file=sys.stderr)
        base_url = server.base_url
        model = model or health.get("model")

    options = ChatOptions(
        base_url=base_url.rstrip("/"),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system=system,
    )
    try:
        MiniserveTUI(options, server).run()
    finally:
        if server is not None:
            server.stop()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None, help="model id sent with each request")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="start `main.py serve` as a child process and stop it on exit",
    )
    parser.add_argument("--host", default="127.0.0.1", help="host for --spawn")
    parser.add_argument("--port", type=int, default=8000, help="port for --spawn")
    parser.add_argument(
        "--serve-args",
        default="",
        help="extra flags for the spawned server, e.g. --serve-args='--device cpu'",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    serve_args = shlex.split(args.serve_args)
    if args.spawn and args.model:
        serve_args = ["--model", args.model, *serve_args]
    run_tui(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system=args.system,
        spawn=args.spawn,
        host=args.host,
        port=args.port,
        serve_args=serve_args,
    )


if __name__ == "__main__":
    main()
