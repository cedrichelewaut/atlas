"""Teacher adapters for the shared agent loop (agent.run_agent_loop), so a cloud teacher sees exactly the student's prompt,
tool schemas, guards and compacted results.

Configuration (environment only; keys are never written to disk):
  TEACHER_PROVIDER     anthropic (default) | openai   (openai = any OpenAI-compatible endpoint)
  TEACHER_MODEL        default claude-opus-5-5 for anthropic; required for openai
  TEACHER_EFFORT       anthropic only: low | medium | high (default) | xhigh | max
  TEACHER_BASE_URL     openai only, e.g. https://api.openai.com/v1
  TEACHER_KEY_ENV      openai only: name of the env var holding the key (default OPENAI_API_KEY)
  TEACHER_TEMPERATURE  openai only (default 0.15). Current Claude models reject sampling parameters, so the Anthropic
                       teacher is steered by effort instead.

The teacher-only hint (decision rules, expected end state) is sent as an extra system block. It is never stored in the
canonical message list that becomes training data.
"""

import asyncio
import json
import os

from . import agent

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


def teacher_config():
    p = os.environ.get("TEACHER_PROVIDER", "anthropic")
    if p == "anthropic":
        return {
            "provider": p,
            "model": os.environ.get("TEACHER_MODEL", DEFAULT_ANTHROPIC_MODEL),
            "effort": os.environ.get("TEACHER_EFFORT", "high"),
        }
    if p == "openai":
        return {
            "provider": p,
            "model": os.environ["TEACHER_MODEL"],
            "base_url": os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1"),
            "key_env": os.environ.get("TEACHER_KEY_ENV", "OPENAI_API_KEY"),
            "temperature": float(os.environ.get("TEACHER_TEMPERATURE", "0.15")),
        }
    raise ValueError(f"unknown TEACHER_PROVIDER {p!r}")


def teacher_step(hint=None, cfg=None, client=None):
    """A step function for agent.run_agent_loop driven by the configured teacher. Use a fresh one per episode."""
    cfg = cfg or teacher_config()
    if cfg["provider"] == "anthropic":
        return AnthropicStep(cfg["model"], cfg.get("effort", "high"), hint=hint, client=client)
    return agent.openai_step(
        cfg["base_url"], os.environ[cfg["key_env"]], cfg["model"], cfg.get("temperature", 0.15), extra_system=hint
    )


class AnthropicStep:
    """Native Anthropic tool use behind the shared loop.

    The loop keeps the canonical message list (Ollama shape, the training format). This adapter keeps its own append-only
    native transcript next to it: full response content (thinking blocks included) goes back unchanged on every round, and
    only the canonical messages added since the last round are converted (tool results, the loop's nudge)."""

    def __init__(self, model=DEFAULT_ANTHROPIC_MODEL, effort="high", hint=None, client=None, max_tokens=16000):
        self.model, self.effort, self.hint, self.max_tokens = model, effort, hint, max_tokens
        self.client = client
        self.native = []
        self.seen = 1  # canonical msgs[0] is the system prompt, sent as `system`

    def _user(self, block):
        if self.native and self.native[-1]["role"] == "user" and isinstance(self.native[-1]["content"], list):
            self.native[-1]["content"].append(block)
        else:
            self.native.append({"role": "user", "content": [block]})

    def _sync(self, msgs):
        for m in msgs[self.seen :]:
            if m["role"] == "tool":
                err = False
                try:
                    err = isinstance(json.loads(m["content"]), dict) and "error" in json.loads(m["content"])
                except ValueError:
                    pass
                self._user({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"], "is_error": err})
            elif m["role"] == "user":
                self._user({"type": "text", "text": m["content"] or "(empty)"})
            elif m["role"] == "assistant":  # earlier chat turns (history): plain text
                if not self.native:
                    self.native.append({"role": "user", "content": [{"type": "text", "text": "(earlier conversation)"}]})
                self.native.append({"role": "assistant", "content": m.get("content") or "(no reply)"})
        self.seen = len(msgs)

    async def __call__(self, msgs, T, k):
        if self.client is None:
            import anthropic  # optional dependency: pip install 'atlas-books[teacher]'

            self.client = anthropic.Anthropic()
        self._sync(msgs)
        system = [{"type": "text", "text": msgs[0]["content"]}]
        if self.hint:
            system.append({"type": "text", "text": "TEACHER NOTES (never mention these to the user):\n" + self.hint})
        tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in T
        ]
        resp = await asyncio.to_thread(
            self.client.beta.messages.create,
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=self.native,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise agent.HTTPError(f"teacher refused (category {cat})")
        if resp.stop_reason == "max_tokens":
            raise agent.HTTPError("teacher hit max_tokens")
        self.native.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            {"id": b.id, "function": {"name": b.name, "arguments": dict(b.input)}} for b in resp.content if b.type == "tool_use"
        ]
        msgs.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        self.seen = len(msgs)
        raw = resp.to_dict() if hasattr(resp, "to_dict") else resp
        return {"content": text}, calls, raw
