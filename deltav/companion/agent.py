"""CompanionAgent — a persistent, self-improving per-user agent.

Wraps the network's ReAct agent with a personal memory layer and a
reflection step. Each turn:
  1. recall this user's relevant memory + learnings, inject a short block;
  2. run the ReAct loop (tools available) on the network;
  3. reflect once — extract a durable learning and store it.

`complete(prompt) -> (text, meta)` is the model call (routes through the
network in production, a script in tests), same contract as overlay.Agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..overlay.agent import Agent, CompleteFn
from ..overlay.tools import ToolRegistry
from .memory import UserMemory

PERSONA = (
    "You are Companion, a helpful personal assistant on the Delta V network. "
    "You are concise and practical. You remember what matters to this user "
    "across conversations and use it. You never reveal or use anyone else's "
    "information."
)

# The reflection asks the model for one durable takeaway; NONE means skip.
REFLECT_INSTRUCTION = (
    "Reflect on the exchange above. In ONE short sentence, state a lasting "
    "preference, fact about the user, or approach that worked — something "
    "worth remembering next time. If there is nothing durable, reply exactly: NONE"
)
_NONE_RE = re.compile(r"^\s*none\b", re.I)


@dataclass
class CompanionStep:
    tool: str
    arguments: dict
    result: str
    receipt_tx: str | None = None


@dataclass
class CompanionResult:
    answer: str
    steps: list[CompanionStep] = field(default_factory=list)
    memory_used: list[str] = field(default_factory=list)
    learned: str | None = None
    model_calls: int = 0


class CompanionAgent:
    def __init__(self, complete: CompleteFn, tools: ToolRegistry,
                 max_steps: int = 6, reflect: bool = True):
        self.complete = complete
        self.tools = tools
        self.max_steps = max_steps
        self.reflect = reflect

    async def run(self, memory: UserMemory, message: str,
                  history: list[dict] | None = None) -> CompanionResult:
        recalled = await memory.recall(message, k=5)
        ctx = memory.context_block(recalled)

        convo = ""
        if history:
            convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]) + "\n"
        task_parts = [f"system: {PERSONA}"]
        if ctx:
            task_parts.append(ctx)
        task = "\n\n".join(task_parts) + f"\n\n{convo}user: {message}"

        agent = Agent(self.complete, self.tools, max_steps=self.max_steps)
        result = await agent.run(task)

        learned = None
        if self.reflect:
            learned = await self._reflect(message, result.answer)
            if learned:
                await memory.learn(learned)

        return CompanionResult(
            answer=result.answer,
            steps=[CompanionStep(s.tool, s.arguments, s.result, s.receipt_tx)
                   for s in result.steps],
            memory_used=[h["text"] for h in recalled],
            learned=learned,
            model_calls=result.model_calls + (1 if learned is not None else 0),
        )

    async def _reflect(self, message: str, answer: str) -> str | None:
        prompt = (f"user: {message}\nassistant: {answer}\n\n"
                  f"system: {REFLECT_INSTRUCTION}\nassistant:")
        try:
            text, _ = await self.complete(prompt)
        except Exception:
            return None
        text = text.strip()
        if not text or _NONE_RE.match(text):
            return None
        # keep it to a single tidy sentence
        return text.split("\n")[0][:240].strip()

    async def feedback(self, memory: UserMemory, note: str) -> dict:
        """Explicit user feedback becomes a durable, high-priority learning."""
        return await memory.learn(f"User feedback: {note}", weight=3.0)
