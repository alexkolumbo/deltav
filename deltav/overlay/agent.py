"""Server-side agent: a ReAct loop over the decentralized network.

Each reasoning step is one paid inference on some node (its receipt hash
is recorded in the step), tool executions happen at the gateway. The
`complete` callable abstracts the model: in production it routes through
SmartRouter, in tests it's a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .toolcall import build_tool_system_prompt, parse_tool_calls, strip_tool_calls
from .tools import ToolRegistry

# complete(prompt) -> (text, meta) where meta may carry node/receipt info
CompleteFn = Callable[[str], Awaitable[tuple[str, dict]]]


@dataclass
class AgentStep:
    tool: str
    arguments: dict
    result: str
    node: str = ""
    receipt_tx: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    model_calls: int = 0
    finished: bool = True  # False when max_steps ran out


class Agent:
    def __init__(self, complete: CompleteFn, tools: ToolRegistry, max_steps: int = 6):
        self.complete = complete
        self.tools = tools
        self.max_steps = max_steps

    async def run(self, task: str) -> AgentResult:
        system = build_tool_system_prompt(self.tools.specs_openai())
        prompt = f"system: {system}\nuser: {task}\nassistant:"
        result = AgentResult(answer="")

        for _ in range(self.max_steps):
            text, meta = await self.complete(prompt)
            result.model_calls += 1
            calls = parse_tool_calls(text)
            if not calls:
                result.answer = text.strip()
                return result

            call = calls[0]  # one tool per step keeps the loop auditable
            observation = await self.tools.execute(call["name"], call["arguments"])
            result.steps.append(AgentStep(
                tool=call["name"],
                arguments=call["arguments"],
                result=observation,
                node=meta.get("node", ""),
                receipt_tx=meta.get("receipt_tx"),
            ))
            prompt += (
                f" {strip_tool_calls(text)}\n"
                f"tool ({call['name']}): {observation}\n"
                "assistant:"
            )

        # Out of steps. Throwing the work away and returning an error string is
        # the worst outcome — the loop has already gathered N observations. Make
        # ONE final tool-free call so the user gets a real answer built from
        # them; finished stays False so the caller can still escalate.
        result.finished = False
        prompt += (
            " \ntool (budget): no steps left. Answer the user NOW using the "
            "observations above. Do NOT call any tool.\nassistant:"
        )
        try:
            text, _ = await self.complete(prompt)
            result.model_calls += 1
            answer = strip_tool_calls(text).strip()
        except Exception:                      # a failed wrap-up must not mask the run
            answer = ""
        result.answer = answer or "(agent stopped: step limit reached)"
        return result
