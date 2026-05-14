"""Minimal chat model for unit tests: supports ``bind_tools`` and scripted ``AIMessage`` replies."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field, PrivateAttr


class ScriptedBindableChatModel(BaseChatModel):
    """
    Deterministic tool-calling test double (fakes do not implement ``bind_tools``).

    Each ``invoke`` pops the next ``AIMessage`` from ``script`` (cycling if exhausted).
    """

    script: list[AIMessage] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted-bindable-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"script_len": len(self.script)}

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.script:
            message = AIMessage(content="(empty script)")
            return ChatResult(generations=[ChatGeneration(message=message)])
        message = self.script[self._cursor % len(self.script)]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        return self
