"""Multi-step agent: route -> (call tools -> read results)* -> answer."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.guardrails import fix_range_direction, ground_arguments
from app.agent.router import Route, Router
from app.config import Settings
from app.gateway import Principal, ToolGateway
from app.llm.base import LLMError, LLMProvider, LLMResponse, Message, OnToken, ToolCall, ToolSpec
from app.observability import Trace

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты ассистент по рынку квартир на продажу в Алматы. Данные — объявления с krisha.kz.
Правила:
- Любые цифры (цены, количество, медианы) бери только из результатов инструментов. Не выдумывай объявления.
- Если инструмент вернул ошибку, исправь аргументы и попробуй ещё раз или объясни пользователю, чего не хватает.
- Цены переписывай ровно в том виде, в каком их вернул инструмент («45,5 млн ₸», «920 тыс. ₸/м²»).
- Передавай в инструменты только те фильтры, которые назвал пользователь.
- Если пользователь упоминает особенности из текста объявления (вид, паркинг, метро, ремонт, мебель, ЖК), используй search_descriptions.
- Для объявлений давай ссылку в формате Markdown: [заголовок](url). Не больше 5 объявлений, по одной строке на каждое.
- Отвечай кратко и по-русски."""

REFUSAL = "Я отвечаю только на вопросы о продаже квартир в Алматы: поиск, сравнение объявлений и статистика по районам."
NO_ACCESS = "Для такого запроса у вашей роли нет доступа к нужным инструментам."
FALLBACK_CLARIFICATION = "Уточните, пожалуйста, запрос: какой район, сколько комнат и какой бюджет?"
FINAL_NUDGE = "Лимит шагов исчерпан. Ответь пользователю по уже полученным данным, без новых вызовов инструментов."
EMPTY_ANSWER = "Не получилось сформировать ответ. Попробуйте переформулировать запрос."
FORCED_CALL_PROMPT = "Выбери один вызов инструмента, который отвечает на запрос пользователя.\nИнструменты:\n{tools}"

StopReason = Literal["answer", "max_steps", "clarify", "refused", "no_access", "error"]


@dataclass
class AgentEvent:
    type: Literal["route", "tool_call", "tool_result", "token"]
    data: dict[str, Any]


Emit = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AgentResult:
    answer: str
    stop_reason: StopReason
    intent: str | None = None
    confidence: float | None = None
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


async def _noop(_: AgentEvent) -> None:
    pass


class Agent:
    def __init__(self, llm: LLMProvider, gateway: ToolGateway, settings: Settings, router: Router | None = None):
        self.llm = llm
        self.gateway = gateway
        self.settings = settings
        self.router = router or Router(llm)

    async def run(
        self,
        *,
        principal: Principal,
        history: list[Message],
        user_message: str,
        trace: Trace,
        emit: Emit = _noop,
        stream: bool = False,
    ) -> AgentResult:
        async with trace.span("agent", "agent", input=user_message) as root:
            result = await self._run(principal, history, user_message, trace, emit, stream)
            root.output = result.answer
            return result

    async def _run(
        self,
        principal: Principal,
        history: list[Message],
        user_message: str,
        trace: Trace,
        emit: Emit,
        stream: bool,
    ) -> AgentResult:
        # 1. Route
        allowed = {t.name for t in self.gateway.tools_for(principal)}
        tool_names: set[str] = allowed
        intent, confidence = None, None
        if self.settings.router_enabled:
            decision = await self._route(trace, history, user_message)
            if decision is not None:
                intent, confidence = decision.intent, decision.confidence
                await emit(AgentEvent("route", {"intent": intent, "confidence": confidence, "source": decision.source}))
                if decision.intent == "out_of_scope" and confidence >= self.settings.router_min_confidence:
                    return AgentResult(REFUSAL, "refused", intent, confidence)
                if confidence < self.settings.router_min_confidence:
                    question = decision.clarifying_question or FALLBACK_CLARIFICATION
                    return AgentResult(question, "clarify", intent, confidence)
                if decision.tools and not decision.tools & allowed:
                    return AgentResult(NO_ACCESS, "no_access", intent, confidence)
                tool_names = decision.tools & allowed

        tools = self.gateway.tools_for(principal, only=tool_names)
        messages = [Message("system", SYSTEM_PROMPT), *history, Message("user", user_message)]
        result = AgentResult("", "answer", intent, confidence)
        user_texts = [m.content for m in history if m.role == "user"] + [user_message]
        # Identical repeated calls are answered from here instead of looping on the tool
        seen_calls: dict[str, str] = {}
        used_forced_call = False

        async def on_token(delta: str) -> None:
            await emit(AgentEvent("token", {"text": delta}))

        # 2. Tool loop
        for step in range(1, self.settings.agent_max_steps + 1):
            result.steps = step
            response = await self._generate(
                trace, f"step-{step}", messages, tools or None, on_token if stream else None
            )
            if not response.tool_calls:
                if response.text.strip() or not tools or used_forced_call:
                    return self._finish(result, response.text)
                # Small models sometimes return nothing at all instead of a tool call.
                # Ask again with structured output, which constrains decoding to a valid call.
                used_forced_call = True
                forced = await self._forced_tool_call(trace, messages, tools)
                if forced is None:
                    return self._finish(result, "")
                response.tool_calls = [forced]

            messages.append(Message("assistant", response.text, tool_calls=response.tool_calls))
            for call in response.tool_calls:
                call.arguments, dropped = ground_arguments(call.arguments, user_texts)
                call.arguments, moved = fix_range_direction(call.arguments, user_texts)
                if dropped or moved:
                    log.info("guardrails on %s: dropped %s, moved %s", call.name, dropped, moved)
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                await emit(AgentEvent("tool_call", {"name": call.name, "arguments": call.arguments}))
                async with trace.span("tool", call.name, input=call.arguments) as span:
                    if key in seen_calls:
                        content, ok = seen_calls[key], True
                        span.output = "(repeated call, cached result)"
                    else:
                        outcome = await self.gateway.call(principal, call)
                        content, ok = outcome.content, outcome.ok
                        span.output, span.error = content[:2000], outcome.error
                        if ok:
                            seen_calls[key] = content
                result.tool_calls.append({"name": call.name, "arguments": call.arguments, "ok": ok})
                await emit(AgentEvent("tool_result", {"name": call.name, "ok": ok, "preview": content[:300]}))
                messages.append(Message("tool", content, tool_call_id=call.id, name=call.name))

        # 3. Out of steps: force an answer from what was gathered
        messages.append(Message("user", FINAL_NUDGE))
        response = await self._generate(trace, "final", messages, None, on_token if stream else None)
        result.stop_reason = "max_steps"
        return self._finish(result, response.text)

    @staticmethod
    def _finish(result: AgentResult, text: str) -> AgentResult:
        result.answer = text.strip()
        if not result.answer:
            result.answer, result.stop_reason = EMPTY_ANSWER, "error"
        return result

    async def _route(self, trace: Trace, history: list[Message], user_message: str) -> Route | None:
        async with trace.span("generation", "router", input=user_message) as span:
            try:
                decision = await self.router.route(history, user_message)
            except (LLMError, ValueError) as exc:
                # Routing is an optimisation: without it the agent still works with all allowed tools
                log.warning("router failed, continuing without it: %s", exc)
                span.error = str(exc)
                return None
            span.record_generation(
                decision.model, decision.usage, {"intent": decision.intent, "confidence": decision.confidence}
            )
            if decision.source == "semantic":
                span.kind = "span"  # no LLM call was made, so it must not count as one
            return decision

    async def _forced_tool_call(self, trace: Trace, messages: list[Message], tools: list[ToolSpec]) -> ToolCall | None:
        schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [t.name for t in tools]},
                "arguments": {"anyOf": [t.parameters for t in tools]},
            },
            "required": ["tool", "arguments"],
        }
        catalog = "\n".join(
            f"- {t.name}: {t.description} Аргументы: {json.dumps(t.parameters, ensure_ascii=False)}" for t in tools
        )
        prompt = [*messages, Message("user", FORCED_CALL_PROMPT.format(tools=catalog))]
        async with trace.span("generation", "forced-tool-call") as span:
            try:
                response = await self.llm.chat(prompt, response_schema=schema)
            except LLMError as exc:
                span.error = str(exc)
                return None
            span.record_generation(response.model, response.usage, response.parsed)
        parsed = response.parsed or {}
        if parsed.get("tool") not in {t.name for t in tools} or not isinstance(parsed.get("arguments"), dict):
            return None
        # Constrained decoding fills every optional field; nulls mean "not set"
        arguments = {k: v for k, v in parsed["arguments"].items() if v is not None}
        return ToolCall(id=f"forced-{trace.id[:8]}", name=parsed["tool"], arguments=arguments)

    async def _generate(
        self,
        trace: Trace,
        name: str,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        on_token: OnToken | None,
    ) -> LLMResponse:
        async with trace.span("generation", name, input={"messages": len(messages)}) as span:
            response = await self.llm.chat(messages, tools=tools, on_token=on_token)
            span.record_generation(
                response.model,
                response.usage,
                {"text": response.text[:500], "tool_calls": [c.name for c in response.tool_calls]},
            )
            return response
