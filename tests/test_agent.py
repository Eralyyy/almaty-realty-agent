"""Agent loop tests with a scripted fake LLM and fake gateway: no model, MCP or database needed."""

import json
from typing import Any

import pytest

from app.agent.guardrails import fix_range_direction, ground_arguments
from app.agent.loop import EMPTY_ANSWER, NO_ACCESS, REFUSAL, Agent
from app.agent.router import Router, ground_intent
from app.config import Settings
from app.gateway import AccessDenied, AccessPolicy, Principal, ToolOutcome
from app.llm.base import LLMResponse, Message, ToolCall, ToolSpec, Usage
from app.observability import Tracer


class FakeLLM:
    """Returns queued responses; a response with `parsed` answers structured-output calls."""

    model = "fake"

    def __init__(self, route: dict[str, Any] | None, steps: list[LLMResponse]):
        self.route = route
        self.steps = steps
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, *, tools=None, response_schema=None, on_token=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "schema": response_schema})
        if response_schema and "intent" in response_schema["properties"]:
            return LLMResponse("", [], Usage(10, 5), "fake", parsed=self.route)
        response = self.steps.pop(0)
        if on_token and response.text:
            await on_token(response.text)
        return response


class FakeGateway:
    def __init__(self, results: dict[str, str] | None = None):
        self.results = results or {}
        self.executed: list[ToolCall] = []
        self.policy = AccessPolicy()

    def tools_for(self, principal: Principal, only: set[str] | None = None) -> list[ToolSpec]:
        names = {"search_listings", "search_descriptions", "district_stats", "get_listing", "compare_listings", "run_sql"}
        return [
            ToolSpec(n, n, {"type": "object", "properties": {}})
            for n in sorted(names)
            if self.policy.allowed(principal, n) and (only is None or n in only)
        ]

    async def call(self, principal: Principal, call: ToolCall) -> ToolOutcome:
        self.executed.append(call)
        return ToolOutcome(ok=True, content=self.results.get(call.name, "{}"))


def text(value: str) -> LLMResponse:
    return LLMResponse(value, [], Usage(100, 20), "fake")


def tool_call(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"call-{name}", name, arguments)], Usage(100, 20), "fake")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(agent_max_steps=3, router_min_confidence=0.6, trace_log_path=str(tmp_path / "t.jsonl"))


@pytest.fixture
def trace(settings):
    return Tracer(settings.trace_log_path).trace("test", user_id="u", session_id="s")


AGENT = Principal.for_role("u", "agent")
GUEST = Principal.for_role("g", "guest")


async def test_calls_tool_then_answers(settings, trace):
    llm = FakeLLM(
        {"intent": "search", "confidence": 0.9, "clarifying_question": ""},
        [tool_call("search_listings", district="Медеуский", rooms=2), text("Нашёл 3 квартиры")],
    )
    gateway = FakeGateway({"search_listings": json.dumps({"total_matches": 3})})
    result = await Agent(llm, gateway, settings).run(
        principal=AGENT, history=[], user_message="двушка в Медеуском", trace=trace
    )

    assert result.answer == "Нашёл 3 квартиры"
    assert result.stop_reason == "answer"
    assert [c.name for c in gateway.executed] == ["search_listings"]
    # The router narrowed the tools offered to the model
    assert {t.name for t in llm.calls[1]["tools"]} == {"search_listings", "search_descriptions", "get_listing"}
    # The tool result was fed back to the model
    assert llm.calls[2]["messages"][-1].role == "tool"
    assert trace.summary()["llm_calls"] == 3


async def test_low_confidence_asks_to_clarify(settings, trace):
    llm = FakeLLM({"intent": "search", "confidence": 0.3, "clarifying_question": "Какой район?"}, [])
    result = await Agent(llm, FakeGateway(), settings).run(
        principal=AGENT, history=[], user_message="квартира", trace=trace
    )
    assert (result.stop_reason, result.answer) == ("clarify", "Какой район?")


async def test_out_of_scope_is_refused_without_tools(settings, trace):
    llm = FakeLLM({"intent": "out_of_scope", "confidence": 0.95, "clarifying_question": ""}, [])
    gateway = FakeGateway()
    result = await Agent(llm, gateway, settings).run(
        principal=AGENT, history=[], user_message="погода?", trace=trace
    )
    assert (result.stop_reason, result.answer) == ("refused", REFUSAL)
    assert gateway.executed == []


async def test_role_without_needed_tools_gets_no_access(settings, trace):
    llm = FakeLLM({"intent": "sql", "confidence": 0.9, "clarifying_question": ""}, [])
    result = await Agent(llm, FakeGateway(), settings).run(
        principal=GUEST, history=[], user_message="select 1", trace=trace
    )
    assert (result.stop_reason, result.answer) == ("no_access", NO_ACCESS)


async def test_stops_after_max_steps_and_answers_from_gathered_data(settings, trace):
    looping = [tool_call("search_listings", rooms=n) for n in range(1, 4)]
    llm = FakeLLM({"intent": "search", "confidence": 0.9, "clarifying_question": ""}, [*looping, text("Итог")])
    result = await Agent(llm, FakeGateway(), settings).run(
        principal=AGENT, history=[], user_message="ищи", trace=trace
    )
    assert result.stop_reason == "max_steps"
    assert result.answer == "Итог"
    assert llm.calls[-1]["tools"] is None  # the final call cannot request more tools


async def test_repeated_identical_call_is_not_executed_twice(settings, trace):
    llm = FakeLLM(
        {"intent": "stats", "confidence": 0.9, "clarifying_question": ""},
        [tool_call("district_stats"), tool_call("district_stats"), text("ok")],
    )
    gateway = FakeGateway()
    await Agent(llm, gateway, settings).run(principal=AGENT, history=[], user_message="статистика", trace=trace)
    assert len(gateway.executed) == 1


async def test_empty_response_falls_back_to_forced_tool_call(settings, trace):
    forced = LLMResponse("", [], Usage(50, 10), "fake", parsed={"tool": "district_stats", "arguments": {"rooms": None}})
    llm = FakeLLM({"intent": "stats", "confidence": 0.9, "clarifying_question": ""}, [text(""), forced, text("Медиана ...")])
    gateway = FakeGateway()
    result = await Agent(llm, gateway, settings).run(
        principal=AGENT, history=[], user_message="статистика по районам", trace=trace
    )
    assert [c.name for c in gateway.executed] == ["district_stats"]
    assert gateway.executed[0].arguments == {}  # nulls from constrained decoding are dropped
    assert result.answer == "Медиана ..."


async def test_empty_final_answer_is_replaced(settings, trace):
    llm = FakeLLM({"intent": "chitchat", "confidence": 0.9, "clarifying_question": ""}, [text("")])
    result = await Agent(llm, FakeGateway(), settings).run(
        principal=AGENT, history=[], user_message="привет", trace=trace
    )
    assert (result.stop_reason, result.answer) == ("error", EMPTY_ANSWER)


async def test_streaming_emits_tokens(settings, trace):
    llm = FakeLLM({"intent": "chitchat", "confidence": 0.9, "clarifying_question": ""}, [text("Привет!")])
    events = []

    async def emit(event):
        events.append(event)

    await Agent(llm, FakeGateway(), settings).run(
        principal=AGENT, history=[Message("user", "x")], user_message="привет", trace=trace, emit=emit, stream=True
    )
    assert [e.type for e in events] == ["route", "token"]


def test_ungrounded_free_text_filters_are_dropped():
    args, dropped = ground_arguments(
        {"district": "Бостандыкский", "house_type": "монолитный", "rooms": 2}, ["двушка в Бостандыкском"]
    )
    assert args == {"district": "Бостандыкский", "rooms": 2}
    assert dropped == ["house_type"]


def test_grounding_uses_earlier_user_turns():
    args, dropped = ground_arguments({"district": "Медеуский"}, ["квартиры в Медеуском", "а двушки?"])
    assert dropped == []


def test_upper_bound_written_as_lower_is_moved():
    args, moved = fix_range_direction({"price_from": 40_000_000, "rooms": 2}, ["двушка до 40 млн"])
    assert args == {"price_to": 40_000_000, "rooms": 2}
    assert moved == ["price_from->price_to"]


def test_explicit_range_is_left_alone():
    args, moved = fix_range_direction({"price_from": 30_000_000, "price_to": 40_000_000}, ["от 30 до 40 млн"])
    assert moved == []


def test_policy_hides_and_denies_admin_tools():
    policy = AccessPolicy()
    assert not policy.allowed(GUEST, "run_sql")
    with pytest.raises(AccessDenied):
        policy.authorize(GUEST, "run_sql", {"query": "select 1"})


def test_policy_caps_result_size_by_attribute():
    policy = AccessPolicy()
    assert policy.authorize(GUEST, "search_listings", {"limit": 15})["limit"] == 5
    assert policy.authorize(AGENT, "search_listings", {"limit": 15})["limit"] == 15


def test_sql_intent_needs_sql_in_the_message():
    assert ground_intent("sql", "Напиши функцию сортировки на Python") == "out_of_scope"
    assert ground_intent("sql", "Выполни SQL: select 1") == "sql"
    assert ground_intent("search", "двушка") == "search"


class FakeSemantic:
    def __init__(self, intent: str, margin: float):
        self.result = (intent, margin)

    async def classify(self, message: str) -> tuple[str, float]:
        return self.result


async def test_confident_semantic_router_skips_the_llm():
    llm = FakeLLM({"intent": "search", "confidence": 1.0, "clarifying_question": ""}, [])
    route = await Router(llm, FakeSemantic("stats", 0.3), min_margin=0.1).route([], "средняя цена")
    assert (route.intent, route.source) == ("stats", "semantic")
    assert llm.calls == []


async def test_unsure_semantic_router_defers_to_the_llm():
    llm = FakeLLM({"intent": "search", "confidence": 0.9, "clarifying_question": ""}, [])
    route = await Router(llm, FakeSemantic("stats", 0.02), min_margin=0.1).route([], "а в Медеуском?")
    assert (route.intent, route.source) == ("search", "llm")
