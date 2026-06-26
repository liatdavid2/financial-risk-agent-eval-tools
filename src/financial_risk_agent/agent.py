from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from financial_risk_agent.cache import CacheClient, stable_hash
from financial_risk_agent.config import RISK_MODEL_PATH, SIMILAR_CASES_INDEX_PATH
from financial_risk_agent.data import normalize_application
from financial_risk_agent.explanation import generate_explanation
from financial_risk_agent.modeling import (
    load_model,
    load_similar_cases_index,
    predict_default_probability,
    search_similar_cases,
)
from financial_risk_agent.policy import apply_policy
from financial_risk_agent.rules import payment_behavior_rules


class RiskAgentState(TypedDict, total=False):
    customer_id: str
    raw_input: dict[str, Any]
    application: dict[str, Any]
    model_score: dict[str, Any]
    rule_result: dict[str, Any]
    similar_cases: list[dict[str, Any]]
    policy_result: dict[str, Any]
    explanation: str
    cache: dict[str, bool]
    tool_trace: list[str]
    latency_ms: float


_CACHE = CacheClient()
_MODEL = None
_INDEX = None


def _trace(state: RiskAgentState, tool_name: str) -> list[str]:
    return [*state.get("tool_trace", []), tool_name]


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(RISK_MODEL_PATH)
    return _MODEL


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = load_similar_cases_index(SIMILAR_CASES_INDEX_PATH)
    return _INDEX


def normalize_node(state: RiskAgentState) -> RiskAgentState:
    raw = state.get("raw_input", {})
    customer_id = str(raw.get("customer_id", raw.get("ID", "unknown")))
    return {
        **state,
        "customer_id": customer_id,
        "application": normalize_application(raw),
        "cache": {"risk_model": False, "similar_cases": False, "explanation": False},
        "tool_trace": _trace(state, "normalize_application"),
    }


def risk_model_node(state: RiskAgentState) -> RiskAgentState:
    application = state["application"]
    key = f"risk_model:{stable_hash(application)}"
    cached = _CACHE.get_json(key)
    cache_state = dict(state.get("cache", {}))

    if cached is not None:
        cache_state["risk_model"] = True
        return {
            **state,
            "model_score": cached,
            "cache": cache_state,
            "tool_trace": _trace(state, "credit_risk_model_predict"),
        }

    probability = predict_default_probability(get_model(), application)
    result = {"default_probability": probability, "model_name": "random_forest_credit_default"}
    _CACHE.set_json(key, result)
    return {
        **state,
        "model_score": result,
        "cache": cache_state,
        "tool_trace": _trace(state, "credit_risk_model_predict"),
    }


def rules_node(state: RiskAgentState) -> RiskAgentState:
    result = payment_behavior_rules(state["application"])
    return {**state, "rule_result": result, "tool_trace": _trace(state, "payment_behavior_rules")}


def similar_cases_node(state: RiskAgentState) -> RiskAgentState:
    application = state["application"]
    key = f"similar_cases:{stable_hash(application)}"
    cached = _CACHE.get_json(key)
    cache_state = dict(state.get("cache", {}))

    if cached is not None:
        cache_state["similar_cases"] = True
        return {
            **state,
            "similar_cases": cached,
            "cache": cache_state,
            "tool_trace": _trace(state, "similar_cases_search"),
        }

    cases = search_similar_cases(get_index(), application, top_k=5)
    _CACHE.set_json(key, cases)
    return {
        **state,
        "similar_cases": cases,
        "cache": cache_state,
        "tool_trace": _trace(state, "similar_cases_search"),
    }


def policy_node(state: RiskAgentState) -> RiskAgentState:
    probability = float(state["model_score"]["default_probability"])
    reasons = list(state["rule_result"]["rule_reasons"])
    similar_cases = state.get("similar_cases", [])
    result = apply_policy(probability, reasons, similar_cases)
    return {**state, "policy_result": result, "tool_trace": _trace(state, "policy_decision")}


def explanation_node(state: RiskAgentState) -> RiskAgentState:
    payload = {
        "policy": state["policy_result"],
        "score": state["model_score"],
        "rules": state["rule_result"],
        "similar_cases": state.get("similar_cases", []),
    }
    key = f"explanation:{stable_hash(payload)}"
    cached = _CACHE.get_json(key)
    cache_state = dict(state.get("cache", {}))

    if cached is not None:
        cache_state["explanation"] = True
        return {
            **state,
            "explanation": cached,
            "cache": cache_state,
            "tool_trace": _trace(state, "explanation_generator"),
        }

    explanation = generate_explanation(
        decision=state["policy_result"]["decision"],
        risk_level=state["policy_result"]["risk_level"],
        default_probability=float(state["model_score"]["default_probability"]),
        reasons=list(state["rule_result"]["rule_reasons"]),
        similar_cases=state.get("similar_cases", []),
    )
    _CACHE.set_json(key, explanation)
    return {
        **state,
        "explanation": explanation,
        "cache": cache_state,
        "tool_trace": _trace(state, "explanation_generator"),
    }


def build_graph():
    graph = StateGraph(RiskAgentState)
    graph.add_node("normalize_application", normalize_node)
    graph.add_node("credit_risk_model_predict", risk_model_node)
    graph.add_node("payment_behavior_rules", rules_node)
    graph.add_node("similar_cases_search", similar_cases_node)
    graph.add_node("policy_decision", policy_node)
    graph.add_node("explanation_generator", explanation_node)

    graph.add_edge(START, "normalize_application")
    graph.add_edge("normalize_application", "credit_risk_model_predict")
    graph.add_edge("credit_risk_model_predict", "payment_behavior_rules")
    graph.add_edge("payment_behavior_rules", "similar_cases_search")
    graph.add_edge("similar_cases_search", "policy_decision")
    graph.add_edge("policy_decision", "explanation_generator")
    graph.add_edge("explanation_generator", END)
    return graph.compile()


_APP_GRAPH = build_graph()


def investigate_application(payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = _APP_GRAPH.invoke({"raw_input": payload})
    latency_ms = (time.perf_counter() - start) * 1000

    cache_result = result.get("cache", {})
    heavy_calls_avoided = sum(1 for value in cache_result.values() if value)

    return {
        "customer_id": result["customer_id"],
        "decision": result["policy_result"]["decision"],
        "risk_level": result["policy_result"]["risk_level"],
        "default_probability": round(float(result["model_score"]["default_probability"]), 4),
        "reasons": result["rule_result"]["rule_reasons"],
        "rule_evidence": result["rule_result"]["rule_evidence"],
        "similar_cases": result.get("similar_cases", []),
        "similar_default_rate": result["policy_result"]["similar_default_rate"],
        "policy_version": result["policy_result"]["policy_version"],
        "explanation": result["explanation"],
        "tool_trace": result.get("tool_trace", []),
        "cache": cache_result,
        "cache_backend": _CACHE.backend,
        "heavy_calls_avoided_by_cache": heavy_calls_avoided,
        "latency_ms": round(latency_ms, 2),
    }


def cache_status() -> dict[str, Any]:
    return _CACHE.status()
