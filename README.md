# Financial Risk Agent Eval Tools

Production-style **Financial Risk Investigation Agent** built on the Kaggle/UCI **Default of Credit Card Clients** dataset.

Recommended repo name:

```text
financial-risk-agent-eval-tools
```

The project demonstrates three interview-friendly engineering themes:

1. **LangGraph tool-calling workflow** for financial risk investigation.
2. **GenAI Evaluation** with a golden dataset, tool-calling correctness, groundedness, hallucination checks, latency and cache metrics.
3. **Redis caching** to reduce repeated heavy model / explanation / similar-case calls.

> Dataset file expected locally: `data/raw/UCI_Credit_Card.csv`  
> The dataset itself is not included in this repository.

---

## What the agent does

Input: one credit-card customer/application record.

Output:

```text
approve / manual_review / reject
```

with:

- default probability from an ML model
- deterministic payment-behavior reasons
- similar historical cases
- policy decision
- grounded explanation
- LangGraph `tool_trace`
- Redis cache status
- latency

---

## Architecture

```text
FastAPI / CLI
   |
   v
LangGraph Financial Risk Agent
   |
   |-- Tool node 1: normalize_application
   |-- Tool node 2: credit_risk_model_predict
   |-- Tool node 3: payment_behavior_rules
   |-- Tool node 4: similar_cases_search
   |-- Tool node 5: policy_decision
   |-- Tool node 6: explanation_generator
   |
   v
Decision + reasons + evidence + explanation + tool_trace
```

Redis cache keys:

```text
risk_model:<payload_hash>        -> cached ML prediction
similar_cases:<payload_hash>     -> cached nearest-neighbor search
explanation:<payload_hash>       -> cached grounded explanation / heavy LLM call
```

---

## Dataset

Put the Kaggle file here:

```text
data/raw/UCI_Credit_Card.csv
```

Expected columns:

```text
LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
BILL_AMT1..BILL_AMT6,
PAY_AMT1..PAY_AMT6,
default.payment.next.month
```

---

# Docker usage

## 1. Build containers

```bash
docker compose build
```

## 2. Train model inside Docker

```bash
docker compose run --rm api python scripts/train_model.py --csv data/raw/UCI_Credit_Card.csv
```

Creates:

```text
models/risk_model.joblib
models/similar_cases_index.joblib
reports/training_metrics.json
```

## 3. Generate golden dataset

```bash
docker compose run --rm api python scripts/make_golden_dataset.py --csv data/raw/UCI_Credit_Card.csv --limit 300
```

Creates:

```text
data/golden/golden_default_credit.jsonl
```

Each golden row includes:

```json
{
  "customer_id": "C123",
  "input": {"LIMIT_BAL": 20000, "PAY_0": 3, "BILL_AMT1": 18000, "PAY_AMT1": 500},
  "expected_decision": "manual_review",
  "expected_reasons": ["recent_payment_delay", "high_credit_utilization"],
  "expected_tools": [
    "normalize_application",
    "credit_risk_model_predict",
    "payment_behavior_rules",
    "similar_cases_search",
    "policy_decision",
    "explanation_generator"
  ],
  "forbidden_claims": ["fraud", "criminal", "illegal activity"]
}
```

## 4. Run GenAI evaluation

```bash
docker compose run --rm api python scripts/run_eval.py --golden data/golden/golden_default_credit.jsonl
```

Creates:

```text
reports/evals/eval_results.json
```

Main GenAI eval metrics:

```text
decision_accuracy
avg_reason_coverage
avg_groundedness_score
hallucination_rate
avg_tool_calling_completeness
tool_order_accuracy
avg_latency_ms
cache_hit_rate
```

## 5. Run API

```bash
docker compose up api
```

Open Swagger:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Cache status:

```bash
curl http://localhost:8000/cache/status
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_application.json
```

---

# Local non-Docker usage

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

python scripts/train_model.py --csv data/raw/UCI_Credit_Card.csv
python scripts/make_golden_dataset.py --csv data/raw/UCI_Credit_Card.csv --limit 300
python scripts/run_eval.py --golden data/golden/golden_default_credit.jsonl
uvicorn financial_risk_agent.api:app --reload
```

Optional local Redis:

```bash
docker compose up -d redis
```

If Redis is not running, the project uses an in-memory fallback cache.

---

## Example API response

```json
{
  "customer_id": "C-DEMO-001",
  "decision": "manual_review",
  "risk_level": "high",
  "default_probability": 0.71,
  "reasons": [
    "recent_payment_delay",
    "repeated_payment_delay",
    "high_credit_utilization",
    "low_recent_payment"
  ],
  "similar_cases": [
    {"customer_id": "C21810", "similarity": 0.92, "actual_default": 1}
  ],
  "explanation": "The application should be sent to manual review...",
  "tool_trace": [
    "normalize_application",
    "credit_risk_model_predict",
    "payment_behavior_rules",
    "similar_cases_search",
    "policy_decision",
    "explanation_generator"
  ],
  "cache": {
    "risk_model": false,
    "similar_cases": false,
    "explanation": false
  },
  "cache_backend": "redis",
  "latency_ms": 42.5
}
```

---

## Why this is a strong GenAI project

This is not just a chatbot. The LLM/explanation step is not allowed to invent a decision.

The system first collects evidence from deterministic tools:

```text
ML score -> rules -> similar cases -> policy -> grounded explanation
```

Then evaluation checks whether the generated answer is safe and correct:

- Did the decision match the golden expected decision?
- Did the explanation cover the expected reasons?
- Did the agent call the required tools?
- Did it call them in the right order?
- Did it avoid forbidden claims like “fraud” or “criminal”?
- Was the answer grounded in tool outputs?
- How much latency and repeated compute did Redis save?

---

## CV line

```text
Built a Dockerized Financial Risk Agent using LangGraph, FastAPI, Redis, ML risk scoring, similar-case retrieval, and automated GenAI evaluation with a golden dataset for decision accuracy, reason coverage, tool-calling correctness, groundedness, hallucination checks, latency, and cache hit rate.
```

## Interview explanation

```text
The agent does not let the LLM decide from imagination. LangGraph orchestrates deterministic tools: model score, payment rules, similar cases, and a policy layer. The explanation is generated only after the evidence is collected. I also added a golden dataset and automated GenAI evaluation to test decision accuracy, reason coverage, tool-calling correctness, groundedness, hallucination rate, latency, and Redis cache behavior.
```
