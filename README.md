# Financial Risk Agent Eval Tools

Production-style Financial Risk Investigation Agent using **FastAPI**, **LangGraph**, **tool-calling**, **ML credit-risk scoring**, **similar-case retrieval**, **Redis caching**, and a dedicated **GenAI Evaluation pipeline**.

The project is built on the **Default of Credit Card Clients** dataset (`UCI_Credit_Card.csv`) and demonstrates how an AI agent can support credit-risk investigation using structured tools, explainable evidence, and automated evaluation.

> This is an educational / portfolio project. It is not a real credit decision system and should not be used for production financial decisions.

---

## Why This Project

Many GenAI demos are simple chatbots.

This project is different: the agent is grounded in deterministic tools, ML model outputs, policy rules, similar historical cases, and measurable evaluation metrics.

The goal is to demonstrate a production-oriented AI/ML system for financial risk:

* ML model predicts credit default risk.
* LangGraph orchestrates a multi-step tool workflow.
* Tools provide structured evidence.
* Similar historical cases are retrieved.
* Policy logic converts risk signals into an operational decision.
* Explanation is generated from tool outputs.
* GenAI evaluation checks correctness, groundedness, hallucinations, and tool usage.
* Redis caches repeated serving-time computations to reduce latency.

---

## Dataset

The project uses the **Default of Credit Card Clients** dataset.

Expected local file:

```text
data/raw/UCI_Credit_Card.csv
```

Dataset properties:

```text
Rows: 30,000
Target: default.payment.next.month
Positive class: customer defaulted next month
```

The dataset includes financial and behavioral features such as:

* Credit limit
* Age
* Repayment history
* Bill amounts
* Previous payment amounts
* Default label for the next month

The CSV file is not committed to the repository.

---

## Main Capabilities

### 1. ML Credit Risk Scoring

The training pipeline trains a credit default prediction model using tabular financial features.

The project supports benchmarking across several tabular ML models:

* RandomForest
* XGBoost
* LightGBM
* CatBoost

The best model is selected by **PR-AUC**, which is more informative than plain accuracy for financial risk detection because the positive class represents customers likely to default.

---

### 2. LangGraph Agent

The API runs a LangGraph workflow with multiple tools:

```text
normalize_application
credit_risk_model_predict
payment_behavior_rules
similar_cases_search
policy_decision
explanation_generator
```

Each prediction returns a `tool_trace`, allowing the evaluation pipeline to verify:

* Which tools were called
* Whether all required tools were called
* Whether the tools ran in the expected order

---

### 3. Similar Historical Cases

The system builds a similar-cases index from the dataset and retrieves nearest historical customers for each new application.

This supports the risk decision with examples of similar past cases.

---

### 4. Policy Decision

The agent combines:

* ML default probability
* Payment behavior rules
* Similar-case default rate
* Policy thresholds

and returns an operational decision:

```text
approve
manual_review
reject
```

In a real production financial system, high-risk cases may be escalated to manual review instead of being automatically rejected, depending on business and regulatory policy.

---

### 5. Grounded Explanation

The explanation is generated from structured tool outputs, such as:

* Model probability
* Rule evidence
* Payment delay behavior
* Credit utilization
* Similar historical cases

This makes the explanation auditable and easier to evaluate.

---

### 6. GenAI Evaluation

The project includes a golden dataset and an evaluation pipeline that measures:

* Decision accuracy
* Reason coverage
* Groundedness
* Hallucination rate
* Tool-calling completeness
* Tool order accuracy
* Latency
* Redis cache hit rate

This evaluates the full agent workflow, not only the ML model.

---

### 7. Redis Caching

Redis is used at **API serving time**, not during model training.

Redis caches repeated:

* Model predictions
* Similar-case retrieval results
* Generated explanations

Redis improves serving-time latency and avoids repeated computation. It does not improve model quality or training metrics.

---

## Architecture

```text
User / Analyst
    |
    v
FastAPI
    |
    v
LangGraph Agent
    |
    |-- Tool 1: normalize_application
    |-- Tool 2: credit_risk_model_predict
    |-- Tool 3: payment_behavior_rules
    |-- Tool 4: similar_cases_search
    |-- Tool 5: policy_decision
    |-- Tool 6: explanation_generator
    |
    v
Decision + Risk Level + Evidence + Explanation + Tool Trace
```

Runtime caching:

```text
FastAPI / LangGraph Agent
    |
    |-- Redis cache for repeated predictions
    |-- Redis cache for similar-case retrieval
    |-- Redis cache for explanations
```

---

## Project Structure

```text
financial-risk-agent-eval-tools/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
├── PROJECT_OVERVIEW_HE.md
│
├── financial_risk_agent/
│   ├── api.py
│   ├── config.py
│   ├── data.py
│   ├── graph.py
│   ├── schemas.py
│   ├── tools.py
│   ├── policy.py
│   └── cache.py
│
├── scripts/
│   ├── train_model.py
│   ├── benchmark_models.py
│   ├── make_golden_dataset.py
│   └── run_eval.py
│
├── examples/
│   └── sample_application.json
│
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv
│   └── golden/
│
├── models/
│   ├── risk_model.joblib
│   └── similar_cases_index.joblib
│
└── reports/
    ├── training_metrics.json
    ├── model_benchmark.json
    └── evals/
        └── eval_results.json
```

---

## Quick Start

### 1. Place the dataset

Put the downloaded Kaggle CSV here:

```text
data/raw/UCI_Credit_Card.csv
```

---

### 2. Build Docker containers

```bash
docker compose build
```

---

### 3. Train the baseline model

```bash
docker compose run --rm api python scripts/train_model.py --csv data/raw/UCI_Credit_Card.csv
```

This creates:

```text
models/risk_model.joblib
models/similar_cases_index.joblib
reports/training_metrics.json
```

---

### 4. Benchmark multiple models

```bash
docker compose run --rm api python scripts/benchmark_models.py --csv data/raw/UCI_Credit_Card.csv
```

This compares:

```text
RandomForest
XGBoost
LightGBM
CatBoost
```

The best model is selected by PR-AUC and saved to:

```text
models/risk_model.joblib
```

---

### 5. Create the golden dataset

```bash
docker compose run --rm api python scripts/make_golden_dataset.py --csv data/raw/UCI_Credit_Card.csv --limit 300
```

This creates:

```text
data/golden/golden_default_credit.jsonl
```

---

### 6. Clear Redis before evaluating a new model

```bash
docker compose exec redis redis-cli FLUSHALL
```

This avoids using cached results from a previous model.

---

### 7. Run GenAI Evaluation

Cold-cache evaluation:

```bash
docker compose run --rm api python scripts/run_eval.py --golden data/golden/golden_default_credit.jsonl
```

Warm-cache evaluation:

```bash
docker compose run --rm api python scripts/run_eval.py --golden data/golden/golden_default_credit.jsonl
```

The second run uses Redis cache and should be faster.

---

### 8. Run the API

```bash
docker compose up api
```

The API runs at:

```text
http://localhost:8001
```

---

## API Usage

### Health check

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok"
}
```

---

### Get a sample payload

```bash
curl http://localhost:8001/sample-payload
```

---

### Predict risk

Windows CMD:

```cmd
curl -X POST http://localhost:8001/predict ^
  -H "Content-Type: application/json" ^
  -d @examples/sample_application.json
```

PowerShell:

```powershell
curl.exe -X POST http://localhost:8001/predict `
  -H "Content-Type: application/json" `
  -d "@examples/sample_application.json"
```

Linux / macOS:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_application.json
```

---

## Example Prediction Response

```json
{
  "customer_id": "C-DEMO-001",
  "decision": "reject",
  "risk_level": "very_high",
  "default_probability": 0.844,
  "reasons": [
    "recent_payment_delay",
    "repeated_payment_delay",
    "high_credit_utilization",
    "low_recent_payment"
  ],
  "rule_evidence": {
    "PAY_0": 3.0,
    "delayed_months": 3,
    "credit_utilization": 0.925,
    "payment_to_bill_ratio": 0.022
  },
  "similar_cases": [
    {
      "customer_id": "C11404",
      "similarity": 0.9842,
      "actual_default": 0
    },
    {
      "customer_id": "C5323",
      "similarity": 0.9829,
      "actual_default": 0
    },
    {
      "customer_id": "C10708",
      "similarity": 0.9769,
      "actual_default": 1
    }
  ],
  "similar_default_rate": 0.4,
  "policy_version": "credit-risk-policy-v1",
  "explanation": "The application should be rejected or escalated according to policy. The estimated default probability is 84.40%, with overall risk level 'very_high'. The main evidence is: recent repayment delay, repeated repayment delays across multiple months, high utilization of the credit limit, low recent payment compared with the latest bill amount. Among the retrieved similar historical cases, 2 had a default label.",
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
  "heavy_calls_avoided_by_cache": 0,
  "latency_ms": 609.49
}
```

---

## Test Redis Caching

Run the same prediction request twice.

First request:

```json
"cache": {
  "risk_model": false,
  "similar_cases": false,
  "explanation": false
}
```

Second request:

```json
"cache": {
  "risk_model": true,
  "similar_cases": true,
  "explanation": true
}
```

This shows that Redis is caching repeated API-serving computations.

---

## Model Benchmark Results

The project benchmarks several tabular ML models and selects the best one by PR-AUC.

Example benchmark result:

```text
CatBoost      PR-AUC: 0.5603   ROC-AUC: 0.7809
XGBoost       PR-AUC: 0.5556   ROC-AUC: 0.7782
RandomForest  PR-AUC: 0.5532   ROC-AUC: 0.7767
LightGBM      PR-AUC: 0.5522   ROC-AUC: 0.7776
```

Selected model:

```text
CatBoost
```

Why PR-AUC?

For financial risk detection, the positive class represents customers likely to default. PR-AUC focuses more directly on the model's ability to identify risky customers than plain accuracy.

---

## Training Metrics Example

Example training output:

```json
{
  "rows": 30000,
  "positive_rate": 0.2212,
  "test_rows": 7500,
  "roc_auc": 0.7766,
  "average_precision_pr_auc": 0.5531
}
```

The positive class rate is approximately 22%, meaning around 22% of customers in the dataset have a default label.

---

## GenAI Evaluation

The evaluation pipeline uses a golden dataset to test the full agent behavior, not only the ML model.

It checks whether the agent:

* Makes the expected decision
* Includes the expected reasons
* Grounds the explanation in tool outputs
* Avoids hallucinated claims
* Calls the required tools
* Calls tools in the expected order
* Improves latency with Redis caching

---

## Example Evaluation Results

### Cold-cache evaluation

After clearing Redis:

```bash
docker compose exec redis redis-cli FLUSHALL
docker compose run --rm api python scripts/run_eval.py --golden data/golden/golden_default_credit.jsonl
```

Example result:

```json
{
  "total_cases": 300,
  "decision_accuracy": 0.6233,
  "avg_reason_coverage": 1.0,
  "avg_groundedness_score": 1.0,
  "hallucination_rate": 0.0,
  "avg_tool_calling_completeness": 1.0,
  "tool_order_accuracy": 1.0,
  "avg_latency_ms": 16.98,
  "cache_hit_rate": 0.0,
  "elapsed_seconds": 5.1
}
```

### Warm-cache evaluation

Running the same evaluation again:

```json
{
  "total_cases": 300,
  "decision_accuracy": 0.6233,
  "avg_reason_coverage": 1.0,
  "avg_groundedness_score": 1.0,
  "hallucination_rate": 0.0,
  "avg_tool_calling_completeness": 1.0,
  "tool_order_accuracy": 1.0,
  "avg_latency_ms": 2.13,
  "cache_hit_rate": 1.0,
  "elapsed_seconds": 0.65
}
```

### Interpretation

The decision accuracy stayed the same because caching does not change model behavior.

Redis only avoids repeated computation.

The warm-cache run reduced average latency:

```text
16.98ms -> 2.13ms
```

and total evaluation time:

```text
5.1s -> 0.65s
```

This shows a clear serving-time performance improvement.

---

## What Redis Does and Does Not Do

Redis is used for runtime serving optimization.

Redis helps with:

```text
API latency
Repeated prediction calls
Repeated retrieval calls
Repeated explanation generation
Cold-cache vs warm-cache evaluation
```

Redis does not help with:

```text
Model training
Model quality
PR-AUC
ROC-AUC
Decision accuracy
```

Model quality is improved through model benchmarking, threshold tuning, feature engineering, and better policy alignment.

Redis improves serving-time efficiency.

---

## Why This Is a GenAI Evaluation Project

This project is not only about predicting default risk.

The important GenAI part is that the agent is evaluated as a full decision workflow:

```text
Input application
    |
    v
Tool calls
    |
    v
ML score + rules + retrieval + policy
    |
    v
Grounded explanation
    |
    v
Evaluation against golden dataset
```

The evaluation measures both classic decision quality and GenAI-specific behavior:

| Metric                    | Meaning                                           |
| ------------------------- | ------------------------------------------------- |
| Decision accuracy         | Did the agent return the expected decision?       |
| Reason coverage           | Did the explanation include the expected reasons? |
| Groundedness              | Is the explanation supported by tool outputs?     |
| Hallucination rate        | Did the agent invent unsupported claims?          |
| Tool-calling completeness | Did the agent call all required tools?            |
| Tool order accuracy       | Did the tools run in the expected order?          |
| Latency                   | How fast did the workflow run?                    |
| Cache hit rate            | How much repeated work was avoided by Redis?      |

---

## Why This Is Useful for Financial Systems

A financial risk agent needs more than a probability score.

It needs:

* Evidence
* Explanations
* Policy alignment
* Auditability
* Repeatable evaluation
* Latency measurement
* Clear separation between model scoring and business decisioning

This project demonstrates those ideas in a compact, runnable system.

---

## Interview Summary

```text
I built a LangGraph-based Financial Risk Agent on the Default of Credit Card Clients dataset.

The system uses ML credit-risk scoring, payment behavior rules, similar-case retrieval, and policy tools to produce a decision and grounded explanation.

I added a GenAI evaluation pipeline with a golden dataset to measure decision accuracy, reason coverage, groundedness, hallucination rate, tool-calling completeness, tool order accuracy, latency, and Redis cache behavior.

I also benchmarked RandomForest, XGBoost, LightGBM, and CatBoost, selecting CatBoost by PR-AUC as the production scoring model.

Redis is used only at API serving time to cache repeated predictions, retrievals, and explanations, reducing warm-run latency from about 17ms to about 2ms.
```

---

## CV Bullet

```text
Built a LangGraph-based Financial Risk Agent using ML credit-risk scoring, policy tools, similar-case retrieval, and GenAI evaluation to test decision accuracy, groundedness, hallucinations, and tool-calling correctness; benchmarked RandomForest, XGBoost, LightGBM, and CatBoost by PR-AUC and added Redis caching to reduce repeated-call latency at API serving time.
```

---

## Tech Stack

```text
Python
FastAPI
LangGraph
scikit-learn
XGBoost
LightGBM
CatBoost
Redis
Docker
Docker Compose
Pandas
Joblib
```

---

## Notes

* The dataset file should be placed manually under `data/raw/`.
* The model artifact is saved under `models/`.
* Redis is used only for runtime caching.
* GenAI evaluation results are saved under `reports/evals/`.
* Model benchmark results are saved under `reports/model_benchmark.json`.

---

## License

For educational and portfolio use.
