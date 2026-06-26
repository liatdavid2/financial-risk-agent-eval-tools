# הסבר קצר בעברית לראיון

שם ריפו מומלץ:

```text
financial-risk-agent-eval-tools
```

## מה הפרויקט עושה?

הפרויקט בונה Agent פיננסי שבודק בקשות אשראי על בסיס הדאטה:

```text
Default of Credit Card Clients / UCI_Credit_Card.csv
```

ה-Agent לא נותן החלטה רק מתוך LLM. הוא קודם מפעיל כלים ברורים:

1. נרמול קלט
2. מודל ML להסתברות default
3. חוקים פיננסיים שקופים
4. חיפוש מקרים דומים
5. מדיניות החלטה
6. יצירת הסבר מבוסס evidence

בסוף מתקבלת החלטה:

```text
approve / manual_review / reject
```

עם הסתברות, סיבות, מקרים דומים, הסבר, latency ו־cache status.

---

## איפה LangGraph נכנס?

LangGraph מנהל את ה-flow:

```text
normalize_application
 -> credit_risk_model_predict
 -> payment_behavior_rules
 -> similar_cases_search
 -> policy_decision
 -> explanation_generator
```

כל שלב הוא tool/node ברור.

---

## איפה Redis נכנס?

Redis משמש להורדת latency ועלות:

```text
risk_model:<hash>        -> prediction cache
similar_cases:<hash>     -> similar cases cache
explanation:<hash>       -> explanation / heavy LLM cache
```

אם Redis לא רץ, יש fallback לזיכרון כדי שהדמו עדיין יעבוד.

---

## מה חשוב ב־GenAI Evaluation?

הפרויקט כולל Golden Dataset שנבנה מהדאטה עצמו.

כל שורה כוללת:

```text
input
expected_decision
expected_reasons
expected_tools
forbidden_claims
```

ואז מריצים Evaluation אוטומטי שבודק:

```text
decision_accuracy
reason_coverage
groundedness_score
hallucination_rate
tool_calling_completeness
tool_order_accuracy
latency
cache_hit_rate
```

זה מדגיש שהבדיקה היא לא רק על מודל ML, אלא על מערכת GenAI Agentic מלאה.

---

## משפט טוב לראיון

```text
Built a Dockerized financial risk investigation agent where LangGraph orchestrates deterministic tools: ML scoring, payment rules, similar-case retrieval, policy decision, and grounded explanation generation. I also added a golden dataset and GenAI evaluation pipeline to measure decision accuracy, reason coverage, tool-calling correctness, groundedness, hallucination rate, latency, and Redis cache hit rate.
```

בעברית:

```text
בניתי Agent פיננסי שבו ה־LLM לא מחליט לבד. LangGraph מפעיל כלים דטרמיניסטיים — מודל סיכון, חוקים פיננסיים, מקרים דומים ומדיניות החלטה — ורק בסוף נוצר הסבר מבוסס evidence. בנוסף בניתי Golden Dataset ו־Evaluation אוטומטי שבודק החלטה, סיבות, שימוש נכון בכלים, groundedness, hallucinations, latency ו־Redis cache.
```
