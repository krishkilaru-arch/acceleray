# Acceleray Storyline (Masala Version)

## Opening Hook (30 seconds)
"Every morning before markets open, portfolio managers ask one question:  
**Are we seeing risk early enough, or reacting too late?**  
At one global asset manager, that question was buried behind slow model refreshes, long simulation queues, and overnight scoring windows that kept slipping."

## The Tension (The Problem)
- Data was growing faster than decision cycles.
- Spark pipelines were strong for ingestion and prep, but Python-heavy model iteration and simulation loops were too slow and rigid.
- Teams had two bad options:
  - Ship stale insights, or
  - Burn extra time and cloud cost to catch up.

## The Turning Point (Why Ray)
"They did not need a new platform. They needed a better execution layer for Python ML."

- Keep Spark + Delta for trusted data engineering.
- Add Ray for distributed tuning, training, and simulation.
- Keep MLflow + Databricks governance for traceability and control.

## What We Actually Implemented
1. **Portfolio model calibration at scale**  
   - Faster parameter search and threshold tuning using Ray Tune.
2. **Investment signal prioritization**  
   - Distributed scoring so analysts focus on highest-value signals first.
3. **Risk stress testing**  
   - Parallel scenario execution with Ray tasks.
4. **Portfolio-wide scoring pipeline**  
   - High-throughput batch inference for daily/overnight refresh.

## The Business Impact Story (Use Your Numbers)
"The technical win was speed.  
The business win was confidence before the bell."

- Model iteration cycle: **[X hours -> Y minutes]**
- Scenario processing throughput: **[Xx faster]**
- Portfolio scoring SLA adherence: **[before vs after]**
- Operational consistency: standardized runtime + MLflow lineage across teams

## Why Spark Alone Was Not Enough (Without Bashing Spark)
- Spark is excellent for ETL, SQL, and table-scale transformations.
- This customer's bottleneck was **dynamic Python ML execution**, not data ingestion.
- Ray solved orchestration of:
  - many concurrent training/tuning tasks,
  - simulation fan-out/fan-in patterns,
  - compute-heavy batch scoring actors.

## 60-Second Talk Track (Use This Verbatim)
"This is not a Spark replacement story.  
This is a **Spark + Ray** story.
Spark remains the system of record for data prep and governance.  
Ray becomes the acceleration layer for Python model iteration and simulation.  
The result: faster portfolio insight cycles, stronger SLA performance, and cleaner operational lineage for regulated environments."

## Slide Punchlines (Masala-Friendly)
- "Before market open, every minute is alpha."
- "From overnight backlog to decision-ready by bell time."
- "Spark for trusted data. Ray for fast decisions."
- "Scale without re-platforming."
- "Speed is good. Governed speed is better."

## Proposed Session Narrative Arc (45 min)
- **Act 1 - Pressure (8 min):** Why portfolio/risk teams hit a wall
- **Act 2 - Shift (12 min):** Why Ray, why now, why with Spark
- **Act 3 - Build (16 min):** Calibration, signal ranking, stress sim, scoring
- **Act 4 - Outcome (7 min):** Operational and business impact
- **Q&A (2 min)**

## Confidentiality-Safe Wording
- Use: "a large global asset manager"
- Use: "an anonymized financial services customer"
- Avoid: legal name, fund names, desk names, internal dataset names

