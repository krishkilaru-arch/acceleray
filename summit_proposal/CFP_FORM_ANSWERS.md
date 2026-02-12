# Databricks Summit CFP Answers

## Session Title
Portfolio Intelligence at Scale: Why Ray with Spark on Databricks for Financial Services

## Elevator Pitch (50 words)
This session shows how a financial services team scales portfolio intelligence and risk workflows with Ray + Spark on Databricks. Includes anonymized lessons from a large global financial services customer on faster model iteration, distributed scenario simulation, and governed production operations.

## Short Abstract (100 words)
This session follows a practical financial services workflow: ingest market and portfolio data, calibrate models, run stress scenarios, and refresh portfolio-level scores. We show how teams build this end-to-end flow on Databricks using Spark for trusted data pipelines and Ray for distributed Python ML workloads, including tuning, training, simulation, and batch scoring. We also share anonymized implementation learnings from a large global financial services customer, focused on runtime consistency, faster iteration, and governance. The result is a practical, reusable architecture for portfolio and risk teams under strict SLA and compliance requirements.

## Long Abstract (250 words)
A financial services team needs to improve turnaround for portfolio analytics and risk evaluation while maintaining strict reliability, governance, and auditability.

This session presents a practical architecture for those workloads on Databricks. Spark and Delta Lake handle ingestion, feature engineering, and data quality. Ray accelerates distributed Python tasks for hyperparameter tuning, model retraining, simulation-heavy risk workflows, and high-throughput batch scoring. MLflow tracks experiments and model lineage, while Databricks governance capabilities support production controls.

Using one coherent story arc, we demonstrate four execution decisions:
1. Portfolio model calibration with fast parameter optimization.
2. Distributed training for larger and more frequent model refresh cycles.
3. Stress and sensitivity risk evaluation using parallel scenario execution.
4. Portfolio-wide overnight scoring for downstream investment operations.

Attendees will leave with:
- A simple Spark-vs-Ray decision framework.
- Implementation patterns for distributed ML in regulated contexts.
- A phased rollout plan from notebook experimentation to repeatable production jobs.

The session also includes anonymized field learnings from a large global financial services customer. We focus on repeatable architecture and operating practices, not client-identifying details.

The goal is not another isolated demo. It is a relatable, end-to-end operating model that technical teams and business stakeholders can both understand and adopt.

## Confidentiality-Safe Wording
- Use: "a large global financial services customer"
- Use: "anonymized customer implementation"
- Avoid: legal entity name, business unit name, internal tool names, dataset identifiers

## Three Learning Objectives
1. Identify which components of financial ML workflows should run on Spark vs Ray.
2. Apply distributed tuning, simulation, and inference patterns on Databricks.
3. Design governance-ready model workflows with lineage and operational controls.

