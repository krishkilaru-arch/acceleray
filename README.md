# Acceleray Databricks Summit Proposal

## Session Title
Portfolio Intelligence at Scale: Why Ray with Spark on Databricks for Financial Services

## Proposal Summary
A financial services team needs to refresh portfolio risk insights faster, run larger scenario simulations, and scale model retraining without breaking governance controls.

This proposal presents a practical Spark + Ray architecture on Databricks:
- Spark + Delta Lake for trusted ingestion and feature engineering
- Ray for distributed Python workloads (tuning, training, simulation, batch scoring)
- MLflow for experiment tracking and lineage
- Databricks governance and operational controls for production reliability

The core message is simple:
- Spark remains the foundation for data engineering.
- Ray accelerates Python-native model execution and simulation.
- Together, they improve decision speed without re-platforming.

## What We Implemented So Far (Customer-Aligned)
- Hyperparameter optimization pattern:
  - `notebooks/01_hyperparameter_tuning.py`
- Distributed training pattern:
  - `notebooks/02_distributed_training.py`
- Portfolio batch scoring pattern:
  - `notebooks/03_batch_inference.py`
- End-to-end pipeline pattern:
  - `notebooks/04_end_to_end_pipeline.py`
- Portfolio model calibration pattern:
  - `notebooks/05_portfolio_model_calibration.py`
- Investment signal prioritization pattern:
  - `notebooks/06_investment_signal_prioritization.py`
- Risk stress testing pattern:
  - `notebooks/07_risk_stress_testing.py`

## Why Ray, Why Not Spark-Only?
- Spark is excellent for ETL, SQL, and table-scale transformations.
- This use case bottleneck is dynamic Python ML execution and simulation fan-out/fan-in.
- Ray improves orchestration for:
  - many concurrent tuning/training tasks,
  - simulation-heavy workloads,
  - compute-intensive portfolio scoring.

This is not a Spark replacement story; it is a Spark + Ray acceleration story.

## Real-World Customer Impact (Anonymized)
From a large global financial services customer:
- Shorter model iteration cycles for portfolio and risk workflows
- Faster simulation throughput for scenario analysis
- Better operational consistency with standardized runtime setup
- Improved reproducibility with MLflow + Delta-based lineage

## Audience
- Portfolio analytics and risk modeling teams
- ML engineers and platform teams using Databricks
- Data engineering teams supporting model operations

## Session Flow (45 minutes)
- 8 min: Problem framing and architecture decision model (Spark vs Ray responsibilities)
- 28 min: Calibrate, Train, Simulate, Score walkthrough
- 7 min: Production readiness, governance, and operational patterns
- 2 min: Q&A

## Databricks Alignment
Aligned with Databricks Ray-on-Spark architecture guidance:
- https://community.databricks.com/t5/technical-blog/ray-on-spark-a-practical-architecture-and-setup-guide/ba-p/127511

## Contact
- Name: [Your Name]
- Title: [Your Title]
- Email: [Email]
- LinkedIn: [LinkedIn]
- GitHub: [GitHub]
