# Project Closure – Fraud Detection MLOps (v2)

## Overview
This project implements an end-to-end MLOps pipeline for fraud detection,
from feature engineering to model deployment via API.

## Final State
- Single production-ready model (v2)
- Versioned feature pipeline
- Trained scaler and model persisted as artifacts
- MLflow used for experiment tracking and final run registration
- FastAPI endpoint exposed for real-time inference

## Key Decisions
- Unified feature pipeline for training and inference
- Target variable (`fraud`) handled only during training
- Stateless API design
- Explicit threshold definition
- Artifact immutability

## Deliverables
- `model.pkl`
- `scaler.pkl`
- `model_info.yaml`
- FastAPI `/predict` endpoint
- MLflow final run

## Status
✅ Project completed and ready for containerization and deployment.
