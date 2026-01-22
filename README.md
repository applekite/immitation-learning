# TSAC-DE
Trust-aware Soft Actor-Critic with Deep Ensembles and Control Barrier Functions for Safe RL in Autonomous Driving (CARLA).
# TSAC-DE

**Trust-aware Soft Actor-Critic with Deep Ensembles and Control Barrier Functions for Safe Reinforcement Learning in Autonomous Driving (CARLA).**

---

## Overview
TSAC-DE is a research project focused on developing a **Safe Reinforcement Learning (Safe RL)** algorithm for autonomous driving using:
- **Spatio-temporal Transformer Encoder** for processing sequential driving data.
- **Deep Ensemble Critics** to provide calibrated uncertainty estimates.
- **Trust Score** to guide safe decision-making.
- **Control Barrier Functions (CBFs)** as a safety layer to guarantee constraint satisfaction.

This repository will serve as the main workspace for implementation, experiments, and documentation.

---

## Project Structure (initial draft)
tsac-de/
├─ tsac_de/ # Core package (agents, models, safety, utils)
├─ configs/ # Experiment configs (YAML)
├─ carla/ # CARLA wrappers and driving scenarios
├─ scripts/ # Training/evaluation scripts
├─ tests/ # Unit tests
├─ docs/ # Documentation and research reports




