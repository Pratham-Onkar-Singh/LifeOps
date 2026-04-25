# LifeOps Architecture

## Overview
LifeOps is built on top of the **OpenEnv** framework, utilizing a client-server architecture to isolate environment execution from agent training.

## Component Breakdown

### 1. Environment Server (`server/`)
- **FastAPI Core:** Exposes the environment over HTTP/WebSocket.
- **LifeopsEnvironment:** Manages simulation state, scenario generation, and NPC dynamics.
- **Reward Engine:** Calculates multi-axis rewards (Career, Family, Stress, Budget) based on the current state transition.
- **NLP Parser:** Handles fallback parsing of natural language justifications into structured actions.

### 2. Domain Models (`models.py`)
- **Action:** Pydantic model for agent inputs (Enum choice + free-text justification).
- **Observation:** Rich state representation including metrics, active conflicts, and available choices.
- **State Snapshot:** Internal state logging for verification.

### 3. Training Pipeline (`scripts/`)
- **GRPO Strategy:** Uses Group Relative Policy Optimization to train models on the environment rewards.
- **Unsloth Integration:** Optimized for efficient training on consumer GPUs.

### 4. Interactive Demo (`app/`)
- **Gradio Frontend:** Provides a visual dashboard for human evaluation and real-time agent monitoring.

## Data Flow
1. **Agent** receives `Observation` (Conflict description + Metrics).
2. **Agent** generates `Action` (Choice + Justification).
3. **Environment** validates and parses `Action`.
4. **Environment** updates internal `State`.
5. **Reward Engine** computes reward breakdown.
6. **Environment** returns new `Observation` + `Reward`.
