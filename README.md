---
title: LifeOps - Chaotic Human Life Simulation
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
app_port: 7860
tags:
  - openenv
  - rl
  - personalized-tasks
---

# 🧬 LifeOps: Training LLM Agents for Real-Life Trade-Offs

> "Most benchmarks ask if the answer is correct. Life asks if the choice is sustainable."

LifeOps is an OpenEnv environment where an LLM must balance career, family, health, stress, and budget over time. It is designed for RL fine-tuning with GRPO/TRL + Unsloth.

## Submission Links (Judge Quick Access)

- Hugging Face Space (repo): [CodeArtisan09/LifeOps](https://huggingface.co/spaces/CodeArtisan09/LifeOps)
- Live environment URL: [https://codeartisan09-lifeops.hf.space](https://codeartisan09-lifeops.hf.space)
- API health check: [https://codeartisan09-lifeops.hf.space/health](https://codeartisan09-lifeops.hf.space/health)
- Gradio demo UI: [https://codeartisan09-lifeops.hf.space/ui](https://codeartisan09-lifeops.hf.space/ui)
- Training notebook (Colab-ready): `notebooks/train_lifeops.ipynb`
- Training script: `scripts/train.py`
- Technical writeup (mini-blog content): `docs/BLOG.md`

## Why this problem matters

Most RL-for-LLM setups optimize a single objective. Real life is multi-objective and stateful:

- career vs family
- short-term gain vs burnout
- money constraints vs social obligations
- delayed consequences from earlier choices

LifeOps trains policies to make balanced decisions under these constraints.

## How the environment works

- Built on `openenv-core` (`openenv-core[core]>=0.2.2`) via OpenEnv interfaces.
- 24-step episodic lifecycle (1 step = 1 hour) with persistent world state.
- Scenario generation + action parsing + NPC updates + autonomous world dynamics.
- Multi-axis reward decomposition in `server/rewards.py`:
  - career, family, friendship, budget, health
  - stress penalties, efficiency, communication bonus
- OpenEnv-compatible API endpoints: `/health`, `/reset`, `/step`.

Core files:

- `server/lifeops_environment.py`
- `server/rewards.py`
- `server/parser.py`
- `server/scenario_generator.py`
- `server/npc_engine.py`

## Training and reproducibility

- RL stack: TRL GRPO + Unsloth (`unsloth/Qwen2.5-3B-Instruct`).
- End-to-end notebook pipeline in `notebooks/train_lifeops.ipynb` includes:
  - dependency install
  - Space/local environment wiring
  - normalized reward heads
  - baseline computation
  - GRPO training
  - reward plotting
  - post-train sanity checks
  - JSON export

## Evidence of training

Training/evaluation plots currently tracked in repo:

- `data/plots/performance.png`
- `data/plots/real_model_performance.png`

![Training Curves](data/plots/performance.png)
![Post-train Performance](data/plots/real_model_performance.png)

The notebook also saves reward-curve outputs under `grpo_out/` when run end-to-end.

## Quick run

### Local

```bash
python -m server.app
```

Then verify:

```bash
curl http://127.0.0.1:7860/health
```

### Train (notebook)

Open `notebooks/train_lifeops.ipynb` and run all cells.  
To force remote Space usage, set:

```python
import os
os.environ["LIFEOPS_ENV_URL"] = "https://codeartisan09-lifeops.hf.space"
```

## Push flow

- Push GitHub: `git push origin main`
- Push HF Space (snapshot replace): `powershell -ExecutionPolicy Bypass -File scripts/push_hf_space.ps1 main`

Use the snapshot script when HF rejects normal pushes due to historical blocked binaries.

## Non-negotiable checklist mapping

- OpenEnv-based environment: yes (`openenv-core`, OpenEnv interfaces in server)
- Working RL training pipeline (Unsloth/TRL): yes (`notebooks/train_lifeops.ipynb`, `scripts/train.py`)
- Training evidence (plots): included in `data/plots/`
- Public HF Space deployment: yes (links above)
- README motivation + environment mechanics + results: yes (this file)
- README includes HF Space + references: yes
- No large video files committed: yes (link external media instead)

## Optional external media

If you publish a Hugging Face community blog post or YouTube demo, add links here so judges can access them immediately.
