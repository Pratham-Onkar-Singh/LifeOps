# LifeOps: How We Taught an LLM to Balance Life, Not Just Answer Prompts

> "Most benchmarks ask: Is the answer correct?  
> Life asks: Was that choice worth it?"

---

## 0) The Moment the Idea Was Born

There is a class of intelligence we do not benchmark enough: **trade-off intelligence**.

You can build an LLM that solves coding tasks and still fail at this:

- Deadline at 10 PM
- Parent's health emergency at 8 PM
- Rent due tomorrow
- You are already mentally exhausted

A human does not solve this with one "correct output." A human negotiates values, time, money, relationships, and energy under uncertainty.

That is where **LifeOps** started.

We asked a simple but uncomfortable question:

**Can we train an AI agent to make "good life decisions" under pressure, not just syntactically correct ones?**

---

## 1) Problem Statement: Why Existing RL-for-LLM Setups Fall Short

Most RL environments for language models evaluate one of the following:

- Accuracy (math, code, QA)
- Rule-following (format constraints)
- Single-goal optimization (maximize one scalar)

Real-life decision making is none of these.

In reality, decisions are:

- **Multi-objective** (career vs family vs health vs budget)
- **Stateful** (yesterday's bad choice changes today's options)
- **Socially coupled** (NPC trust, patience, memory)
- **Time-sensitive** (same action at hour 8 and hour 23 has different impact)

So we built an environment where "being right" is less important than **being balanced over time**.

---

## 2) What LifeOps Is (In One Sentence)

**LifeOps is an OpenEnv-compatible, multi-axis life simulation where an LLM chooses actions plus justification, receives verifiable state transitions, and is trained with GRPO to optimize long-horizon human balance.**

---

## 3) Design Principles We Refused to Compromise On

### 3.1 Verifiable Consequences
Every decision must alter measurable metrics, not vibes.

### 3.2 Explainability at Action Time
Agent output includes:

- `Action: <enum>`
- `Justification: <free text>`

So we can inspect *why* a decision was made, not only *what* was chosen.

### 3.3 Reward as a Composition, Not a Guess
Reward is derived from explicit state deltas and constraints, then normalized for RL stability.

### 3.4 Production-Ready Training Loop
Not just local scripts. We needed:

- Colab-compatible notebook training
- Public API endpoint via Hugging Face Space
- End-to-end reproducibility for judges

---

## 4) Architecture Story: From Scenario to Reward Signal

At high level:

1. Agent receives observation (conflict + metrics + context).
2. Agent emits structured action + natural language justification.
3. Environment validates action and advances simulation.
4. Reward engine scores consequences across multiple dimensions.
5. GRPO updates policy based on relative group performance.

Core building blocks:

- `server/lifeops_environment.py`: world-state updates, conflict logic
- `server/rewards.py`: multi-axis scoring and aggregation
- `scripts/dataset_builder.py`: RL-ready scenario/prompt generation
- `scripts/rl_action_utils.py`: parsing, mapping, formatting reward, normalization
- `scripts/train.py`: portable training entrypoint with reward hooks
- `notebooks/train_lifeops.ipynb`: full train/eval/plot/save flow for Colab

---

## 5) The Hard Problems We Faced (and How We Solved Them)

This section is intentionally detailed because this is where the real engineering happened.

### Problem A: Reward Curves Went Flat After a Few Steps

**Symptom:** Training reward looked healthy early, then became almost constant.

**What looked suspicious:** People often assume this means "reward bug" or "optimizer bug."

**Root cause (deep):**
GRPO computes **within-group relative advantages**. If all completions in a group get near-identical rewards, reward standard deviation collapses toward zero, and advantages collapse too. Updates become tiny even though training still "runs."

In LifeOps this happened when model generations converged to nearly identical `Action:` lines and similar justifications, causing near-deterministic env returns.

**Fixes we shipped:**

- Improved parsing and anti-spill cleanup in `scripts/rl_action_utils.py`
- Strengthened justification extraction path to preserve signal
- Better sampling/diversity settings in training config
- Added clear diagnostic explanation directly in notebook docs so this behavior is interpretable, not mysterious

---

### Problem B: Rewards Were on Incompatible Scales

**Symptom:** Env reward and formatting reward had different ranges and could dominate each other unpredictably.

**Impact:** Unstable interpretation of "reward improvements."

**Fixes:**

- Added monotonic env normalization to [0, 1]: `normalize_lifeops_env_reward(...)`
- Added formatting normalization to [0, 1]: `normalize_format_reward_unit(...)`
- Kept ordering monotonic, preserving policy gradient usefulness
- Normalized `reward_weights` so combined trainer reward remains in a bounded interpretable range

Result: easier debugging, cleaner reward narratives, stronger comparability with baseline-style notebooks.

---

### Problem C: Model Output Formatting Drift

**Symptom:** Model started outputting extra chat-role artifacts and multiline spill, breaking strict parser assumptions.

**Fixes:**

- Hardened spill stripping and action phrase extraction
- Added strict two-line format reward pressure
- Added readable debug windows in notebook for completion inspection

---

### Problem D: Baseline Comparisons Were Missing

**Symptom:** Reward curve alone was hard to interpret without a stable reference line.

**Fix:**

- Added `scripts/lifeops_grpo_metrics.py` to compute a **uniform-action baseline**
- Notebook now overlays baseline on reward chart
- Results export now captures both training curve and baseline in JSON

This made the story judge-friendly: not just "it improved," but "it improved against a defined policy baseline."

---

### Problem E (Engineered by Design): Reward Hacking Risk

Even in controlled environments, models can exploit shallow reward channels.

**Preventive strategy we implemented and iterated on:**

- Action validity checks against allowed action sets
- Mapping penalties for non-compliant action strings
- Format reward as secondary (not sole) signal
- Environment-driven consequence as primary learning anchor

---

## 6) Training Stack and Why We Chose It

### Model & Optimization

- Base: `unsloth/Qwen2.5-3B-Instruct`
- Method: GRPO with LoRA fine-tuning
- Motivation: strong cost/performance balance for hackathon-scale experimentation

### Why GRPO Here?

Because our setting is **relative preference under constraints**, not binary correctness.
GRPO's group-relative objective is aligned with comparative life-choice quality, provided we preserve group reward variance.

### Reward Heads

- Head 1: Environment consequence reward (normalized)
- Head 2: Output format/compliance reward (normalized)
- Weighted combination for stable and interpretable learning

---

## 7) Evaluation Philosophy: "Does It Make Better Decisions?"

We avoid vanity metrics and focus on policy behavior evidence:

- Mean weighted reward on sampled scenarios
- Per-scenario score slices
- Output validity and spill rates
- Baseline-vs-trained reward gap

Notebook workflow now includes:

1. Train
2. Plot reward vs baseline
3. Run post-train sanity evaluation
4. Save reproducible artifact `grpo_out/training_results.json`

This mirrors the reference style judges expect while retaining LifeOps-specific rigor.

---

## 8) Productization Beyond a Notebook

We treated deployment as first-class:

- OpenEnv HTTP endpoints on HF Space root (`/health`, `/reset`, `/step`)
- Gradio UI at `/ui` for live interaction demos
- GitHub + HF push automation
- Dockerized environment reproducibility

So judges can:

- read code,
- run notebook,
- hit live API,
- inspect outputs,

without hidden local magic.

---

## 9) Why This Project Matters (Bigger Than a Hackathon)

Current AI assistants optimize micro-tasks.
Human life requires macro-balance.

LifeOps points toward a different training paradigm:

- value-aware instead of answer-only,
- long-horizon instead of one-shot,
- explainable trade-offs instead of opaque outputs.

An assistant that knows "how to write an email" is useful.
An assistant that knows "when not to optimize work at the cost of burnout and family trust" is transformative.

---

## 10) What We Would Build Next

If we had another month, we would ship:

1. **Long-horizon episodic memory** across multiple simulated days/weeks
2. **Counterfactual evaluator** ("What if we chose X instead of Y?")
3. **Adaptive user profiles** (risk tolerance, family obligations, financial constraints)
4. **Causal reward decomposition dashboard** in UI
5. **A/B trained policy tournament** against scripted and heuristic agents

---

## 11) Judge-Facing Highlights (Quick Scan)

- Novel domain: RL for real-life trade-off intelligence
- Technically grounded environment + verifiable reward mechanics
- Deep debugging journey: reward collapse, normalization, parser robustness, deployment resilience
- Full reproducible pipeline from training to evaluation to saved artifacts
- Public demo + API readiness
- Strong storytelling with real engineering underneath

---

## 12) Final Reflection

LifeOps began as an idea about chaotic evenings and impossible choices.
It became an engineering experiment in teaching machines something subtle:

**not how to be "correct," but how to be responsibly helpful when everything important conflicts at once.**

If AI is going to become a life companion rather than just a text generator, this is the direction it must learn to walk.

And this is only Day 1.

---

*Built for the OpenEnv Hackathon 2026 with equal parts curiosity, stubborn debugging, and belief that AI should understand human trade-offs.*
