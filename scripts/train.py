# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LifeOps Training Pipeline (Phase 9).
Uses GRPO (Group Relative Policy Optimization) with Unsloth for efficient RL.
"""

import torch
import re
import inspect
from typing import List, Dict, Optional
from datetime import datetime

import sys
import os
import torch

# --- CRITICAL PYDANTIC PATCH (Run before TRL imports) ---
try:
    from pydantic import BaseModel, ConfigDict
    # This forces all models to allow complex types like torch.Tensor
    BaseModel.model_config = ConfigDict(arbitrary_types_allowed=True)
except Exception:
    pass
# -------------------------------------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TRL & Unsloth Imports
try:
    # Critical Fix for Pydantic V2 + TRL torch.Tensor schema error
    from pydantic import ConfigDict
    from trl import GRPOTrainer, GRPOConfig
    from unsloth import FastLanguageModel, PatchFastRL
    try:
        PatchFastRL()  # Patch for GRPO (older Unsloth versions)
    except OSError:
        # Newer Unsloth versions may already be patched in Colab/Python 3.12.
        pass
except Exception as e:
    print("\n" + "="*50)
    print("💡 LIFE-OPS TIP: Handling Library Import...")
    print("="*50 + "\n")
    
    class GRPOTrainer: 
        def __init__(self, **kwargs): pass
        def train(self): pass
    class GRPOConfig: 
        # Add ConfigDict to mock for compatibility
        model_config = {"arbitrary_types_allowed": True}
        def __init__(self, **kwargs): pass

# Local LifeOps Engine
try:
    from server.lifeops_environment import LifeopsEnvironment
    from models import LifeopsAction, LifeActionChoice
    from scripts.dataset_builder import LifeopsDatasetBuilder
except ImportError:
    sys.path.append(os.getcwd())
    from server.lifeops_environment import LifeopsEnvironment
    from models import LifeopsAction, LifeActionChoice
    from scripts.dataset_builder import LifeopsDatasetBuilder



# --- Reward Functions ---

def _extract_action_name(text: str) -> Optional[str]:
    """Extract Action value and normalize spaces/hyphens to enum style."""
    m = re.search(r"^\s*Action\s*:\s*([a-zA-Z0-9_ -]+)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    return re.sub(r"[\s-]+", "_", m.group(1).strip().lower())


def _coerce_prompt_text(prompt: object) -> Optional[str]:
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    # Common HF datasets use conversational message lists.
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        parts: List[str] = []
        for msg in prompt:
            role = str(msg.get("role", "")).strip()
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}:\n{content}".strip())
            else:
                parts.append(f"{role}:\n{str(content)}".strip())
        return "\n\n".join([p for p in parts if p])
    return str(prompt)


def _parse_allowed_actions(prompt: Optional[str]) -> Optional[set]:
    if not prompt:
        return None
    prompt = _coerce_prompt_text(prompt)
    if not prompt:
        return None
    m = re.search(r"Allowed Actions:\s*([^\n]+)", prompt, re.IGNORECASE)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return set(parts) if parts else None


def _format_reward(text: str) -> float:
    """Stronger format shaping in roughly [-3, +2]."""
    score = 0.0
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]

    if len(lines) == 2:
        score += 0.6
    else:
        score -= 1.2

    if len(lines) >= 1 and re.match(r"^\s*Action\s*:\s*\S", lines[0], re.IGNORECASE):
        score += 0.4
    else:
        score -= 0.8

    if len(lines) >= 2 and re.match(r"^\s*Justification\s*:\s*\S", lines[1], re.IGNORECASE):
        score += 0.4
    else:
        score -= 0.8

    if len(lines) >= 2:
        just_line = lines[1]
        m = re.match(r"^\s*Justification\s*:\s*(.+)\s*$", just_line, re.IGNORECASE)
        if m:
            body = m.group(1).strip()
            if "\n" in body:
                score -= 1.0
            wc = len(body.split())
            if wc > 28:
                score -= min(2.0, 0.08 * (wc - 28))
            if len(body) > 220:
                score -= min(2.0, 0.01 * (len(body) - 220))

    if re.search(r"\b(however|therefore|moreover|additionally)\b", text, re.IGNORECASE):
        score -= 0.35

    return float(max(-3.0, min(2.0, score)))


def lifeops_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Primary Reward: Uses the LifeOps environment to evaluate outcomes.
    """
    rewards = []
    env = LifeopsEnvironment()

    if prompts is None:
        prompts = kwargs.get("prompts")
    
    for idx, completion in enumerate(completions):
        prompt = None
        if prompts is not None and idx < len(prompts):
            prompt = prompts[idx]

        allowed = _parse_allowed_actions(prompt)
        action_str = _extract_action_name(completion)
        if not action_str:
            rewards.append(-2.0)
            continue

        if allowed is not None and action_str not in allowed:
            rewards.append(-2.0)
            continue

        try:
            choice = LifeActionChoice(action_str)
            action = LifeopsAction(choice=choice, justification="RL Training")
            env.reset()
            obs = env.step(action)
            rewards.append(obs.reward)
        except ValueError:
            rewards.append(-2.0)  # Penalty for invalid action names
            
    return rewards


def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward for strict 2-line formatting + brevity."""
    return [_format_reward(c) for c in completions]


def _build_grpo_config(**kwargs):
    """Construct GRPOConfig across TRL versions (some drop max_prompt_length)."""
    sig = inspect.signature(GRPOConfig.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print("Note: dropping unsupported GRPOConfig args:", dropped)
    return GRPOConfig(**filtered)


# --- Training Loop ---

def run_training(
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    num_samples: int = 100,
    output_dir: str = "lifeops_llama3_trained",
    push_to_hub: bool = True,
    hf_repo_id: str = "CodeArtisan09/lifeops-agent" # Change this!
):
    print(f"🚀 Initializing LifeOps RL Pipeline for {model_name}...")
    
    # 1. Prepare Data
    dataset = LifeopsDatasetBuilder.generate_rl_dataset(num_samples)
    
    # 2. Configure GRPO
    training_args = _build_grpo_config(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        max_prompt_length=256,
        max_completion_length=128,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.08,
        mask_truncated_completions=True,
        reward_weights=[1.0, 0.45],
        num_generations=8,
        push_to_hub=push_to_hub,
        hub_model_id=hf_repo_id,
        report_to="tensorboard", # Or "wandb" if you have an account
    )

    print("\nReward functions connected:")
    print("- lifeops_reward_func (Environment Logic)")
    print("- format_reward_func (Instruction Following)")
    
    print("\n[INFO] To execute training:")
    print("1. Upload this repo to Google Colab.")
    print("2. Run: !pip install unsloth trl")
    print("3. Execute this script with a GPU runtime.")
    
    # trainer = GRPOTrainer(
    #     model=model,
    #     reward_funcs=[lifeops_reward_func, format_reward_func],
    #     args=training_args,
    #     train_dataset=dataset,
    # )
    # trainer.train()

if __name__ == "__main__":
    run_training()
