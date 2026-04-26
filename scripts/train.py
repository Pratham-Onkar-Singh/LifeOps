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
    from scripts.rl_action_utils import (
        resolve_allowed_actions,
        strip_generative_spill,
        extract_raw_action_phrase,
        map_phrase_to_allowed_action,
        action_line_is_snake_enum,
        extract_justification_phrase,
        compute_format_reward,
        normalize_lifeops_env_reward,
        normalize_format_reward_unit,
    )
except ImportError:
    sys.path.append(os.getcwd())
    from server.lifeops_environment import LifeopsEnvironment
    from models import LifeopsAction, LifeActionChoice
    from scripts.dataset_builder import LifeopsDatasetBuilder
    from scripts.rl_action_utils import (
        resolve_allowed_actions,
        strip_generative_spill,
        extract_raw_action_phrase,
        map_phrase_to_allowed_action,
        action_line_is_snake_enum,
        extract_justification_phrase,
        compute_format_reward,
        normalize_lifeops_env_reward,
        normalize_format_reward_unit,
    )



# --- Reward Functions ---

def lifeops_reward_func(
    prompts,
    completions,
    allowed_actions_json=None,
    **kwargs,
) -> List[float]:
    """
    Primary Reward: Uses the LifeOps environment to evaluate outcomes.
    """
    rewards = []
    env = LifeopsEnvironment()

    if prompts is None:
        prompts = kwargs.get("prompts")
    if allowed_actions_json is None:
        allowed_actions_json = kwargs.get("allowed_actions_json")

    tok = kwargs.get("processing_class") or kwargs.get("tokenizer")

    for idx, completion in enumerate(completions):
        prompt = None
        if prompts is not None and idx < len(prompts):
            prompt = prompts[idx]

        allowed_blob = None
        if allowed_actions_json is not None and idx < len(allowed_actions_json):
            allowed_blob = allowed_actions_json[idx]

        allowed = resolve_allowed_actions(prompt=prompt, allowed_actions_json=allowed_blob)
        cleaned = strip_generative_spill(completion, tokenizer=tok)
        phrase = extract_raw_action_phrase(cleaned)
        mapped, map_pen = map_phrase_to_allowed_action(phrase or "", allowed)
        if not mapped:
            rewards.append(normalize_lifeops_env_reward(-2.0))
            continue

        try:
            choice = LifeActionChoice(mapped)
            justification = extract_justification_phrase(cleaned, tokenizer=tok) or " "
            action = LifeopsAction(choice=choice, justification=justification)
            env.reset()
            obs = env.step(action)
            raw = float(obs.reward) + float(map_pen)
            rewards.append(normalize_lifeops_env_reward(raw))
        except ValueError:
            rewards.append(normalize_lifeops_env_reward(-2.0))
            
    return rewards


def format_reward_func(prompts, completions, **kwargs) -> List[float]:
    """Reward for strict 2-line formatting + brevity (0–1)."""
    tok = kwargs.get("processing_class") or kwargs.get("tokenizer")
    return [
        normalize_format_reward_unit(compute_format_reward(c, tokenizer=tok))
        for c in completions
    ]


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
        reward_weights=[1.0 / 1.45, 0.45 / 1.45],
        remove_unused_columns=False,
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
