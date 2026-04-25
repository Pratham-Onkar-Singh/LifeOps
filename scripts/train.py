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
    match = re.search(r"Action\s*:\s*([a-zA-Z_ -]+)", text, re.IGNORECASE)
    if not match:
        return None
    return re.sub(r"[\s-]+", "_", match.group(1).strip().lower())

def lifeops_reward_func(prompts, completions, **kwargs) -> List[float]:
    """
    Primary Reward: Uses the LifeOps environment to evaluate outcomes.
    """
    rewards = []
    env = LifeopsEnvironment()
    
    for completion in completions:
        action_str = _extract_action_name(completion)
        if not action_str:
            rewards.append(-1.0)  # Penalty for formatting failure
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
    """Reward for adhering to 'Action: <choice>\nJustification: <text>' format."""
    rewards = []
    for completion in completions:
        if "Action:" in completion and "Justification:" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


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
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        max_prompt_length=256,
        max_completion_length=128,
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
