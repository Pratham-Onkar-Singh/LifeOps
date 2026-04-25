# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset Builder for LifeOps (Phase 9).
Generates synthetic life conflict prompts for RL training.
"""

import json
import os
from datetime import datetime
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from server.scenario_generator import LifeopsScenarioGenerator
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.getcwd())
    from server.scenario_generator import LifeopsScenarioGenerator

class LifeopsDatasetBuilder:
    """
    Generates training datasets by sampling from the Scenario Generator.
    """

    ACTION_CHOICES = [
        "stay_late_work",
        "work_hard",
        "delegate_work",
        "skip_work",
        "go_to_family_event",
        "ask_for_understanding",
        "cancel_plans",
        "rest",
        "spend_money",
        "exercise",
        "do_nothing",
    ]

    SYSTEM_PROMPT = (
        "You are an AI agent managing a chaotic human life. "
        "You must balance Career, Family, Health, and Budget. "
        "Pick exactly ONE action from the scenario's Allowed Actions list. "
        "Hard constraints (non-negotiable):\\n"
        "- Output EXACTLY 2 lines (no blank lines, no markdown, no bullet lists).\\n"
        "- Line 1 must start with: Action: <snake_case_action>\\n"
        "- Line 2 must start with: Justification: <ONE sentence, max 28 words, max 220 chars>\\n"
        "- The action MUST be copied verbatim from Allowed Actions (lowercase snake_case).\\n"
        "- Do NOT invent new action names like 'Prioritize ...' or 'Attend ...'.\\n"
        "- Do not output anything else before/after those two lines."
    )

    @staticmethod
    def generate_rl_dataset(num_samples: int = 100) -> List[Dict[str, str]]:
        """
        Generates a list of prompts for RL training (GRPO/PPO).
        Each sample contains a system prompt and a user prompt (scenario).
        """
        dataset = []
        base_time = datetime(2026, 4, 25, 9, 0)
        
        for _ in range(num_samples):
            scenario = LifeopsScenarioGenerator.generate(base_time)
            
            # Construct the prompt
            user_prompt = (
                f"Current Time: {base_time.strftime('%A, %I:%M %p')}\n"
                f"Situation: {scenario.initial_description}\n"
                f"Your Task: {scenario.trigger_tasks[0].title} (Deadline: {scenario.trigger_tasks[0].deadline.strftime('%I:%M %p')})\n"
                f"Your Calendar: {scenario.trigger_events[0].title} at {scenario.trigger_events[0].start_time.strftime('%I:%M %p')}\n"
                f"Allowed Actions: {', '.join([choice.value for choice in scenario.valid_choices])}\n"
                f"\nWhat is your next move?"
            )
            
            dataset.append({
                "prompt": [
                    {"role": "system", "content": LifeopsDatasetBuilder.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            })
            
        return dataset

    @staticmethod
    def save_to_json(dataset: List[Dict], filename: str = "data/train_dataset.json"):
        """Saves the generated dataset to a JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    builder = LifeopsDatasetBuilder()
    data = builder.generate_rl_dataset(50)
    builder.save_to_json(data)
