# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline Agents for LifeOps (Phase 8).
Provides comparison points for RL training.
"""

import random
from typing import List, Optional
try:
    from .models import LifeopsObservation, LifeopsAction, LifeActionChoice, Priority
except ImportError:
    from models import LifeopsObservation, LifeopsAction, LifeActionChoice, Priority

class BaseBaselineAgent:
    """Base class for all baseline agents."""
    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        raise NotImplementedError


class RandomAgent(BaseBaselineAgent):
    """Picks a valid action entirely at random."""
    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        choice = random.choice(obs.available_choices)
        return LifeopsAction(choice=choice, justification="Randomly selected action.")


class GreedyAgent(BaseBaselineAgent):
    """Always picks the action that sounds most productive for career."""
    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        if LifeActionChoice.STAY_LATE_WORK in obs.available_choices:
            choice = LifeActionChoice.STAY_LATE_WORK
        elif LifeActionChoice.WORK_HARD in obs.available_choices:
            choice = LifeActionChoice.WORK_HARD
        else:
            choice = obs.available_choices[0]
        return LifeopsAction(choice=choice, justification="I am prioritizing immediate career gains.")


class RuleBasedAgent(BaseBaselineAgent):
    """Balances life based on hardcoded safety rules."""
    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        stress = obs.metrics.get("stress", 0)
        energy = obs.metrics.get("energy", 100)
        
        # Rule 1: Emergency Rest
        if stress > 80 or energy < 20:
            if LifeActionChoice.REST in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.REST, justification="My stress is too high, I need to rest.")
        
        # Rule 2: Prioritize Family if relations are low
        if obs.metrics.get("family", 50) < 40:
            if LifeActionChoice.GO_TO_FAMILY_EVENT in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.GO_TO_FAMILY_EVENT, justification="Family relations are suffering.")
            if LifeActionChoice.ASK_FOR_UNDERSTANDING in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.ASK_FOR_UNDERSTANDING, justification="Communicating with family to maintain trust.")

        # Rule 3: Default to work
        if LifeActionChoice.WORK_HARD in obs.available_choices:
            return LifeopsAction(choice=LifeActionChoice.WORK_HARD, justification="Routine work.")
        
        return LifeopsAction(choice=obs.available_choices[0], justification="Following basic heuristics.")


class PrioritizedPlannerAgent(BaseBaselineAgent):
    """
    Advanced Baseline: Looks at active conflicts and metrics to plan.
    Simulates 'smart' human behavior.
    """
    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        conflict = obs.active_conflict.lower()
        
        # 1. Handle Critical Deadlines
        if "critical" in conflict or "urgent" in conflict:
            if LifeActionChoice.STAY_LATE_WORK in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.STAY_LATE_WORK, justification="Addressing a critical deadline.")
            if LifeActionChoice.DELEGATE_WORK in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.DELEGATE_WORK, justification="Critical task, but I need to delegate to manage capacity.")

        # 2. Handle Major Family Events
        if "birthday" in conflict or "wedding" in conflict:
            if LifeActionChoice.GO_TO_FAMILY_EVENT in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.GO_TO_FAMILY_EVENT, justification="This is a milestone event, I must attend.")

        # 3. Resource Management (Energy/Budget)
        if obs.metrics.get("energy") < 40:
            if LifeActionChoice.REST in obs.available_choices:
                return LifeopsAction(choice=LifeActionChoice.REST, justification="Recovering energy before I burnout.")

        # 4. Default to balanced action
        return LifeopsAction(choice=obs.available_choices[0], justification="Selecting the first viable option based on current priority analysis.")
