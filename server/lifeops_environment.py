# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LifeOps Environment Engine (OpenEnv Core).
Production-grade implementation of Phase 3.
"""

import json
import os
import random
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, Any, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        LifeopsAction, LifeopsObservation, LifeActionChoice,
        EpisodeState, AgentState, Budget, Calendar, TravelState,
        Person, Task, Event, Message, RewardBreakdown, Priority
    )
    from .rewards import LifeopsRewardEngine
    from .scenario_generator import LifeopsScenarioGenerator
    from .parser import LifeopsActionParser
    from .npc_engine import LifeopsNPCEngine
except ImportError:
    from models import (
        LifeopsAction, LifeopsObservation, LifeActionChoice,
        EpisodeState, AgentState, Budget, Calendar, TravelState,
        Person, Task, Event, Message, RewardBreakdown, Priority
    )
    from server.rewards import LifeopsRewardEngine
    from server.scenario_generator import LifeopsScenarioGenerator
    from server.parser import LifeopsActionParser
    from server.npc_engine import LifeopsNPCEngine


class LifeopsEnvironment(Environment):
    """
    LifeOps: Phase 3 Core Engine.
    Implements full episode lifecycle, time advancement, and world dynamics.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 24  # A full day cycle (1 step = 1 hour)

    def __init__(self):
        """Initialize the LifeOps engine."""
        self.scenarios_raw = self._load_scenarios_raw()
        self._reset_state()

    def _load_scenarios_raw(self) -> List[Dict]:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scenarios.json")
        try:
            with open(data_path, "r") as f:
                return json.load(f)
        except Exception:
            return [{"id": "default", "description": "Routine day.", "choices": ["do_nothing"]}]

    def _reset_state(self):
        """Initialize world state with full lifecycle parameters."""
        # 1. Episode Lifecycle Init
        self.start_time = datetime(2026, 4, 25, 9, 0) # Start at 9 AM
        self.current_time = self.start_time
        
        # 2. Use Procedural Generator (Phase 4)
        scenario = LifeopsScenarioGenerator.generate(self.start_time)
        self._active_description = scenario.initial_description
        self._available_choices = scenario.valid_choices
        
        # 3. Build rich World Entities
        self.world_state = EpisodeState(
            agent=AgentState(stress=10.0, energy=100.0, health=100.0, career_progress=50.0, family_affinity=50.0),
            budget=Budget(balance=1200.0, monthly_limit=3000.0, savings_goal=5000.0),
            calendar=Calendar(events=scenario.trigger_events, current_date=self.start_time),
            inbox=scenario.trigger_messages,
            tasks=scenario.trigger_tasks,
            npcs={
                "Boss": Person(name="Mr. Henderson", relationship_type="Boss", patience=80.0),
                "Mom": Person(name="Mom", relationship_type="Mom", patience=100.0)
            },
            travel=TravelState(location="Home"),
            step_count=0,
            is_done=False
        )
        self._episode_id = str(uuid4())

    def reset(self) -> LifeopsObservation:
        """Reset the lifecycle."""
        self._reset_state()
        return self._make_observation("You wake up. It's a busy day ahead.")

    def step(self, action: LifeopsAction) -> LifeopsObservation:
        """Advance time and update world dynamics."""
        if self.world_state.is_done:
            return self._make_observation("Episode already finished.", reward=0.0)

        prev_state_dict = self.world_state.agent.dict()
        prev_state_dict["budget"] = self.world_state.budget.balance

        # 1. Advance Time (1 Step = 1 Hour)
        self.world_state.step_count += 1
        self.current_time += timedelta(hours=1)
        self.world_state.calendar.current_date = self.current_time

        # 2. Process Action using Phase 5 Parser
        choice, action_metadata = LifeopsActionParser.parse(action)
        
        if not choice or choice not in self._available_choices:
            return self._make_observation("Action unclear or invalid for this scenario.", reward=-2.0)
        
        message = self._apply_action_logic(choice)

        # 3. NPC Dynamics (Phase 7)
        npc_messages = LifeopsNPCEngine.update_npcs(self.world_state, choice, self.current_time)
        if npc_messages:
            message += " " + " ".join(npc_messages)

        # 4. Autonomous World Updates
        self._update_world_dynamics()

        # 4. Check Termination
        if self.world_state.step_count >= self.MAX_STEPS or self.world_state.agent.health <= 0:
            self.world_state.is_done = True
            message += " The day has ended."

        # 5. Reward Calculation (Phase 6)
        curr_state_dict = self.world_state.agent.dict()
        curr_state_dict["budget"] = self.world_state.budget.balance
        
        # Construct flat dictionaries for the Reward Engine
        reward_input_curr = {
            "career": self.world_state.agent.career_progress,
            "family": self.world_state.agent.family_affinity,
            "friendship": 50.0, # Placeholder for now
            "stress": self.world_state.agent.stress,
            "health": self.world_state.agent.health,
            "energy": self.world_state.agent.energy,
            "budget": self.world_state.budget.balance
        }
        reward_input_prev = {
            "career": prev_state_dict["career_progress"],
            "family": prev_state_dict["family_affinity"],
            "friendship": 50.0,
            "stress": prev_state_dict["stress"],
            "health": prev_state_dict.get("health", 100.0),
            "energy": prev_state_dict.get("energy", 100.0),
            "budget": prev_state_dict["budget"]
        }
        
        # action_metadata comes from Phase 5 Parser
        action_metadata["choice"] = choice.value if choice else "none"
        
        reward_breakdown = LifeopsRewardEngine.calculate_reward(
            reward_input_curr, 
            reward_input_prev, 
            action_metadata,
            self.world_state.step_count
        )
        
        return self._make_observation(
            message, 
            reward=reward_breakdown.total, 
            metadata=reward_breakdown.dict()
        )

    def _apply_action_logic(self, choice: Optional[LifeActionChoice]) -> str:
        if not choice: return "You spent an hour being indecisive."
        
        if choice == LifeActionChoice.STAY_LATE_WORK or choice == LifeActionChoice.WORK_HARD:
            self.world_state.agent.career_progress += 5
            self.world_state.agent.stress += 10
            self.world_state.agent.energy -= 15
            return "You put in an hour of hard work."
        
        if choice == LifeActionChoice.REST:
            self.world_state.agent.stress -= 5
            self.world_state.agent.energy += 10
            return "You took a break."
        
        if choice == LifeActionChoice.GO_TO_FAMILY_EVENT:
            self.world_state.agent.family_affinity += 5
            self.world_state.agent.stress -= 5
            return "You spent time with loved ones."

        return f"You performed: {choice.value}"

    def _update_world_dynamics(self):
        """Internal logic for state decay and autonomous NPC behavior."""
        # Natural energy drain
        self.world_state.agent.energy -= 2
        
        # Stress-Health link
        if self.world_state.agent.stress > 80:
            self.world_state.agent.health -= 5
        
        # NPC Patience decay for unhandled tasks
        for task in self.world_state.tasks:
            if not task.is_completed and self.current_time > task.deadline:
                self.world_state.npcs["Boss"].patience -= 10
                self.world_state.agent.career_progress -= 5

    def _parse_action(self, action: LifeopsAction) -> Optional[LifeActionChoice]:
        if action.choice: return action.choice
        text = (action.justification + " " + (action.message_reply or "")).lower()
        for c in LifeActionChoice:
            if c.value in text or c.value.replace("_", " ") in text: return c
        return None

    def _make_observation(self, feedback: str, reward: float = 0.0, metadata: Optional[Dict] = None) -> LifeopsObservation:
        """Partial Observability: Agent only sees current events and top messages."""
        return LifeopsObservation(
            current_time=self.current_time.strftime("%A, %I:%M %p"),
            metrics={
                "stress": round(self.world_state.agent.stress, 1),
                "energy": round(self.world_state.agent.energy, 1),
                "career": round(self.world_state.agent.career_progress, 1),
                "family": round(self.world_state.agent.family_affinity, 1),
                "budget": self.world_state.budget.balance
            },
            active_conflict=self._active_description,
            inbox_preview=[f"{m.sender}: {m.content[:30]}..." for m in self.world_state.inbox[-2:]],
            calendar_today=[f"{e.title} ({e.start_time.strftime('%H:%M')})" for e in self.world_state.calendar.events 
                            if e.start_time.date() == self.current_time.date()],
            available_choices=self._available_choices,
            environment_feedback=feedback,
            reward_metadata=metadata or {},
            done=self.world_state.is_done,
            reward=reward
        )

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self.world_state.step_count)

    def close(self):
        """Cleanup."""
        pass
