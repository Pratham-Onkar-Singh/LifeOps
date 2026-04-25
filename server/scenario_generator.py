# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LifeOps Scenario Generation System (Phase 4).
Hybrid approach: Deterministic templates + Probabilistic generation.
"""

import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
try:
    from .models import Scenario, Task, Event, Message, Priority, LifeActionChoice
except ImportError:
    from models import Scenario, Task, Event, Message, Priority, LifeActionChoice

class LifeopsScenarioGenerator:
    """
    Procedurally generates life conflicts by combining stressors across domains.
    """
    
    WORK_STRESSORS = [
        {"title": "Critical Project Deadline", "priority": Priority.CRITICAL, "impact": 20.0},
        {"title": "Angry Client Escalation", "priority": Priority.HIGH, "impact": 15.0},
        {"title": "Quarterly Review Prep", "priority": Priority.MEDIUM, "impact": 10.0},
        {"title": "Surprise Team Meeting", "priority": Priority.LOW, "impact": 5.0}
    ]
    
    LIFE_CONFLICTS = [
        {"title": "Partner's Birthday Dinner", "importance": 0.9, "mandatory": False, "npc": "Partner"},
        {"title": "Emergency Vet Appointment", "importance": 0.95, "mandatory": True, "npc": "Vet"},
        {"title": "Best Friend's Wedding", "importance": 1.0, "mandatory": True, "npc": "Friend"},
        {"title": "Child's School Performance", "importance": 0.8, "mandatory": False, "npc": "Child"},
        {"title": "Burst Pipe at Home", "importance": 0.9, "mandatory": True, "npc": "Plumber"}
    ]

    @classmethod
    def generate(cls, base_time: datetime) -> Scenario:
        """
        Hybrid Generation: Selects a template and populates it probabilistically.
        """
        work = random.choice(cls.WORK_STRESSORS)
        life = random.choice(cls.LIFE_CONFLICTS)
        
        # 1. Create a dynamic ID and Title
        scenario_id = f"gen_{random.randint(1000, 9999)}"
        title = f"{work['title']} vs {life['title']}"
        
        # 2. Procedural Description (Deterministic Template)
        description = (
            f"A high-stakes situation has emerged: {work['title']}. "
            f"At the same time, you have the {life['title']} scheduled. "
            f"Missing the work task will hit your career by {work['impact']} points, "
            f"but the {life['npc']} is counting on you."
        )

        # 3. Probabilistic Triggers
        trigger_tasks = [
            Task(
                id="T_GEN", title=work['title'], description="Procedural task",
                priority=work['priority'], deadline=base_time + timedelta(hours=random.randint(2, 8)),
                impact_career=work['impact'], impact_stress=work['impact'] * 1.5
            )
        ]
        
        trigger_events = [
            Event(
                id="E_GEN", title=life['title'], start_time=base_time + timedelta(hours=random.randint(4, 10)),
                end_time=base_time + timedelta(hours=12), importance=life['importance'],
                is_mandatory=life['mandatory'], participants=[life['npc']]
            )
        ]

        # 4. LLM Enrichment Hook (Placeholder)
        # description = cls._enrich_with_llm(description)

        return Scenario(
            id=scenario_id,
            title=title,
            initial_description=description,
            trigger_messages=[
                Message(sender=life['npc'], content=f"Hey, are you coming to the {life['title']}?", timestamp=base_time)
            ],
            trigger_tasks=trigger_tasks,
            trigger_events=trigger_events,
            valid_choices=[
                LifeActionChoice.STAY_LATE_WORK,
                LifeActionChoice.GO_TO_FAMILY_EVENT,
                LifeActionChoice.DELEGATE_WORK,
                LifeActionChoice.ASK_FOR_UNDERSTANDING
            ]
        )

    @staticmethod
    def _enrich_with_llm(text: str) -> str:
        """
        Hook for LLM enrichment. In production, this would call GPT-4/Claude 
        to make the scenario description more varied and realistic.
        """
        # Example: return client.chat.completions.create(...)
        return text
