# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LifeOps NPC Dynamics Engine (Phase 7).
Handles NPC behavior, demands, and reactions.
"""

import random
from typing import Dict, List, Optional
from datetime import datetime
try:
    from .models import Person, LifeActionChoice, Message, EpisodeState
except ImportError:
    from models import Person, LifeActionChoice, Message, EpisodeState

class LifeopsNPCEngine:
    """
    Manages multi-agent interactions and NPC internal states.
    """

    @staticmethod
    def update_npcs(state: EpisodeState, choice: Optional[LifeActionChoice], current_time: datetime):
        """
        Updates NPC states based on the agent's action and time passage.
        """
        messages = []
        
        for name, npc in state.npcs.items():
            # 1. Reaction to Action
            LifeopsNPCEngine._apply_reaction(npc, choice)
            
            # 2. Mood update based on patience
            if npc.patience < 30:
                npc.mood = "Angry"
            elif npc.patience < 60:
                npc.mood = "Annoyed"
            elif npc.patience > 90:
                npc.mood = "Happy"
            else:
                npc.mood = "Neutral"

            # 3. Proactive Demands (Probabilistic)
            if random.random() < 0.1: # 10% chance per hour per NPC
                demand = LifeopsNPCEngine._generate_demand(npc, current_time)
                if demand:
                    state.inbox.append(demand)
                    messages.append(f"New message from {name}")

        return messages

    @staticmethod
    def _apply_reaction(npc: Person, choice: Optional[LifeActionChoice]):
        """NPCs react differently based on their relationship."""
        if not choice:
            npc.patience -= 2
            return

        rel = npc.relationship_type.lower()
        
        if rel == "boss":
            if choice == LifeActionChoice.STAY_LATE_WORK or choice == LifeActionChoice.WORK_HARD:
                npc.trust += 5
                npc.patience += 5
                npc.memory.append(f"Hard worker at {choice.value}")
            elif choice == LifeActionChoice.GO_TO_FAMILY_EVENT:
                npc.patience -= 10
                npc.trust -= 2
                npc.memory.append("Left early for family event")

        elif rel == "mom" or rel == "partner":
            if choice == LifeActionChoice.GO_TO_FAMILY_EVENT:
                npc.trust += 10
                npc.patience += 10
            elif choice == LifeActionChoice.STAY_LATE_WORK:
                npc.patience -= 15
                npc.memory.append("Chose work over us again")
            elif choice == LifeActionChoice.ASK_FOR_UNDERSTANDING:
                npc.patience += 2 # Softens the blow
                npc.trust -= 1

    @staticmethod
    def _generate_demand(npc: Person, current_time: datetime) -> Optional[Message]:
        """NPCs send contextual demands."""
        rel = npc.relationship_type.lower()
        
        demands = {
            "boss": [
                "Can you check those numbers again?",
                "Client is asking for an update.",
                "Are you coming into the office early tomorrow?"
            ],
            "mom": [
                "Don't forget to eat dinner!",
                "Call me when you're free.",
                "Your cousin is asking about you."
            ],
            "partner": [
                "Do we need anything from the grocery store?",
                "Thinking of you!",
                "Are we still on for tonight?"
            ]
        }
        
        if rel in demands:
            content = random.choice(demands[rel])
            # If annoyed, change tone
            if npc.mood == "Angry" or npc.mood == "Annoyed":
                content = content.replace("?", "!!!").replace("Can you", "You need to")
            
            return Message(
                sender=npc.name,
                content=content,
                timestamp=current_time,
                is_read=False,
                requires_action=True
            )
        return None
