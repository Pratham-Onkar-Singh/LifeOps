# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Production-grade Domain Models for LifeOps.

This module defines the rich entity-relationship model for the simulation,
enabling complex multi-step reasoning and long-horizon planning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import Field, BaseModel
from openenv.core.env_server.types import Action, Observation


# --- Enums for Type Safety ---

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LifeActionChoice(str, Enum):
    # Career Actions
    STAY_LATE_WORK = "stay_late_work"
    WORK_HARD = "work_hard"
    DELEGATE_WORK = "delegate_work"
    SKIP_WORK = "skip_work"
    
    # Social/Family Actions
    GO_TO_FAMILY_EVENT = "go_to_family_event"
    ASK_FOR_UNDERSTANDING = "ask_for_understanding"
    CANCEL_PLANS = "cancel_plans"
    
    # Personal/Health
    REST = "rest"
    SPEND_MONEY = "spend_money"
    EXERCISE = "exercise"
    
    # Generic
    DO_NOTHING = "do_nothing"


# --- Core Domain Objects ---

class Person(BaseModel):
    """Represents an NPC in the LifeOps world."""
    name: str
    relationship_type: str  # e.g., "Boss", "Mom", "Partner", "Client"
    patience: float = Field(100.0, ge=0.0, le=100.0)
    trust: float = Field(50.0, ge=0.0, le=100.0)
    mood: str = "Neutral"
    memory: List[str] = Field(default_factory=list)
    recent_interaction: Optional[LifeActionChoice] = None


class Task(BaseModel):
    """A unit of work or responsibility."""
    id: str
    title: str
    description: str
    priority: Priority
    deadline: datetime
    impact_career: float
    impact_stress: float
    is_completed: bool = False


class Event(BaseModel):
    """A scheduled moment in time (e.g., Birthday, Dinner, Meeting)."""
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    importance: float
    is_mandatory: bool
    participants: List[str]  # Names of Persons


class Message(BaseModel):
    """Communications received by the agent."""
    sender: str
    content: str
    timestamp: datetime
    is_read: bool = False
    requires_action: bool = True


class Budget(BaseModel):
    """Financial tracking."""
    balance: float
    monthly_limit: float
    savings_goal: float


class Calendar(BaseModel):
    """The agent's schedule."""
    events: List[Event]
    current_date: datetime


class TravelState(BaseModel):
    """Current location and transit info."""
    location: str
    destination: Optional[str] = None
    eta: Optional[datetime] = None
    is_in_transit: bool = False


class RewardBreakdown(BaseModel):
    """Detailed multi-axis reward decomposition."""
    career_score: float = 0.0
    family_score: float = 0.0
    friendship_score: float = 0.0
    budget_score: float = 0.0
    health_score: float = 0.0
    stress_penalty: float = 0.0
    efficiency_score: float = 0.0
    communication_score: float = 0.0
    total: float = 0.0
    explanation: str = ""


# --- Simulation State Objects ---

class AgentState(BaseModel):
    """The internal state of the agent being trained."""
    stress: float = Field(0.0, ge=0.0, le=100.0)
    energy: float = Field(100.0, ge=0.0, le=100.0)
    health: float = Field(100.0, ge=0.0, le=100.0)
    career_progress: float = 0.0
    family_affinity: float = 0.0


class EpisodeState(BaseModel):
    """The full 'World State' for a LifeOps episode."""
    agent: AgentState
    budget: Budget
    calendar: Calendar
    inbox: List[Message]
    tasks: List[Task]
    npcs: Dict[str, Person]
    travel: TravelState
    step_count: int = 0
    is_done: bool = False


# --- OpenEnv API Models ---

class LifeopsAction(Action):
    """Action for the LifeOps environment."""
    choice: Optional[LifeActionChoice] = Field(None, description="The high-level action category")
    justification: str = Field(..., description="The rationale or reasoning behind the action")
    message_reply: Optional[str] = Field(None, description="The text to send if the action involves messaging an NPC")


class LifeopsObservation(Observation):
    """The data the model sees at each step."""
    current_time: str
    metrics: Dict[str, float]  # Stress, Career, Family, Budget, etc.
    active_conflict: str
    inbox_preview: List[str]
    calendar_today: List[str]
    available_choices: List[LifeActionChoice]
    environment_feedback: str
    reward_metadata: Optional[Dict[str, Any]] = None


class Scenario(BaseModel):
    """Template for generating conflicts."""
    id: str
    title: str
    initial_description: str
    trigger_messages: List[Message]
    trigger_tasks: List[Task]
    trigger_events: List[Event]
    valid_choices: List[LifeActionChoice]
