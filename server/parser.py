# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LifeOps Action Parsing Layer (Phase 5).
Handles the conversion of natural language model outputs into structured actions and tool calls.
"""

import re
from typing import Optional, Tuple, Dict
try:
    from .models import LifeActionChoice, LifeopsAction
except ImportError:
    from models import LifeActionChoice, LifeopsAction

class LifeopsActionParser:
    """
    Robust parser for agentic actions.
    Supports structured extraction and keyword-based fallback.
    """

    @staticmethod
    def parse(action: LifeopsAction) -> Tuple[Optional[LifeActionChoice], Dict]:
        """
        Parses the agent's input to determine the intended choice and any tool-related metadata.
        Returns (Choice, Metadata).
        """
        choice = action.choice
        metadata = {"is_tool_call": False, "tool_target": None}
        
        # 1. Source text for parsing
        source_text = (action.justification + " " + (action.message_reply or "")).lower()

        # 2. If choice is missing, attempt NLP Fallback Extraction
        if not choice:
            choice = LifeopsActionParser._fallback_extraction(source_text)

        # 3. Tool Routing Detection (Phase 5 Feature)
        # Detect if the agent is trying to 'message' someone or 'book' something
        if "text" in source_text or "message" in source_text or "tell" in source_text:
            metadata["is_tool_call"] = True
            metadata["tool"] = "messenger"
            # Try to extract the target (Boss, Mom, etc.)
            for target in ["boss", "mom", "partner", "friend", "vet"]:
                if target in source_text:
                    metadata["tool_target"] = target.capitalize()
                    break

        elif "book" in source_text or "cab" in source_text or "uber" in source_text:
            metadata["is_tool_call"] = True
            metadata["tool"] = "transport_app"

        return choice, metadata

    @staticmethod
    def _fallback_extraction(text: str) -> Optional[LifeActionChoice]:
        """Keyword-based fallback for messy LLM outputs."""
        # Mapping common natural phrases to Enums
        mappings = {
            "stay late": LifeActionChoice.STAY_LATE_WORK,
            "work late": LifeActionChoice.STAY_LATE_WORK,
            "go home": LifeActionChoice.GO_TO_FAMILY_EVENT,
            "family event": LifeActionChoice.GO_TO_FAMILY_EVENT,
            "delegate": LifeActionChoice.DELEGATE_WORK,
            "help": LifeActionChoice.DELEGATE_WORK,
            "sorry": LifeActionChoice.ASK_FOR_UNDERSTANDING,
            "understand": LifeActionChoice.ASK_FOR_UNDERSTANDING,
            "cancel": LifeActionChoice.CANCEL_PLANS,
            "rest": LifeActionChoice.REST,
            "sleep": LifeActionChoice.REST,
            "buy": LifeActionChoice.SPEND_MONEY,
            "spend": LifeActionChoice.SPEND_MONEY,
            "nothing": LifeActionChoice.DO_NOTHING
        }
        
        for keyword, choice in mappings.items():
            if keyword in text:
                return choice
        return None
