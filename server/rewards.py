from typing import Dict, Any, List
try:
    from .models import RewardBreakdown
except ImportError:
    from models import RewardBreakdown

class LifeopsRewardEngine:
    """
    Production-grade Reward Engine for LifeOps (Phase 6).
    Implements multi-axis evaluation, anti-cheating, and reward explainability.
    """
    
    @staticmethod
    def calculate_reward(
        current_state: Dict[str, Any], 
        prev_state: Dict[str, Any],
        action_metadata: Dict[str, Any],
        step_count: int
    ) -> RewardBreakdown:
        """
        Calculates a detailed decomposition of rewards.
        """
        # 1. Delta Calculation
        career_delta = current_state.get("career", 0) - prev_state.get("career", 0)
        family_delta = current_state.get("family", 0) - prev_state.get("family", 0)
        friendship_delta = current_state.get("friendship", 0) - prev_state.get("friendship", 0)
        budget_delta = current_state.get("budget", 0) - prev_state.get("budget", 0)
        health_delta = current_state.get("health", 0) - prev_state.get("health", 0)
        stress_delta = current_state.get("stress", 0) - prev_state.get("stress", 0)
        energy_delta = current_state.get("energy", 0) - prev_state.get("energy", 0)

        # 2. Score Computation
        breakdown = RewardBreakdown()
        explanations = []

        # Career: direct impact
        breakdown.career_score = career_delta * 1.5
        if career_delta > 0: explanations.append("Productive work advanced career goals.")
        elif career_delta < 0: explanations.append("Career progress stalled due to neglect.")

        # Family & Friendship
        breakdown.family_score = family_delta * 2.0  # High weight for personalized theme
        if family_delta > 0: explanations.append("Strengthened family bonds.")
        breakdown.friendship_score = friendship_delta * 1.0
        
        # Budget
        breakdown.budget_score = budget_delta * 0.01 # Small scale for money
        if budget_delta < 0: explanations.append(f"Spent ${abs(budget_delta)} on life requirements.")

        # Health & Stress
        breakdown.health_score = health_delta * 1.0
        breakdown.stress_penalty = -max(0, stress_delta) * 0.5
        if stress_delta > 10: explanations.append("Significant stress increase detected.")
        if current_state.get("stress", 0) > 80:
            breakdown.stress_penalty -= 5.0 # Extra penalty for burnout zone
            explanations.append("Burnout warning: chronic high stress.")

        # Efficiency: reward for finishing tasks vs energy spent
        # If energy delta is small but career/family is high = efficient
        breakdown.efficiency_score = (abs(career_delta) + abs(family_delta)) / (abs(energy_delta) + 1)
        
        # Communication: reward for tool usage (Messenger)
        if action_metadata.get("is_tool_call") and action_metadata.get("tool") == "messenger":
            breakdown.communication_score = 2.0
            explanations.append(f"Proactive communication with {action_metadata.get('tool_target', 'NPC')}.")

        # 3. Anti-Cheating Logic
        # Penalty for being indecisive or repeating 'do_nothing' in a crisis
        if action_metadata.get("choice") == "do_nothing" and step_count > 5:
            breakdown.total -= 10.0
            explanations.append("Anti-Cheat: Penalized for excessive passivity.")

        # 4. Final Aggregation
        breakdown.total = (
            breakdown.career_score + 
            breakdown.family_score + 
            breakdown.friendship_score + 
            breakdown.budget_score + 
            breakdown.health_score + 
            breakdown.stress_penalty + 
            breakdown.efficiency_score + 
            breakdown.communication_score
        )
        
        breakdown.explanation = " | ".join(explanations)
        return breakdown
