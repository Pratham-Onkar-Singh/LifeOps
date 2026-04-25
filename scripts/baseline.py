import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from server.agents import RandomAgent, GreedyAgent, RuleBasedAgent, PrioritizedPlannerAgent

def evaluate_agent(agent_class, name):
    print(f"\n--- Evaluating Agent: {name} ---")
    env = LifeopsEnvironment()
    obs = env.reset()
    agent = agent_class()
    
    total_reward = 0
    steps = 0
    
    print(f"Initial Scenario: {obs.active_conflict}")
    
    while not obs.done and steps < 10: # Test for 10 hours
        action = agent.act(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        print(f"Step {steps} | Action: {action.choice.value if action.choice else 'NLP'} | Reward: {obs.reward:.2f} | Stress: {obs.metrics['stress']}")

    print(f"Final Score for {name}: {total_reward:.2f} over {steps} steps.")

if __name__ == "__main__":
    try:
        agents = [
            (RandomAgent, "Random"),
            (GreedyAgent, "Greedy"),
            (RuleBasedAgent, "Rule-Based"),
            (PrioritizedPlannerAgent, "Smart Planner")
        ]
        
        for agent_class, name in agents:
            evaluate_agent(agent_class, name)
            
        print("\n[SUCCESS] Phase 8 Baselines Verified!")
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
