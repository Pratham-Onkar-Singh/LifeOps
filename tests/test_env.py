import sys
import os
from uuid import uuid4

# Add the root directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

def test_lifeops_env():
    print("Initializing LifeOps MVP Environment...")
    env = LifeopsEnvironment()
    
    print("\nTesting Reset...")
    obs = env.reset()
    print(f"Initial Feedback: {obs.environment_feedback}")
    print(f"Active Conflict: {obs.active_conflict}")
    print(f"Metrics: {obs.metrics}")
    
    print("\nTesting Step: GO_TO_FAMILY_EVENT...")
    # Note: Scenario might not have this choice, but let's try if it does or pick a valid one
    choice = LifeActionChoice.GO_TO_FAMILY_EVENT
    if choice not in obs.available_choices:
        choice = obs.available_choices[0]
        
    action = LifeopsAction(
        choice=choice,
        justification="Balanced approach."
    )
    obs = env.step(action)
    print(f"Feedback: {obs.environment_feedback}")
    print(f"Reward: {obs.reward}")
    print(f"New Metrics: {obs.metrics}")
    print(f"Done: {obs.done}")

def test_reward_function():
    print("\nTesting Training Reward Function...")
    from scripts.train import lifeops_reward_func, format_reward_func
    
    prompts = ["You have a conflict."] * 3
    completions = [
        "Action: go_to_family_event\nJustification: Mom is key.", 
        "Action: stay_late_work\nJustification: Career first.",   
        "I will just sleep.",                                     
    ]
    
    env_rewards = lifeops_reward_func(prompts, completions)
    fmt_rewards = format_reward_func(prompts, completions)
    
    for i in range(len(completions)):
        print(f"Completion {i}: Env Reward={env_rewards[i]}, Format Reward={fmt_rewards[i]}")

def test_all_scenarios():
    print("\nTesting All Scenarios...")
    env = LifeopsEnvironment()
    
    for i in range(5):
        obs = env.reset()
        print(f"\nIteration {i+1}:")
        print(f"Scenario: {obs.active_conflict}")
        
        choice = obs.available_choices[0]
        action = LifeopsAction(choice=choice, justification="Auto-test")
        obs = env.step(action)
        print(f"Choice: {choice.value}")
        print(f"Feedback: {obs.environment_feedback}")
        print(f"Reward: {obs.reward}")

if __name__ == "__main__":
    try:
        test_lifeops_env()
        test_reward_function()
        test_all_scenarios()
        print("\n[SUCCESS] Rich Domain Environment verified!")
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
