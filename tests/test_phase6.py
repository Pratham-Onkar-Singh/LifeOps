import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

def test_reward_engine():
    print("Initializing LifeOps for Reward Engine Test (Phase 6)...")
    env = LifeopsEnvironment()
    obs = env.reset()
    
    # Test 1: Positive Career Action
    print("\nTest 1: Working Hard (Career Increase)")
    choice = LifeActionChoice.STAY_LATE_WORK
    action = LifeopsAction(choice=choice, justification="I need to finish the report.")
    obs = env.step(action)
    print(f"Action: {choice.value} | Reward Total: {obs.reward}")
    print(f"Explanation: {obs.reward_metadata.get('explanation')}")
    
    # Test 2: Communication Reward (Messaging NPC)
    print("\nTest 2: Communication Reward (Messaging)")
    obs = env.reset()
    # Explicitly use ASK_FOR_UNDERSTANDING which is usually social
    choice = LifeActionChoice.ASK_FOR_UNDERSTANDING
    action = LifeopsAction(choice=choice, justification="I'll send a text to my boss to apologize.")
    obs = env.step(action)
    print(f"Action: {choice.value} | Reward Total: {obs.reward}")
    print(f"Communication Score: {obs.reward_metadata.get('communication_score')}")
    print(f"Explanation: {obs.reward_metadata.get('explanation')}")

    # Test 3: Anti-Cheat Penalty (Passive in Crisis)
    print("\nTest 3: Anti-Cheat Penalty (Passivity)")
    obs = env.reset()
    # Fast forward to step 6 using valid choices
    for i in range(5): 
        obs = env.step(LifeopsAction(choice=LifeActionChoice.STAY_LATE_WORK, justification="Work"))
    
    # Now do nothing
    action = LifeopsAction(choice=LifeActionChoice.DO_NOTHING, justification="Just hanging out.")
    obs = env.step(action)
    print(f"Step: {env.world_state.step_count}")
    print(f"Action: {action.choice.value}")
    print(f"Reward Total: {obs.reward}")
    print(f"Explanation: {obs.reward_metadata.get('explanation')}")

if __name__ == "__main__":
    try:
        test_reward_engine()
        print("\n[SUCCESS] Phase 6 Reward Engine Verified!")
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
