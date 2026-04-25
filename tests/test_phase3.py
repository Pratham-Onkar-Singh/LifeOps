import sys
import os
from uuid import uuid4

# Add the root directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

def test_full_lifecycle():
    print("Initializing Phase 3 Production Engine...")
    env = LifeopsEnvironment()
    
    print("\nTesting 24-Hour Cycle...")
    obs = env.reset()
    print(f"Start Time: {obs.current_time}")
    
    for i in range(5):
        choice = obs.available_choices[0]
        action = LifeopsAction(choice=choice, justification="Testing Phase 3 lifecycle.")
        obs = env.step(action)
        print(f"Step {i+1} | Time: {obs.current_time} | Stress: {obs.metrics['stress']} | Reward: {obs.reward}")
        
    print("\n[SUCCESS] Phase 3 Core Engine Verified!")

if __name__ == "__main__":
    try:
        test_full_lifecycle()
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
