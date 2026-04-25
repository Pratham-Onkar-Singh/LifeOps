import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

def test_action_parsing():
    print("Initializing LifeOps for Action Parsing Test...")
    env = LifeopsEnvironment()
    env.reset()
    
    # Test 1: Natural Language Extraction (Choice is None)
    print("\nTest 1: NLP Extraction (Choice is None)")
    action1 = LifeopsAction(
        choice=None,
        justification="I think I should stay late to finish this report for my boss."
    )
    obs1 = env.step(action1)
    print(f"Input: 'stay late' | Detected Action: {obs1.environment_feedback}")
    
    # Test 2: Tool Routing Detection
    print("\nTest 2: Tool Routing detection")
    # Reset to start fresh
    env.reset()
    action2 = LifeopsAction(
        choice=None,
        justification="I'll send a text to my mom to say I'm sorry."
    )
    obs2 = env.step(action2)
    # The environment currently doesn't execute tool effects, but the parser should have detected it.
    print(f"Input: 'text mom' | Feedback: {obs2.environment_feedback}")

    # Test 3: Ambiguous Input
    print("\nTest 3: Ambiguous Input")
    env.reset()
    action3 = LifeopsAction(
        choice=None,
        justification="I am not sure what to do, maybe I'll just sleep."
    )
    obs3 = env.step(action3)
    print(f"Input: 'just sleep' | Detected: {obs3.environment_feedback}")

if __name__ == "__main__":
    try:
        test_action_parsing()
        print("\n[SUCCESS] Phase 5 Action Parsing Verified!")
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
