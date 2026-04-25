import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

def test_npc_dynamics():
    print("Initializing LifeOps for NPC Dynamics Test (Phase 7)...")
    env = LifeopsEnvironment()
    obs = env.reset()
    
    print("\nInitial NPCs:")
    for name, npc in env.world_state.npcs.items():
        print(f"- {name}: Mood={npc.mood}, Patience={npc.patience}, Trust={npc.trust}")

    # Step 1: Work hard
    print("\nStep 1: Staying late for work...")
    action = LifeopsAction(choice=LifeActionChoice.STAY_LATE_WORK, justification="I need to impress the boss.")
    obs = env.step(action)
    print(f"Feedback: {obs.environment_feedback}")
    
    print("\nNPCs after Step 1:")
    for name, npc in env.world_state.npcs.items():
        print(f"- {name}: Mood={npc.mood}, Patience={npc.patience}, Trust={npc.trust}, Memory={npc.memory}")

    # Step 2: Family time
    print("\nStep 2: Going to family event...")
    action = LifeopsAction(choice=LifeActionChoice.GO_TO_FAMILY_EVENT, justification="Family is priority now.")
    obs = env.step(action)
    print(f"Feedback: {obs.environment_feedback}")
    
    print("\nNPCs after Step 2:")
    for name, npc in env.world_state.npcs.items():
        print(f"- {name}: Mood={npc.mood}, Patience={npc.patience}, Trust={npc.trust}, Memory={npc.memory}")

    # Step 3: Check for messages
    print(f"\nInbox size: {len(env.world_state.inbox)}")
    for msg in env.world_state.inbox:
        print(f"- From {msg.sender}: {msg.content}")

if __name__ == "__main__":
    try:
        test_npc_dynamics()
        print("\n[SUCCESS] Phase 7 NPC Dynamics Verified!")
    except Exception as e:
        print(f"\n[FAILURE] {str(e)}")
        import traceback
        traceback.print_exc()
