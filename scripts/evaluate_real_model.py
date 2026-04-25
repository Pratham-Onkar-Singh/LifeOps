import os
import sys
import json
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice, LifeopsObservation
from server.agents import RandomAgent, GreedyAgent, RuleBasedAgent, PrioritizedPlannerAgent

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import InferenceClient

class RealLLMAgent:
    """An agent that uses a real model with Memory and Chain-of-Thought."""
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.history = []

    def act(self, obs: LifeopsObservation) -> LifeopsAction:
        if not HF_TOKEN or "your_token" in HF_TOKEN:
            return LifeopsAction(choice=obs.available_choices[0], justification="Fallback.")

        # Build a "Memory" of previous actions and rewards
        history_str = "\n".join(self.history[-3:]) # Last 3 steps
        
        prompt = (
            "SYSTEM: You are a high-performance Life Management Agent. Your goal is to maximize Career and Family scores while keeping Stress below 80.\n"
            f"RECENT HISTORY:\n{history_str}\n"
            f"CURRENT STATUS:\n- Time: {obs.current_time}\n- Metrics: {obs.metrics}\n- Conflict: {obs.active_conflict}\n"
            f"AVAILABLE CHOICES: {[c.value for c in obs.available_choices]}\n\n"
            "TASK: Think step-by-step about the long-term impact of your choice. Then respond with ONLY a JSON object: "
            "{\"choice\": \"...\", \"justification\": \"...\"}"
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(messages, max_tokens=300)
            text = response.choices[0].message.content
            
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                choice = LifeActionChoice(data['choice'].strip().lower())
                
                # Record this in history for the next step
                self.history.append(f"- At {obs.current_time}, you chose {choice.value}. Result: {obs.environment_feedback}")
                
                return LifeopsAction(choice=choice, justification=data['justification'])
        except Exception as e:
            print(f"⚠️ [RealLLMAgent] Exception: {e}")
        
        return LifeopsAction(choice=obs.available_choices[0], justification="Fallback.")

def run_evaluation(num_episodes=2, max_steps=15): # Longer horizon to show burnout
    env = LifeopsEnvironment()
    
    agents = {
        "Random": RandomAgent(),
        "Greedy": GreedyAgent(),
        "Smart Planner": PrioritizedPlannerAgent(),
        "Real LLM (Llama-70B)": RealLLMAgent()
    }
    
    results = {}

    for name, agent in agents.items():
        print(f"\n🚀 Evaluating {name}...")
        total_metrics = {"career": 0, "family": 0, "stress": 0, "budget": 0}
        
        for _ in range(num_episodes):
            obs = env.reset()
            for _ in range(max_steps):
                action = agent.act(obs)
                obs = env.step(action)
                if obs.done: break
            
            # Capture final metrics
            for k in total_metrics:
                total_metrics[k] += obs.metrics.get(k, 0)
        
        # Average
        results[name] = {k: v / num_episodes for k, v in total_metrics.items()}
        print(f"✅ {name} results: {results[name]}")

    # Generate Plot
    from scripts.evaluate import generate_performance_plot
    generate_performance_plot(results, output_path="data/plots/real_model_performance.png")

if __name__ == "__main__":
    if not HF_TOKEN or "your_token" in HF_TOKEN:
        print("🛑 Please fill in your .env file with HF_TOKEN to test the real model.")
    else:
        run_evaluation()
