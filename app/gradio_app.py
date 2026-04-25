import gradio as gr
import sys
import os

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.lifeops_environment import LifeopsEnvironment
from models import LifeopsAction, LifeActionChoice

env = LifeopsEnvironment()

def reset_env():
    obs = env.reset()
    return (
        obs.active_conflict,
        obs.metrics.get("career", 0),
        obs.metrics.get("family", 0),
        obs.metrics.get("stress", 0),
        obs.metrics.get("budget", 0),
        obs.environment_feedback,
        gr.update(choices=[c.value for c in obs.available_choices], value=obs.available_choices[0].value if obs.available_choices else None),
        "\n".join(obs.inbox_preview)
    )

def take_step(choice_str, justification):
    try:
        action = LifeopsAction(choice=LifeActionChoice(choice_str), justification=justification)
        obs = env.step(action)
        return (
            obs.metrics.get("career", 0),
            obs.metrics.get("family", 0),
            obs.metrics.get("stress", 0),
            obs.metrics.get("budget", 0),
            obs.environment_feedback,
            str(obs.reward_metadata),
            "\n".join(obs.inbox_preview)
        )
    except Exception as e:
        return 0, 0, 0, 0, f"Error: {str(e)}", "{}", ""

with gr.Blocks(title="LifeOps - Agent Training Dashboard") as demo:
    gr.Markdown("# 🧬 LifeOps: Chaotic Life Management")
    gr.Markdown("Training LLMs to balance the impossible trade-offs of modern life.")
    
    with gr.Row():
        with gr.Column(scale=2):
            conflict_box = gr.Textbox(label="Active Conflict", interactive=False, lines=3)
            inbox_box = gr.Textbox(label="Recent Messages", interactive=False, lines=3)
            feedback_box = gr.Textbox(label="Environment Feedback", interactive=False)
            
            with gr.Row():
                choice_dropdown = gr.Dropdown(label="Select Action", choices=[])
                justification_input = gr.Textbox(label="Justification", placeholder="Why this choice?")
            
            step_btn = gr.Button("Execute Action", variant="primary")
            reset_btn = gr.Button("New Scenario")

        with gr.Column(scale=1):
            gr.Markdown("### Life Metrics")
            career_stat = gr.Number(label="💼 Career Score")
            family_stat = gr.Number(label="🏠 Family Score")
            stress_stat = gr.Number(label="🔥 Stress Level")
            budget_stat = gr.Number(label="💰 Budget ($)")
            
            gr.Markdown("### Reward Breakdown")
            reward_json = gr.Code(label="Metadata / Rubric", language="json")

    # Wire up events
    reset_btn.click(
        reset_env, 
        outputs=[conflict_box, career_stat, family_stat, stress_stat, budget_stat, feedback_box, choice_dropdown, inbox_box]
    )
    
    step_btn.click(
        take_step,
        inputs=[choice_dropdown, justification_input],
        outputs=[career_stat, family_stat, stress_stat, budget_stat, feedback_box, reward_json, inbox_box]
    )

    # Initialize
    demo.load(reset_env, outputs=[conflict_box, career_stat, family_stat, stress_stat, budget_stat, feedback_box, choice_dropdown, inbox_box])

if __name__ == "__main__":
    demo.launch()
