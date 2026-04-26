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
        obs.metrics.get("energy", 0),
        obs.environment_feedback,
        gr.update(choices=[c.value for c in obs.available_choices], value=obs.available_choices[0].value),
        "\n".join(obs.inbox_preview),
        f"📅 {obs.current_time}"
    )

def take_step(choice_str, justification):
    try:
        action = LifeopsAction(choice=LifeActionChoice(choice_str), justification=justification)
        obs = env.step(action)
        
        # Color formatting for stress
        stress = obs.metrics.get("stress", 0)
        stress_color = "🔴" if stress > 80 else "🟢"
        
        return (
            obs.metrics.get("career", 0),
            obs.metrics.get("family", 0),
            f"{stress_color} {stress}",
            obs.metrics.get("energy", 0),
            obs.environment_feedback,
            str(obs.reward_metadata.get('explanation', "N/A")),
            "\n".join(obs.inbox_preview),
            f"📅 {obs.current_time}",
            "Episode Complete" if obs.done else "In Progress"
        )
    except Exception as e:
        return 0, 0, 0, 0, f"Error: {str(e)}", "{}", "", "", "Error"

# CSS for a modern Look
theme = gr.themes.Soft(
    primary_hub_palette=gr.themes.colors.blue,
    secondary_hub_palette=gr.themes.colors.slate,
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(theme=theme, title="LifeOps Agent Dashboard") as demo:
    gr.Markdown("# 🧬 LifeOps: Agent Dashboard")
    gr.Markdown("Training AI to balance Career, Family, and Health in a chaotic world simulation.")
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group():
                gr.Markdown("### 📥 Active Context")
                time_label = gr.Markdown("📅 **Friday, 9:00 AM**")
                conflict_box = gr.Textbox(label="Conflict Situation", interactive=False, lines=3)
                inbox_box = gr.Textbox(label="Recent Messages", interactive=False, lines=3)
            
            with gr.Group():
                gr.Markdown("### 🕹️ Action Controls")
                with gr.Row():
                    choice_dropdown = gr.Dropdown(label="Select Action", choices=[])
                    justification_input = gr.Textbox(label="Thought / Justification", placeholder="Explain your decision...")
                
                with gr.Row():
                    step_btn = gr.Button("🚀 Take Step", variant="primary")
                    reset_btn = gr.Button("♻️ Reset Simulation")

            feedback_box = gr.Textbox(label="Environment Feedback", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Metrics")
            with gr.Row():
                career_stat = gr.Number(label="💼 Career Score")
                family_stat = gr.Number(label="🏠 Family Score")
            with gr.Row():
                stress_stat = gr.Textbox(label="🔥 Stress Level")
                energy_stat = gr.Number(label="⚡ Energy")
            
            gr.Markdown("### ⚖️ Reward Rubric (Explanation)")
            reward_explanation = gr.Textbox(label="Judge's Notes", interactive=False, lines=4)
            
            status_label = gr.Label(value="Ready", label="Episode Status")

    # Wire up events
    reset_btn.click(
        reset_env, 
        outputs=[conflict_box, career_stat, family_stat, stress_stat, energy_stat, feedback_box, choice_dropdown, inbox_box, time_label]
    )
    
    step_btn.click(
        take_step,
        inputs=[choice_dropdown, justification_input],
        outputs=[career_stat, family_stat, stress_stat, energy_stat, feedback_box, reward_explanation, inbox_box, time_label, status_label]
    )

    # Initialize
    demo.load(reset_env, outputs=[conflict_box, career_stat, family_stat, stress_stat, energy_stat, feedback_box, choice_dropdown, inbox_box, time_label])

if __name__ == "__main__":
    demo.launch()
