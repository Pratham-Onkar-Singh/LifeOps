import matplotlib.pyplot as plt
import json
import os

def generate_performance_plot(results_data: dict, output_path: str = "data/plots/performance.png"):
    """
    Generates a professional bar chart comparing agents across life axes.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    agents = list(results_data.keys())
    metrics = ["career", "family", "stress", "budget"]
    
    # Example structure: {"Random": {"career": 10, "family": 20, ...}, "Trained": {...}}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.2
    
    for i, agent in enumerate(agents):
        values = [results_data[agent].get(m, 0) for m in metrics]
        ax.bar([p + i*width for p in x], values, width, label=agent)

    ax.set_title("LifeOps: Agent Performance Comparison")
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylabel("Score")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Performance plot saved to {output_path}")

if __name__ == "__main__":
    # Sample data for the README (to be replaced by real evaluation results)
    sample_results = {
        "Random": {"career": 5, "family": 10, "stress": 40, "budget": 900},
        "Greedy": {"career": 80, "family": -20, "stress": 95, "budget": 1100},
        "Trained (Llama-3)": {"career": 65, "family": 75, "stress": 25, "budget": 1050}
    }
    generate_performance_plot(sample_results)
