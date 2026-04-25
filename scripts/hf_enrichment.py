import os
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def enrich_scenarios_with_hf(num_scenarios: int = 10):
    """
    Uses the Hugging Face Inference API to generate highly realistic, 
    emotionally complex life conflicts.
    """
    if not HF_TOKEN or "your_token" in HF_TOKEN:
        print("❌ Error: HF_TOKEN not found in .env file.")
        return

    print(f"📡 Connecting to Hugging Face Inference API for {num_scenarios} scenarios...")
    
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-70B-Instruct"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    prompt = (
        "Generate a list of 10 unique and emotionally complex life conflict scenarios for an AI training environment. "
        "Each scenario should force a trade-off between Career, Family, Health, or Budget. "
        "Format as a JSON list: "
        '[{"id": "...", "description": "...", "choices": ["stay_late_work", "go_to_family_event", ...]}]'
    )

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        result = response.json()
        print("✅ Received enriched scenarios from Llama-70B!")
        return result

    except Exception as e:
        print(f"❌ Failed to connect to HF API: {str(e)}")
        return None

if __name__ == "__main__":
    enrich_scenarios_with_hf()
