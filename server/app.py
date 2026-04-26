# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integrated FastAPI + Gradio Application for LifeOps.
This file serves as the main entry point for the Hugging Face Space.
"""

import os
import sys
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from models import LifeopsAction, LifeopsObservation
from server.lifeops_environment import LifeopsEnvironment

# 1. Create the OpenEnv standard FastAPI app
# This provides /reset, /step, /state, /health
app = create_app(
    LifeopsEnvironment,
    LifeopsAction,
    LifeopsObservation,
    env_name="lifeops",
    max_concurrent_envs=5,
)

# 2. Build the "Judge-Ready" Gradio UI
def build_demo():
    from app.gradio_app import demo as gradio_ui
    return gradio_ui

# 3. Mount Gradio onto the FastAPI app
# Judges can visit the root URL to see the dashboard
app = gr.mount_gradio_app(app, build_demo(), path="/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = parser.parse_args()
    
    print(f"🚀 LifeOps Production Server starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
