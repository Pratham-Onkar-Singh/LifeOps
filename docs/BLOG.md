# Teaching AI to Choose: The LifeOps Story

## The Problem
We've all been there: a critical deadline at work vs. a precious moment with family. As humans, we navigate these "chaotic" trade-offs every day. But can an AI assistant do the same?

Existing RL environments for LLMs focus on "correct" answers (Math, Code). But real life isn't about being correct; it's about being balanced.

## Our Solution: LifeOps
We built **LifeOps**, an OpenEnv-compliant simulation where LLM agents are dropped into high-pressure human conflicts. Using **GRPO** and **verifiable rewards**, we train agents not just to solve tasks, but to manage life.

### Key Innovations
1. **Multi-axis Rewards:** We don't just give a +1 or -1. We score agents on Career, Family, Stress, and Budget simultaneously.
2. **Dynamic Scenario Engine:** From "Health vs. Deadlines" to "Budget vs. Socializing," the environment generates unpredictable life events.
3. **Verifiable Trade-offs:** Every choice has measurable consequences in a simulated world model.

## Results
During our initial training runs with **Unsloth**, we saw agents move from "blindly working" to "strategic balancing." By rewarding family time and penalizing excessive stress, the models learned to prioritize long-term sustainability over short-term gains.

## The Future
LifeOps is just the beginning. We envision a future where every AI assistant is trained in environments like LifeOps to become more empathetic, balanced, and truly helpful in the chaotic mess of human existence.

---
*Built with ❤️ for the OpenEnv Hackathon 2026.*
