# WebResearcher: An Iterative Deep-Research Agent

<p align="center">
  <!-- Placeholder for a logo. You can replace this with your own logo. -->
  <img src="./assets/webresearcher.jpg" alt="logo" width="30%"/>
</p>

![version](https://img.shields.io/badge/version-1.0.0-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2509.13309-b31b1b.svg)](https://arxiv.org/abs/2509.13309) <!-- Placeholder link -->


## 🥇 Introduction

- **WebResearcher** is an autonomous agent built upon a novel **Iterative Deep-Research Paradigm**. It is designed to emulate the sophisticated cognitive workflow of human experts, moving beyond simple information retrieval to autonomously deconstruct complex problems, orchestrate advanced tool use, and synthesize findings into coherent, evidence-grounded narratives.

- Current open-source research agents often rely on a **mono-contextual, linear accumulation** of information. This approach is fundamentally flawed, suffering from:
    1.  **Cognitive Workspace Suffocation:** An ever-expanding context window constrains the model's ability to perform deep, complex reasoning.
    2.  **Irreversible Noise Contamination:** Irrelevant information and early errors accumulate and dilute the context, propagating biases.
    3.  **Lack of Periodic Synthesis:** The linear process prevents the agent from pausing to distill, re-evaluate, and strategically plan its next steps.

- **WebResearcher** overcomes these limitations by deconstructing the research process into discrete rounds. In each round, the agent reasons over its current knowledge, synthesizes new insights into an evolving **summary report**, and then charts its course for the next action. This evolving report acts as the agent's central memory, ensuring a focused cognitive workspace and enabling sustained, high-quality reasoning and practically unbounded research depth.

- To fuel our agent, we developed a **Scalable Data Synthesis Engine** that programmatically generates large-scale, high-quality, HLE-style datasets. This data powers a specialized multi-stage training pipeline, including Rejection-based Fine-Tuning (RFT) and Reinforcement Learning with Verifiable Rewards (RLVR), to instill robust tool use and sharpen logical deduction.


## The WebResearcher Paradigm

### 1. The Iterative Deep-Research Paradigm

Instead of linearly accumulating information, WebResearcher deconstructs research into discrete rounds. Each round is powered by a lean, reconstructed **Workspace** and produces a structured response containing `Think`, `Report`, and `Action`.

-   **Think:** The agent's internal monologue for reasoning and planning. It is not passed to subsequent rounds to prevent clutter.
-   **Report:** The agent’s evolving central memory. It synthesizes new findings into a coherent, high-density summary that is carried forward to the next round.
-   **Action:** The final, machine-parseable decision, which is either a `Tool Call` (e.g., Search, Visit, Python) or the `Final Answer`.

This cyclical process of synthesis and reconstruction prevents cognitive suffocation and noise contamination, enabling sustained, deep reasoning.

<p align="center">
  <img src="./assets/paradigm.png" alt="Paradigm Comparison" width="100%"/>
  <br>
  <em>Figure: Mono-contextual Paradigm (Top) vs. WebResearcher Paradigm (Bottom).</em>
</p>

### 2. Scalable Data Synthesis Engine

To overcome the data bottleneck for training advanced agents, we built a scalable data engine. This engine uses a multi-agent framework in a three-stage workflow to automatically generate large-scale, high-quality, and complex reasoning tasks.

1.  **Seed Data Generation:** An `ItemWriter` agent creates initial question-answer pairs from a curated corpus of documents.
2.  **Iterative Complexity Escalation:** The agent, now augmented with tools (Search, Scholar, Python), iteratively refines and expands the questions, increasing their intellectual depth and complexity.
3.  **Rigorous Quality Control:** A `QuestionSolver` agent and a `Judge` agent form a gauntlet to filter out simple questions, verify the correctness of complex ones, and ensure the final dataset is challenging and accurate.

<p align="center">
  <img src="./assets/webresearcher-data.png" alt="Data Synthesis Workflow" width="90%"/>
  <br>
  <em>Figure: The three-stage data synthesis workflow.</em>
</p>

### 3. Training and Inference

-   **Rejection Sampling Fine-Tuning (RFT):** We first fine-tune the base model on high-quality trajectories where the final answer exactly matches the ground truth. This instills robust tool-use competence and knowledge-grounded reasoning.
-   **Reinforcement Learning (RL):** We further sharpen the agent's multi-step logical deduction abilities using Reinforcement Learning with Verifiable Rewards (RLVR).
-   **Test-Time Scaling (TTS) with `last-k-fusion`:** At inference, we boost performance by running multiple parallel inference rollouts and using a dedicated **Fusion Agent** to synthesize the final answer from the most critical final steps of each trajectory.

<p align="center">
  <img src="./assets/tts-fig-v1.png" alt="Last-k-Fusion" width="100%"/>
  <br>
  <em>Figure: Illustration of our `last-k-fusion` technique for Test-Time Scaling.</em>
</p>

## 🎥 Demos

⌛️ Demos showcasing WebResearcher's capabilities on complex research tasks will be released soon!

## 📑 Citation

If you find our work helpful, please kindly cite our paper:

```bibtex
@misc{qiao2025webresearcherunleashingunboundedreasoning,
      title={WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents}, 
      author={Zile Qiao and Guoxin Chen and Xuanzhong Chen and Donglei Yu and Wenbiao Yin and Xinyu Wang and Zhen Zhang and Baixuan Li and Huifeng Yin and Kuan Li and Rui Min and Minpeng Liao and Yong Jiang and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2509.13309},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.13309}, 
}
```
