# WebDancer: Towards Autonomous Information Seeking Agency

<p align="center">
  <img src="./assets/webdancer.png" alt="logo" width="40%"/>
</p>

![version](https://img.shields.io/badge/version-1.0.0-blue)<a href="https://arxiv.org/pdf/2505.22648">![arXiv](https://img.shields.io/badge/arXiv-2505.22648-b31b1b)</a>

## 🕺 Introduction

- We propose **WebDancer**, a novel end-to-end agentic training framework designed to enhance the multi-step information-seeking capabilities of web-based agents.
- We introduce a four-stage training paradigm comprising **browsing data construction, trajectory sampling, supervised fine-tuning for effective cold start, and reinforcement learning for improved generalization**, enabling the agent to autonomously acquire robust search and reasoning skills.
- Our data-centric approach integrates trajectory-level supervision and online learning to develop a scalable pipeline for **training agentic systems**.
- We instantiate this framework in a ReAct-based agent and conduct extensive experiments on **GAIA** and **WebWalkerQA** benchmarks. Results demonstrate that WebDancer achieves strong performance across diverse tasks, validating the effectiveness of our proposed paradigm and providing systematic insights for future agent development.

## 🚀 Performance

<p align="center">
  <img src="./assets/performance.png" alt="logo" width="80%"/>
</p>

## 🚀 Quick Start

### Step 0: Set Up the Environment

```bash
conda create -n webdancer python=3.12
pip install -r requirements.txt
```

### Step 1: Deploy the Model

Download the WebDancer model from [🤗 HuggingFace](https://huggingface.co/Alibaba-NLP/WebDancer-32B) and deploy it using the provided scripts with [sglang](https://github.com/sgl-project/sglang).

```bash
cd scripts
bash deploy_model.sh WebDancer_PATH
```

> **Note:** Replace `WebDancer_PATH` with the actual path to the downloaded model.

### Step 2: Run the Demo

Edit the following keys in [`scripts/run_demo.sh`](scripts/run_demo.sh):

- `GOOGLE_SEARCH_KEY`, you can get it from [serper](https://serper.dev/).
- `JINA_API_KEY`, you can get it from [jina](https://jina.ai/api-dashboard/).
- `DASHSCOPE_API_KEY`, you can get it from [dashscope](https://dashscope.aliyun.com/).

Then, launch the demo with Gradio to interact with the WebDancer model:

```bash
cd scripts
bash run_demo.sh
```

## 🎥 Demos

We provide demos for WebWalkerQA, GAIA and Daily Use.
Our model can execute the long-horizon tasks with **multiple steps** and **complex reasoning**, such as web traversal, information seeking and question answering.

<div align="center">
    <h3>WebWalkerQA</h3>
    <video src="https://github.com/user-attachments/assets/0bbaf55b-897e-4c57-967d-a6e8bbd2167e" />
</div>

<div align="center">
    <h3>GAIA</h3>
    <video src="https://github.com/user-attachments/assets/0bbaf55b-897e-4c57-967d-a6e8bbd2167e" />
</div>

<div align="center">
    <h3>Daily Use</h3>
    <video src="https://github.com/user-attachments/assets/d1d5b533-4009-478b-bd87-96b86389327d" />
</div>

⌛️ The deployment of models and demos will be updated soon.

## Four-Stage Training Paradigm

### 1. Browsing Data Construction

<p align="center">
  <img src="./assets/data_construction.png" alt="logo" width="80%"/>
</p>

The sampled QA data can be found in
[`datasets/sample_qa.jsonl`](datasets/sample_qa.jsonl).

### 2. Trajectory Sampling

The sampled trajectory data for SFT can be found in
[`datasets/sample_qa.jsonl`](datasets/sample_traj.jsonl).

<p align="center">
  <img src="./assets/framework.png" alt="logo" width="80%"/>
</p>

### 3. Supervised Fine-Tuning

For SFT training, you can refer to the training scipts of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### 4. Reinforcement Learning

We use the modified [verl](https://github.com/volcengine/verl) for RL training.

## 🤩 Acknowledgements

This work is implemented based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [verl](https://github.com/volcengine/verl).
We greatly appreciate their valuable contributions to the community, especially for [WebThinker](https://github.com/RUC-NLPIR/WebThinker).

## 📑 Citation

If this work is helpful, please kindly cite as:

```bigquery
@misc{wu2025webdancer,
      title={WebDancer: Towards Autonomous Information Seeking Agency},
      author={Jialong Wu and Baixuan Li and Runnan Fang and Wenbiao Yin and Liwen Zhang and Zhengwei Tao and Dingchu Zhang and Zekun Xi and Yong Jiang and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2505.22648},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22648},
}
```
