<div align="center">

<h2>WebAgent for Information Seeking built by Tongyi Lab, Alibaba Group <img src="./assets/tongyi.png" width="30px" style="display:inline;"></h2>

</div>
<p align="center">
<a href="https://trendshift.io/repositories/14217" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14217" 
alt="Alibaba-NLP%2FWebAgent | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Alibaba-NLP/WebShaper" target="_blank">WebShaperQA</a> ｜
<img src="./assets/tongyi.png" width="14px" style="display:inline;"> <a href="https://modelscope.cn/datasets/iic/WebShaper" target="_blank">WebShaperQA</a> ｜
🤗 <a href="https://huggingface.co/Alibaba-NLP/WebSailor-3B" target="_blank">WebSailor-3B</a> ｜
<img src="./assets/tongyi.png" width="14px" style="display:inline;"> <a href="https://modelscope.cn/models/iic/WebSailor-3B" target="_blank">ModelScope WebSailor-3B</a> |
</p>

<p align="center">
🤗 <a href="https://huggingface.co/Alibaba-NLP/WebDancer-32B" target="_blank">WebDancer-QwQ-32B</a>  | 
<img src="./assets/tongyi.png" width="14px" style="display:inline;"> <a href="https://modelscope.cn/models/iic/WebDancer-32B" target="_blank">ModelScope WebDancer-QwQ-32B</a> |
🤗 <a href="https://huggingface.co/datasets/callanwu/WebWalkerQA" target="_blank">WebWalkerQA</a>

</p>

<div align="center">
<p align="center">
  <img src="assets/roadmap.png" width="100%" height="400%" />
</p>
</div>

> You can check the paper of [WebDancer](https://arxiv.org/pdf/2505.22648) and [WebWalker](https://arxiv.org/pdf/2501.07572) and [WebSailor](https://arxiv.org/pdf/2507.02592) and [WebShaper](https://arxiv.org/pdf/2507.15061).

> 💥 💥 💥 Stay tuned for more updates! We are working on building native agentic model based on the Browser and more open-domain environments!

- [**WebShaper**](WebShaper) (Preprint 2025) - WebShaper: Agentically Data Synthesizing via Information-Seeking
  Formalization
- [**WebSailor**](WebSailor) (Preprint 2025) - WebSailor: Navigating Super-human Reasoning for Web Agent
- [**WebDancer**](WebDancer) (Preprint 2025) - WebDancer: Towards Autonomous Information Seeking Agency
- [**WebWalker**](WebWalker) (ACL 2025) - WebWalker: Benchmarking LLMs in Web Traversal

## 📰 News and Updates

- `2025.08.06` 🔥🔥🔥**WebSailor-7B** is [released](https://huggingface.co/Alibaba-NLP/WebSailor-7B).
- `2025.07.22` 🔥🔥🔥We release **WebShaper**: Agentically Data Synthesizing via Information-Seeking Formalization.
- `2025.07.11` 🔥🔥🔥**WebSailor-3B** is [released](https://huggingface.co/Alibaba-NLP/WebSailor-3B). You can deploy it with one click using <img src="./assets/aliyun.png" width="14px" style="display:inline;"> [Alibaba Cloud's FunctionAI](https://functionai.console.aliyun.com/template-detail?template=Alibaba-NLP-WebSailor-3B) in ten minutes!
- `2025.07.03` 🔥🔥🔥We release **WebSailor**, an agentic search model specialized in performing extremely complex information seeking tasks, achieving open-source SOTA on some of the most difficult browsing benchmarks. **WebSailor** topped the HuggingFace [daily papers](https://huggingface.co/papers/2507.02592).
- `2025.06.23` 🔥🔥🔥The model, interactive demo, and some of the data of **WebDancer** have been open-sourced. You're welcome to try them out!
- `2025.05.29` 🔥🔥🔥We release **WebDancer**, a native agentic search model towards autonomous information seeking agency and _Deep Research_-like model.
- `2025.05.15` **WebWalker** is accepted by ACL 2025 main conference.
- `2025.01.14` We release **WebWalker**, a benchmark for LLMs in web traversal and a multi-agent framework for information seeking.

## 💎 Results Showcase

<div align="center">
<p align="center">
  <img src="assets/webagent-gaia.png" width="800%" height="400%" />
</p>
</div>

<div align="center">
<p align="center">
  <img src="assets/webagent-bc.png" width="800%" height="400%" />
</p>
</div>

## 💡 Features for WebShaper

- A **`formalization-driven`** data synthesis method for information-seeking agents, grounded in our proposed task formalization. Leveraging this method, we construct the **WebShaper** dataset, which enables systematic generation of IS instances.
- We propose an agentic Expander that iteratively generates and validates questions in alignment with the formalization.
- We conduct extensive experiments across multiple benchmarks to evaluate the effectiveness of WebShaper. We achieve new state-of-the-art results on **GAIA** (**60.19**) and **WebWalkerQA** (**52.50**) benchmarks.

## ⛵️ Features for WebSailor

- A complete post-training methodology enabling models to engage in extended thinking and information seeking, ultimately allowing them to successfully complete extremely complex tasks previously considered unsolvable.
- Introduces **SailorFog-QA**, a scalable QA benchmark with high uncertainty and difficulty, curated with a novel data synthesis method through graph sampling and information obfuscation. Example SailorFog-QA data samples can be found at: [`WebSailor/dataset/sailorfog-QA.jsonl`](WebSailor/dataset/sailorfog-QA.jsonl)
- Effective post-training pipeline consisting of (1) high-quality reconstruction of concise reasoning from expert trajectories for clean supervision, (2) a two-stage training process involving an RFT cold start stage, followed by **Duplicating Sampling Policy Optimization (DUPO)**, an efficient agentic RL algorithm excelling in effectiveness and efficiency.
- WebSailor-72B significantly outperforms all open-source agents and frameworks while closing the performance gap with leading proprietary systems, achieving a score of **12.0%** on BrowseComp-en, **30.1%** on BrowseComp-zh, and **55.4%** on GAIA.
- **The checkpoint is coming soon.**

## 🌐 Features for WebDancer

- Native agentic search reasoning model using ReAct framework towards autonomous information seeking agency and _Deep Research_-like model.
- We introduce a four-stage training paradigm comprising **browsing data construction, trajectory sampling, supervised fine-tuning for effective cold start, and reinforcement learning for improved generalization**, enabling the agent to autonomously acquire autonomous search and reasoning skills.
- Our data-centric approach integrates trajectory-level supervision fine-tuning and reinforcement learning (DAPO) to develop a scalable pipeline for **training agentic systems** via SFT or RL.
- WebDancer achieves a Pass@3 score of 64.1% on GAIA and 62.0% on WebWalkerQA.

## 🚀 Quick Start

You need to enter the [`WebDancer`](WebDancer) folder for the following commands.

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

Edit the following keys in [`WebDancer/scripts/run_demo.sh`](WebDancer/scripts/run_demo.sh):

- `GOOGLE_SEARCH_KEY`, you can get it from [serper](https://serper.dev/).
- `JINA_API_KEY`, you can get it from [jina](https://jina.ai/api-dashboard/).
- `DASHSCOPE_API_KEY`, you can get it from [dashscope](https://dashscope.aliyun.com/).

Then, launch the demo with Gradio to interact with the WebDancer model:

```bash
cd scripts
bash run_demo.sh
```

## 🎥 WebSailor Demos

We provide demos for BrowseComp-en, BrowseComp-zh and Daily Use. Our model can complete highly difficult and uncertain tasks requiring massive information acquisition and complex reasoning.

<div align="center">
    <h3>BrowseComp-en</h3>
    <video src= "https://github.com/user-attachments/assets/2dc0b03a-c241-4f70-bf11-92fda28020fa"/>
</div>

<div align="center">
    <h3>BrowseComp-zh</h3>
    <video src="https://github.com/user-attachments/assets/f9aed746-ffc8-4b76-b135-715ec0eab544" />
</div>

<div align="center">
    <h3>Daily Use</h3>
    <video src="https://github.com/user-attachments/assets/1299c5a8-cee3-4a70-b68b-c5d227cf8055" />
</div>

## 🎥 WebDancer Demos

We provide demos for WebWalkerQA, GAIA and Daily Use.
Our model can execute the long-horizon tasks with **multiple steps** and **complex reasoning**, such as web traversal, information seeking and question answering.

<div align="center">
    <h3>WebWalkerQA</h3>
    <video src="https://github.com/user-attachments/assets/0bbaf55b-897e-4c57-967d-a6e8bbd2167e" />
</div>

<div align="center">
    <h3>GAIA</h3>
    <video src="https://github.com/user-attachments/assets/935c668e-6169-4712-9c04-ac80f0531872" />
</div>

<div align="center">
    <h3>Daily Use</h3>
    <video src="https://github.com/user-attachments/assets/d1d5b533-4009-478b-bd87-96b86389327d" />
</div>

## 📃 License

The content of this project itself is licensed under [LICENSE](LICENSE).

## 🚩 Citation

If this work is helpful, please kindly cite as:

```bigquery
@misc{tao2025webshaper,
      title={WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization},
      author={Zhengwei Tao and Jialong Wu and Wenbiao Yin and Junkai Zhang and Baixuan Li and Haiyang Shen and Kuan Li and Liwen Zhang and Xinyu Wang and Yong Jiang and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2507.15061},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.15061},
}
@misc{li2025websailor,
      title={WebSailor: Navigating Super-human Reasoning for Web Agent},
      author={Kuan Li and Zhongwang Zhang and Huifeng Yin and Liwen Zhang and Litu Ou and Jialong Wu and Wenbiao Yin and Baixuan Li and Zhengwei Tao and Xinyu Wang and Weizhou Shen and Junkai Zhang and Dingchu Zhang and Xixi Wu and Yong Jiang and Ming Yan and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2507.02592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.02592},
}
@misc{wu2025webdancer,
      title={WebDancer: Towards Autonomous Information Seeking Agency},
      author={Jialong Wu and Baixuan Li and Runnan Fang and Wenbiao Yin and Liwen Zhang and Zhengwei Tao and Dingchu Zhang and Zekun Xi and Yong Jiang and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2505.22648},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22648},
}
@misc{wu2025webwalker,
      title={WebWalker: Benchmarking LLMs in Web Traversal},
      author={Jialong Wu and Wenbiao Yin and Yong Jiang and Zhenglin Wang and Zekun Xi and Runnan Fang and Deyu Zhou and Pengjun Xie and Fei Huang},
      year={2025},
      eprint={2501.07572},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.07572},
}
```

## 🌟 Misc

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Alibaba-NLP/WebAgent&type=Date)](https://www.star-history.com/#Alibaba-NLP/WebAgent&Date)

</div>

## 🚩 Talent Recruitment

🔥🔥🔥 We are hiring! Research intern positions are open (based in Hangzhou、Beijing、Shanghai)

📚 **Research Area**：Web Agent, Search Agent, Agent RL, MultiAgent RL, Agentic RAG

☎️ **Contact**：[yongjiang.jy@alibaba-inc.com]()

## Contact Information

For communications, please contact Yong Jiang (yongjiang.jy@alibaba-inc.com).
