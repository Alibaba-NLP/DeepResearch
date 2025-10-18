<div align="center">
<p align="center">
  <img src="assets/overall.jpg" width="50%" height="50%" />
</p>
</div>

<div align="center">
<h1>WebWalker: Benchmarking LLMs in Web Traversal</h1>
</div>

<div align="center">
<img src="https://img.shields.io/github/stars/Alibaba-NLP/WebWalker?color=yellow" alt="Stars">
<a href='https://huggingface.co/papers/2501.07572'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Discussion-orange'></a>
<a href='https://huggingface.co/collections/callanwu/webwalker-677f6527407edfda44098b09'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Colloectionss-blue'></a>
<a href='https://huggingface.co/datasets/callanwu/WebWalkerQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green'></a>
<a href='https://huggingface.co/spaces/callanwu/WebWalkerQALeadeboard'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-yellow'></a><br>
<a href='https://arxiv.org/pdf/2501.07572'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>

<!-- **Authors:** -->
<br>

_**Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang**_

_**Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, Fei Huang<br>**_

<!-- **Affiliations:** -->

_Tongyi Lab <img src="./assets/tongyi.png" width="14px" style="display:inline;">, Alibaba Group_

👏 Welcome to try web traversal via our **[<img src="./assets/tongyi.png" width="14px" style="display:inline;"> Modelscope online demo](https://www.modelscope.cn/studios/iic/WebWalker/)** or **[🤗 Huggingface online demo](https://huggingface.co/spaces/callanwu/WebWalker)**!

<p align="center">
<a href="https://alibaba-nlp.github.io/WebWalker/">[🤖Project]</a>
<a href="https://arxiv.org/pdf/2501.07572">[📄Paper]</a>
<a href="## 🚩Citation">[🚩Citation]</a>

</div>

Repo for [_WebWalker: Benchmarking LLMs in Web Traversal_](https://arxiv.org/pdf/2501.07572)

# 📖 Quick Start

- 🌏 The **Online Demo** is available at [ModelScope](https://www.modelscope.cn/studios/jialongwu/WebWalker/) and [HuggingFace](https://huggingface.co/spaces/callanwu/WebWalker) now！

- 🤗 The **WebWalkerQA** dataset is available at[ HuggingFace Datasets](https://huggingface.co/datasets/callanwu/WebWalkerQA)!

- 🤗 The **WebWalkerQA** Leaderborad is available at[ HuggingFace Space](https://huggingface.co/spaces/callanwu/WebWalkerQALeadeboard)!

<img src="assets/demo.gif">

# 📌 Introduction

- We construct a challenging benchmark, **WebWalkerQA**, which is composed of **680** queries from four real-world scenarios across over **1373** webpages.
- To tackle the challenge of web-navigation tasks requiring long context, we propose **WebWalker**, which utilizes a multi-agent framework for effective memory management.
- Extensive experiments show that the WebWalkerQA is **challenging**, and for information-seeking tasks, **vertical exploration** within the page proves to be beneficial.

<div align="center">
    <img src="assets/method.jpg" width="80%" height="auto" />
</div>

# 📚 WebWalkerQA Dataset

The json item of WebWalkerQA dataset is organized in the following format:

```json
{
  "Question": "When is the paper submission deadline for the ACL 2025 Industry Track, and what is the venue address for the conference?",
  "Answer": "The paper submission deadline for the ACL 2025 Industry Track is March 21, 2025. The conference will be held in Brune-Kreisky-Platz 1.",
  "Root_Url": "https://2025.aclweb.org/",
  "Info": {
    "Hop": "multi-source",
    "Domain": "Conference",
    "Language": "English",
    "Difficulty_Level": "Medium",
    "Source_Website": [
      "https://2025.aclweb.org/calls/industry_track/",
      "https://2025.aclweb.org/venue/"
    ],
    "Golden_Path": ["root->call>student_research_workshop", "root->venue"]
  }
}
```

🤗 The WebWalkerQA Leaderboard is is available at[ HuggingFace](https://huggingface.co/spaces/callanwu/WebWalkerQALeadeboard)!

You can load the dataset via the following code:

```python
from datasets import load_dataset
ds = load_dataset("callanwu/WebWalkerQA", split="main")
```

Additionally, we possess a collection of approximately 14k silver QA pairs, which, although not yet carefully human-verified.
You can load the silver dataset by changing the split to `silver`.

## 💡 Perfomance

### 📊 Result on Web Agents

The performance on Web Agents are shown below:

<div align="center">
    <img src="assets/agent_result.jpg" width="80%" height="auto" />
</div>

### 📊 Result on RAG-Systems

<div align="center">
    <img src="assets/rag_result.jpg" width="80%" height="auto" />
</div>

🤗 The WebWalkerQA Leaderboard is is available at[ HuggingFace](https://huggingface.co/spaces/callanwu/WebWalkerQALeadeboard)!

🚩 Welcome to submit your method to the leaderboard!

# 🛠 Dependencies

```bash
conda create -n webwalker python=3.10
git clone https://github.com/alibaba-nlp/WebWalker.git
cd WebWalker

# Install requirements
pip install -r requirements.txt
# Run post-installation setup
crawl4ai-setup
# Verify your installation
crawl4ai-doctor
```

### 💻 Running WebWalker Demo Locally

🔑 Before running, please export the OPENAI API key or Dashscope API key as an environment variable:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_API_BASE_URL=YOUR_API_BASE_URL
```

or

```bash
export DASHSCOPE_API_KEY=YOUR_API_KEY
```

> You can use other supported API keys with Qwen-Agent. For more details, please refer to the [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent/tree/main/qwen_agent/llm). To configure the API key, modify the code in lines 44-53 of [`src/app.py`](https://github.com/Alibaba-NLP/WebWalker/blob/main/src/app.py#L44-L53).

Then, run the `app.py` file with Streamlit:

```bash
cd src
streamlit run app.py
```

### Runing RAG-System on WebWalkerQA

```bash
cd src
python rag_system.py --api_name [API_NAME] --output_file [OUTPUT_PATH]
```

The details of environment setup can be found in the [README.md](./src/README.md) in the `src` folder.

# 🔍 Evaluation

The evaluation script for accuracy of the output answers using GPT-4 can be used as follows:

```bash
cd src
python evaluate.py --input_path [INPUT_PATH]--output_path [OUTPUT_PATH]
```

## 🌻Acknowledgement

- This work is implemented by [ReACT](https://github.com/ysymyth/ReAct), [Qwen-Agents](https://github.com/QwenLM/Qwen-Agent), [LangChain](https://github.com/langchain-ai/langchain). Sincere thanks for their efforts.
- We sincerely thank the contributors and maintainers of [Crawl4AI](https://github.com/unclecode/crawl4ai) for their open-source tool❤️, which helped us get web pages in a Markdown-like format.

  <a href="https://github.com/unclecode/crawl4ai">
  <img src="https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/assets/powered-by-disco.svg" alt="Powered by Crawl4AI" width="100"/>
</a>

- The repo is contributed by [Jialong Wu](https://callanwu.github.io/), if you have any questions, please feel free to contact via jialongwu@alibaba-inc.com or jialongwu@seu.edu.cn or create an issue.

## 🚩Citation

If this work is helpful, please kindly cite as:

```bigquery
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

## Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=Alibaba-NLP/WebAgent&type=Date)](https://www.star-history.com/#Alibaba-NLP/WebAgent&Date)

</div>
