# WebWatcher: Breaking New Frontier of Vision-Language Deep Research Agent

<p align="center">
  <img src="./assets/webwatcher_logo.png" alt="logo" width="30%"/>
</p>



## 🥇 Introduction
In this paper, we introduce **WebWatcher**, a multimodal agent for deep research that possesses enhanced visual-language reasoning capabilities. Our work presents a unified framework that combines complex vision-language reasoning with multi-tool interaction.

<p align="center">
  <img src="./assets/webwatcher_main.png" alt="logo" width="80%"/>
</p>


Key features of our approach include:

<p align="center">
  <img src="./assets/distribution_level.png" alt="logo" width="80%"/>
</p>

- BrowseComp-VL Benchmark: We propose a new benchmark, BrowseComp-VL, to evaluate the capabilities of multimodal agents. This challenging dataset is designed for in-depth multimodal reasoning and strategic planning, mirroring the complexity of BrowseComp but extending it into the visual domain. It emphasizes tasks that require both visual perception and advanced information-gathering abilities.

<p align="center">
  <img src="./assets/data_pipelines.png" alt="logo" width="80%"/>
</p>

- Automated Trajectory Generation: To provide robust tool-use capabilities, we developed an automated pipeline to generate high-quality, multi-step reasoning trajectories. These trajectories, which are grounded in actual tool-use behavior and reflect procedural decision-making, are used for efficient cold-start training and further optimization via reinforcement learning. The agent is equipped with several tools, including Web Image Search, Web Text Search, Webpage Visit, Code Interpreter, and an internal OCR tool.

- Superior Performance: WebWatcher significantly outperforms proprietary baselines, RAG workflows, and other open-source agents across four challenging VQA benchmarks: Humanity's Last Exam (HLE)-VL, BrowseComp-VL, LiveVQA, and MMSearch. The WebWatcher-32B model, in particular, achieves an average score of 18.2% on HLE, surpassing the GPT-4o-based OmniSearch baseline. It also achieves top-tier performance on LiveVQA (58.7%) and MMSearch (55.3%), demonstrating stable and superior results on demanding, real-world visual search benchmarks.

## 🚀 Performance Highlights
<p align="center">
  <img src="./assets/webwatcher_performance_general.png" alt="logo" width="80%"/>
</p>

1. Complex Reasoning (HLE-VL): On the Human Life Exam (HLE-VL), a benchmark for multi-step complex reasoning, WebWatcher achieved a commanding lead with a Pass@1 score of 13.6%, substantially outperforming representative models including GPT-4o (9.8%), Gemini2.5-flash (9.2%), and Qwen2.5-VL-72B (8.6%).

2. Information Retrieval (MMSearch): In the MMSearch evaluation, WebWatcher demonstrated exceptional retrieval accuracy with a Pass@1 score of 55.3%, significantly surpassing Gemini2.5-flash (43.9%) and GPT-4o (24.1%), showcasing superior precision in retrieval tasks and robust information aggregation capabilities in complex scenarios.

3. Knowledge-Retrieval Integration (LiveVQA): On the LiveVQA benchmark, WebWatcher achieved a Pass@1 score of 58.7%, outperforming Gemini2.5-flash (41.3%), Qwen2.5-VL-72B (35.7%), and GPT-4o (34.0%).

4. Information Optimization and Aggregation (BrowseComp-VL): On BrowseComp-VL, the most comprehensively challenging benchmark, WebWatcher dominated with an average score of 27.0%, more than doubling the performance of mainstream models including GPT-4o (13.4%), Gemini2.5-flash (13.0%), and Claude-3.7 (11.2%).


## 🔧 Quick Start

### Step 1: Download the WebWatcher model

You can download WebWatcher via Hugging Face [🤗 HuggingFace](https://huggingface.co/Alibaba-NLP/).

### Step 2: Data Preparation

Before running inference, test set images need to be downloaded to the `infer/scripts_eval/images` folder. This can be accomplished by running `infer/scripts_eval/download_image.py`. If you encounter issues downloading images from our provided OSS URLs, please obtain the images from the original dataset source and place them in the corresponding `infer/scripts_eval/images` folder.

### Step 3: Model Preparation

Put a trained model that needs inference and a summery model (qwen-2.5-72b or more) into `infer/pretrain_model`.

### Step 4: Inference

Run `infer/scripts_eval/scripts/eval.sh` with the following required parameters:

- **benchmark**: Name of the dataset to test. Available options: `'hle'`, `'gaia'`, `'livevqa'`, `'mmsearch'`, `'simplevqa'`, `'bc_vl_v1'`, `'bc_vl_v2'`. These test sets should be pre-stored in `infer/vl_search_r1/eval_data` with naming convention like `hle.jsonl`. We have provided format examples for some datasets in `infer/vl_search_r1/eval_data`. If extending to new datasets, please ensure consistent formatting.
- **EXPERIMENT_NAME**: Name for this experiment (user-defined)
- **MODEL_PATH**: Path to the trained model
- **SUMMERY_MODEL_PATH**: Path to the summery model (qwen-2.5-72b)
- **IMG_SEARCH_KEY**: Google SerpApi key for image search
- **JINA_API_KEY**: Jina API key
- **TEXT_SEARCH_KEY**: Google SerpApi key for text search

**Note**: For image search tools, if you need to upload searched images to OSS, the following are required:
- **ALIBABA_CLOUD_ACCESS_KEY_ID**: Alibaba Cloud OSS access key ID
- **ALIBABA_CLOUD_ACCESS_KEY_SECRET**: Alibaba Cloud OSS access key secret

### Step 5: Evaluation

Run `infer/evaluation/run_eval.sh` with the following required parameters:

- **API_KEY**
- **BASE_URL**
- **JUDGE_MODEL**
- **MAX_WORKERS**


## 📑 Citation

If this work is helpful, please kindly cite as:

```bigquery
@article{geng2025webwatcher,
  title={WebWatcher: Breaking New Frontiers of Vision-Language Deep Research Agent},
  author={Geng, Xinyu and Xia, Peng and Zhang, Zhen and Wang, Xinyu and Wang, Qiuchen and Ding, Ruixue and Wang, Chenxi and Wu, Jialong and Zhao, Yida and Li, Kuan and others},
  journal={arXiv preprint arXiv:2508.05748},
  year={2025}
}
