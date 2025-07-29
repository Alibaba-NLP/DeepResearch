# nanoWebSailor: A Re-implementation of WebSailor

## Project Overview

`nanoWebSailor` is a third-party open-source re-implementation based on the paper [《WebSailor: Navigating Super-human Reasoning for Web Agent》](https://arxiv.org/abs/2507.02592). This project aims to reproduce the core innovations from the paper:

- **SailorFog-QA Data Synthesis**: Generate high-difficulty training data through complex knowledge graphs and information obfuscation techniques
- **Knowledge Graph Expansion**: Build densely interconnected graph structures starting from rare entities
- **Trajectory Reconstruction Methods**: Reconstruct concise and effective reasoning chains from expert LLM trajectories
- **ReAct Agent Framework**: Support complex web information retrieval and reasoning tasks

## Key Features

### 🔬 Data Synthesis Module
- **Entity Generator**: Generate rare and complex knowledge entities
- **Knowledge Graph Builder**: Construct graph structures based on Wikidata and random walks
- **Graph Expander**: Iteratively expand knowledge graphs from seed entities
- **Subgraph Sampler**: Multi-strategy sampling of high-quality subgraphs
- **QA Generator**: Level 3 complex question-answer pair generation
- **Information Obfuscation**: Intelligent obfuscation techniques to increase task difficulty

### 🧠 Trajectory Processing Module  
- **E2E Agent**: End-to-end question-answering agent implementation
- **Trajectory Generator**: Expert LLM trajectory collection and processing
- **Reasoning Reconstruction**: Trajectory optimization and quality filtering

### 🤖 Agent Core
- **ReAct Framework**: Reflection-Action loop implementation
- **Web Tools**: Integrated search and web access capabilities
- **Memory System**: Support for long-sequence reasoning memory management
- **Wiki Retrieval**: Dedicated Wiki knowledge retrieval tool

### 🛠️ Tools and Configuration
- **Configuration Management**: Flexible YAML configuration system
- **Caching System**: Improve data processing efficiency
- **Logging System**: Complete logging and monitoring
- **Data Tools**: Rich data processing and IO utilities

## Quick Start

### Environment Requirements
- Python 3.8+
- PyTorch 2.0+
- Sufficient GPU memory (8GB+ recommended)

### Install Dependencies
```bash
# Clone the project
git clone https://github.com/your-repo/nanoWebSailor.git
cd nanoWebSailor

# Install dependencies
pip install -r requirements.txt

# Install the project
pip install -e .
```

### Configuration Setup
```bash
# Copy and edit configuration file
cp configs/default_config.yaml configs/config.yaml
# Set your API keys in config.yaml
```

### Generate Training Data

#### 1. Generate Seed Entities
```bash
python scripts/generate_entities.py --num_entities 100 --config config/default_config.yaml --output data/generated_entities 
```

#### 2. Expand Knowledge Graph
```bash
python scripts/expand_graphs.py --entities_file data/generated_entities/entities.json --config config/default_config.yaml --output_dir data/expanded_graphs
```

#### 3. Sample Subgraphs
```bash
python scripts/sample_subgraphs.py --input-dir data/expanded_graphs --config config/default_config.yaml --output-dir data/sampled_subgraphs
```

#### 4. Generate QA Dataset
```bash
python scripts/generate_sailorfog_qa.py --input-dir data/sampled_subgraphs/ --config config/default_config.yaml --output-dir data/generated_qa --num-qa 3
```

#### 5. Visualize Results
```bash
python scripts/visualize_graphs.py --input_path data/path_to_your_graph.json --output_dir data/visualization_results
```

### Trajectory Reconstruction and Optimization
```bash
# Reconstruct generated trajectories to extract concise reasoning chains
python scripts/reconstruct_reasoning.py \
  --input_file data/trajectories/sailorfog_trajectories.jsonl \
  --output_file data/reconstructed_trajectories.jsonl \
  --config configs/default_config.yaml

# Batch reconstruct multiple trajectory files
python scripts/reconstruct_reasoning.py \
  --input_dir data/trajectories/ \
  --output_dir data/reconstructed/ \
  --filter_success_only true
```

### Run Agent to Generate Trajectories

#### Prepare Input Data
First prepare JSONL format question-answer data, each line containing:
```json
{"id": "question_1", "query": "Your question", "golden_answers": ["correct_answer_1", "correct_answer_2"], "source": "dataset_name"}
```

#### Generate Trajectory Data
```bash
# Basic usage - generate single trajectory
python websailor/trajectory/trajectory_generator.py \
  --data_path data/your_questions.jsonl \
  --save_path data/trajectories/output.jsonl \
  --rollout 1 \
  --max_turns 30

# High-quality trajectory generation - multiple sampling for best results
python websailor/trajectory/trajectory_generator.py \
  --data_path data/generated_qa/sailorfog_qa_dataset.json \
  --save_path data/trajectories/sailorfog_trajectories.jsonl \
  --rollout 3 \
  --max_turns 32 \
  --retry_times 2 \
  --max_workers 5

# Test mode - quick validation
python websailor/trajectory/trajectory_generator.py \
  --data_path data/test_questions.jsonl \
  --save_path data/test_results.jsonl \
  --test
```

#### Parallel Processing for Large-scale Data
```bash
# Batch process large datasets
python websailor/trajectory/trajectory_generator.py \
  --data_path data/large_dataset.jsonl \
  --save_path data/trajectories/batch_1.jsonl \
  --start_index 0 \
  --max_workers 10 \
  --rollout 5

# Continue processing from specified position
python websailor/trajectory/trajectory_generator.py \
  --data_path data/large_dataset.jsonl \
  --save_path data/trajectories/batch_2.jsonl \
  --start_index 1000 \
  --max_workers 10
```

#### Output Results Description
Generated trajectory files contain:
- `conversations`: Complete ReAct reasoning process
- `success_rate`: Success rate of correct answers
- `tools`: Tool descriptions used
- `query`: Original question
- `answer`: Standard answer

## Project Structure

```
nanoWebSailor/
├── websailor/           # Core modules
│   ├── data_synthesis/  # Data synthesis
│   │   ├── entity_generator.py      # Entity generator
│   │   ├── graph_builder.py         # Knowledge graph builder
│   │   ├── graph_expander.py        # Graph expander
│   │   ├── qa_generator.py          # QA generator
│   │   ├── subgraph_sampler.py      # Subgraph sampler
│   │   ├── triplet_extractor.py     # Triplet extractor
│   │   └── mock_web_search.py       # Mock web search
│   ├── trajectory/      # Trajectory processing  
│   │   ├── e2e_agent.py             # End-to-end agent
│   │   └── trajectory_generator.py  # Trajectory generator
│   ├── agent/          # Agent core
│   │   ├── react_agent.py           # ReAct agent
│   │   ├── web_tools.py             # Web tools
│   │   └── memory.py                # Memory system
│   └── utils/          # Common utilities
│       ├── config.py                # Configuration management
│       ├── data_utils.py            # Data utilities
│       ├── logger.py                # Logging system
│       ├── cache.py                 # Caching system
│       └── wiki_retrieval_tool.py   # Wiki retrieval
├── scripts/            # Execution scripts
│   ├── generate_entities.py         # Generate entities
│   ├── expand_graphs.py             # Expand graphs
│   ├── sample_subgraphs.py          # Sample subgraphs
│   ├── generate_sailorfog_qa.py     # Generate QA data
│   ├── visualize_graphs.py          # Visualize graphs
│   └── reconstruct_reasoning.py     # Reconstruct reasoning
├── configs/            # Configuration files
│   └── default_config.yaml          # Default configuration
├── data/              # Data directory
├── tests/             # Test files
├── log/               # Log files
├── README.md
├── requirements.txt
└── setup.py
```

## Dataset Description

Main datasets generated by the project:

- **SailorFog-QA**: High-difficulty question-answer dataset containing Level 3 complex reasoning tasks
- **Knowledge Graphs**: Densely interconnected graph structures expanded from Wikipedia
- **Subgraph Samples**: High-quality subgraph data from multi-strategy sampling
- **Expert Trajectories**: Reasoning trajectory data generated by LLMs

## Configuration Description

Main configuration items:

- **API Keys**: Set API keys for OpenAI or other LLM services
- **Model Selection**: Support for GPT-4, Claude, and other models
- **Sampling Strategies**: Configurable BFS, DFS, community detection, and other sampling strategies
- **Generation Parameters**: Adjustable temperature, max tokens, and other generation parameters

## Development and Contribution

### Adding New Features
1. Create new files in the appropriate module
2. Inherit base classes and implement necessary methods
3. Add relevant parameters to configuration files
4. Write unit tests

### Code Standards
- Use Python type annotations
- Follow PEP 8 code style
- Add complete docstrings
- Use loguru for logging

## License

This project is open-sourced under the Apache 2.0 license.

## Citation

If you use this project, please cite the original paper:
```bibtex
@misc{li2025websailornavigatingsuperhumanreasoning,
      title={WebSailor: Navigating Super-human Reasoning for Web Agent}, 
      author={Kuan Li and Zhongwang Zhang and Huifeng Yin and Liwen Zhang and Litu Ou and Jialong Wu and Wenbiao Yin and Baixuan Li and Zhengwei Tao and Xinyu Wang and Weizhou Shen and Junkai Zhang and Dingchu Zhang and Xixi Wu and Yong Jiang and Ming Yan and Pengjun Xie and Fei Huang and Jingren Zhou},
      year={2025},
      eprint={2507.02592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.02592}, 
}
```

## Authors

- **Evan Zuo** - Main Developer
- Email: zuoyifan132@gmail.com

## Changelog

### v0.1.0 (July 2025)
- Initial release
- Implemented core data synthesis pipeline
- Completed ReAct agent framework
- Support for multiple subgraph sampling strategies
- Integrated visualization features 