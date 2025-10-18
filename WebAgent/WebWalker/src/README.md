# RAG_System

## Environment Setup

If you want to run the RAG-System on WebWalkerQA, you need to export the API keys as environment variables. You can run the following command to export the API keys:

```bash
export OPENAI_API_KEY=YOUR_OPEN_API_KEY
export OPENAI_BASE_URL=YOUR_OPENAI_API_BASE_URL
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
export GEMINI_BASE_URL=YOUR_GEMINI_API_BASE_URL
export ARK_API_KEY=YOUR_ARK_API_KEY
export ARK_MODEL=YOUR_ARK_MODEL
export MOONSHOT_API_KEY=YOUR_MOONSHOT_API_KEY
export BAIDU_API_KEY=YOUR_BAIDU_API
export BAIDU_SECRET_KEY=YOUR_BAIDU_SECRET_KEY
```

```
usage: rag_system.py [-h] [--api_name API_NAME] [--output_path OUTPUT_PATH]

Run different API models.

options:
  -h, --help            show this help message and exit
  --api_name API_NAME   Name of the API to run.
  --output_path OUTPUT_PATH
                        Path to the output file. If api_name is 'all', this should be a directory. If api_name is others,
                        this should be a file.
```

If you want to run all the APIs, you can run the following command:

```bash
python rag_system.py --api_name all --output_path results
```

# Evaluation

## Environment Setup

The evaluation is based on the GPT-4. You need to export the API keys as environment variables. You can run the following command to export the API keys:

```bash
export OPENAI_API_KEY=YOUR_OPEN_API_KEY
export OPENAI_BASE_URL=YOUR_OPENAI_API_BASE_URL
```

usage: evaluate.py [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]

```
options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Input prediction result path
  --output_path OUTPUT_PATH
                        Evaluation output path
```

```bash
python evaluate.py --input_path [INPUT_PATH]--output_path [OUTPUT_PATH]
```
