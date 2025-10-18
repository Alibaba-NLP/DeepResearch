For hle,

1. Export what you need

```
export API_KEY=Your api key
export BASE_URL=Your base url
```

2. Run this command

```
python eval_hle_old_react.py --input_fp your_input_folder --model_path your_qwen_model_path
```


For other benchmarks,

1. Export what you need
```
export OPENAI_API_KEY=Your openai api key
export OPENAI_API_BASE=Your openai api base
export API_KEY=Your api key
export BASE_URL=Your base url
export Qwen2_5_7B_PATH=Your qwen model path
```

2. Run this command

```
python evaluate_all_official.py --input_fp your_input_folder --dataset your_evaluated_dataset 


