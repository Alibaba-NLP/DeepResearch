#!/bin/bash

MODEL_PATH=$1
sglang.launch_server --model-path $MODEL_PATH --host 0.0.0.0 --tp 4 --port 8004