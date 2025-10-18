#!/bin/bash

cd $(dirname $0) || exit

# GOOGLE_SEARCH_KEY
export GOOGLE_SEARCH_KEY=''
# JINA
export JINA_API_KEY=''
# DASHSCOPE
export DASHSCOPE_API_KEY=''

cd ..

python -m demos.assistant_qwq_chat