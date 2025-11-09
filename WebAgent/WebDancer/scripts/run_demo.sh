#!/bin/bash

cd $(dirname $0) || exit

source ../.env

cd ..

python -m demos.assistant_qwq_chat