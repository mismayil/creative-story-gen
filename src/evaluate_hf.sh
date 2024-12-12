#!/bin/bash

experiment=${1:-"default"}
model=${2:-"yi-1.5-9b-chat"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

models=(
    # "yi-1.5-9b-chat"
    # "stablelm-2-12b-chat"
    # "stablelm-zephyr-3b"
    # "olmo-2-7b"
    # "olmo-2-13b"
    # "mpt-7b-8k-chat"
    # "llama-3.2-1b-instruct"
    # "deepseek-llm-7b-chat"
    # "baichuan2-7b-chat"
    # "zamba2-2.7b-instruct"
    # "zamba2-1.2b-instruct"
    # "granite-3.0-2b-instruct"
    "mpt-30b-chat"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} \
                          -c /mnt/scratch/home/ismayilz/.cache \
                          -m ${model} -b 4 -t ${t} -p ${p} -g 256 -s "\n"
done