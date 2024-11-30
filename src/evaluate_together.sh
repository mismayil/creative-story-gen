#!/bin/bash

experiment=${1:-"default"}
model=${2:-"llama-3.1-405b-instruct"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

# temperatures=(0.7 0.9 1)
# top_ps=(0.7 0.9 0.95 1)
# eval_files=(
#     # ${data_dir}/pilot/eval/pilot_data_eval_default.json
#     # ${data_dir}/pilot/eval/pilot_data_eval_paraphrased.json
#     ${data_dir}/pilot/eval/pilot_data_eval_simple.json
# )

# # # Pilot experiments
# for jsonfile in ${eval_files[@]}
# do
#     echo "Evaluating ${jsonfile}"
#     for t in ${temperatures[@]}
#     do
#         for p in ${top_ps[@]}
#         do
#             echo "Temperature: ${t}, Top-p: ${p}"
#             python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot2/${experiment}/temp${t}_p${p} -m ${model} -b 2 -t ${t} -p ${p} -g 256 -s "\n"
#         done
#     done
# done

models=(
    # "llama-3.2-3b-instruct"
    # "llama-3.1-8b-instruct"
    # "llama-3.1-70b-instruct"
    # "llama-3.1-405b-instruct"
    # "qwen-2.5-coder-32b-instruct"
    # "wizardlm-2-8x22b"
    # "gemma-2-27b"
    # "gemma-2-9b"
    # "dbrx-instruct"
    # "deepseek-llm-chat-67b"
    # "mythomax-l2-13b"
    # "mistral-7b-instruct-v0.3"
    # "mixtral-8x7b-instruct"
    # "mixtral-8x22b-instruct"
    # "nous-hermes-2-mixtral-8x7b-dpo"
    # "qwen-2.5-7b-instruct"
    # "qwen-2.5-72b-instruct"
    # "stripedhyena-nous-7b"
    # "solar-10.7b-instruct-v1.0"
    "gemma-2-2b"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p} -g 256 -s "\n"
done