#!/bin/bash

experiment=${1:-"default"}
model=${2:-"yi-1.5-9b-chat"}
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
    "yi-1.5-9b-chat"
    "stablelm-2-12b-chat"
    "stablelm-zephyr-3b"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p} -g 256 -s "\n"
done