#!/bin/bash

experiment=${1:-"default"}
model=${2:-"claude-3-5-sonnet-20240620"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

# Test experiments
# for jsonfile in ${data_dir}/test/eval/*.json
# do
#     echo "Evaluating ${jsonfile}"
#     python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/test/${experiment} -m ${model} -b 2 -oa -t 0.7 -p 0.95 -g 256
# done

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
#             python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot2/${experiment}/temp${t}_p${p}/ -m ${model} -b 2 -t ${t} -p ${p} -g 256
#         done
#     done
# done

# t=0.7
# p=0.95
# jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

# echo "Evaluating ${model} on ${jsonfile}"
# python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p} -g 256

# LLM judge experiments
for jsonfile in ${data_dir}/llm_judge/eval/*.json
do
    echo "Evaluating ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/llm_judge/${experiment} -m ${model} -b 8 -t 0.0 -p 1.0 -g 512
done