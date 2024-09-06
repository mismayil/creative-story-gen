#!/bin/bash

experiment=${1:-"run1"}
model=${2:-"claude-3-5-sonnet-20240620"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

# Test experiments
for jsonfile in ${data_dir}/test/eval/*.json
do
    echo "Evaluating ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/test/${experiment} -m ${model} -b 2 -oa -t 0.7 -p 0.95 -g 256
done

# # Pilot experiments
# for jsonfile in ${data_dir}/pilot/eval/*.json
# do
#     echo "Evaluating ${jsonfile}"
#     python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/${experiment} -m ${model} -b 2 -oa
# done