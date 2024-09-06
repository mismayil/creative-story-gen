#!/bin/bash

experiment=${1:-"test"}
model=${2:-"gpt-4"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

# Test experiments
for jsonfile in ${data_dir}/test/eval/*.json
do
    echo "Evaluating ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/test/${experiment} -m ${model} -b 2 -oa -t 0.7 -p 0.95
done

# # Pilot experiments
# for jsonfile in ${data_dir}/pilot/eval/*.json
# do
#     echo "Evaluating ${jsonfile}"
#     python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/${experiment} -m ${model} -b 2 -oa
# done