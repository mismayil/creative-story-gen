#!/bin/bash

experiment=${1:-"test"}
model=${2:-"gpt-4"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

# Test experiments
# for jsonfile in ${data_dir}/test/eval/*.json
# do
#     echo "Evaluating ${jsonfile}"
#     python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/test/${experiment} -m ${model} -b 2 -oa -t 0.7 -p 0.95
# done

# temperatures=(0 0.3 0.5 0.7 0.9 1)
# top_ps=(0.3 0.5 0.7 0.9 0.95 1)

# # # Pilot experiments
# for jsonfile in ${data_dir}/pilot/eval/*.json
# do
#     echo "Evaluating ${jsonfile}"
#     for t in ${temperatures[@]}
#     do
#         for p in ${top_ps[@]}
#         do
#             echo "Temperature: ${t}, Top-p: ${p}"
#             python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot/temp${t}_p${p}/${experiment} -m ${model} -b 2 -oa -t ${t} -p ${p}
#         done
#     done
# done

# Summary experiments
for jsonfile in ${data_dir}/summary/eval/*.json
do
    echo "Evaluating ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/summary/${experiment} -m ${model} -b 8 -t 0.7 -p 0.95
done