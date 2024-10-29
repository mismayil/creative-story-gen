#!/bin/bash

experiment=${1:-"test"}
model=${2:-"llama-3.1-70b-instruct"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

temperatures=(0 0.3 0.5 0.7 0.9 1)
top_ps=(0.3 0.5 0.7 0.9 0.95 1)

# # Pilot experiments
for jsonfile in ${data_dir}/pilot/eval/*.json
do
    echo "Evaluating ${jsonfile}"
    for t in ${temperatures[@]}
    do
        for p in ${top_ps[@]}
        do
            echo "Temperature: ${t}, Top-p: ${p}"
            python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot/temp${t}_p${p}/${experiment} -m ${model} -b 2 -t ${t} -p ${p} -g 256 -s "\n"
        done
    done
done