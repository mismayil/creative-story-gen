#!/bin/bash

experiment=${1:-"default"}
model=${2:-"nemotron-4-340b-instruct"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

echo "Evaluating ${model} on ${jsonfile}"
python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 1 -t ${t} -p ${p} -g 256 -s "\n"