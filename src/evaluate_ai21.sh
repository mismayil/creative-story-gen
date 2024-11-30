#!/bin/bash

experiment=${1:-"default"}
model=${2:-"jamba-1.5-mini"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

models=(
    "jamba-1.5-mini"
    "jamba-1.5-large"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p} -g 256 -s "\n"
done