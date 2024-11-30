#!/bin/bash

experiment=${1:-"default"}
model=${2:-"glm-4-0520"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

models=(
    "glm-4-0520"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p}
done