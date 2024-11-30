#!/bin/bash

experiment=${1:-"default"}
model=${2:-"reka-core"}
data_dir="../experiments/data"
outputs_dir="../experiments/outputs"

models=(
    "reka-edge"
    # "reka-flash"
    # "reka-core"
)

t=0.7
p=0.95
jsonfile=${data_dir}/pilot/eval/pilot_data_eval_default.json

for model in ${models[@]}
do
    echo "Evaluating ${model} on ${jsonfile}"
    python evaluate_lm.py -d ${jsonfile} -o ${outputs_dir}/${model}/pilot3/${experiment}/temp${t}_p${p} -m ${model} -b 4 -t ${t} -p ${p}
done