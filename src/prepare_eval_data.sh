#!/bin/bash

experiment=${1:-"pilot"}
data_dir=${2:-"../data"}
output_dir=${3:-"../experiments/data"}

# Pilot experiments
python prepare_eval_data.py -d ${data_dir}/pilot_data.json -o ${output_dir}/pilot/eval -t pilot