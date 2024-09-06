#!/bin/bash

experiment=${1:-"test"}
data_dir=${2:-"../data"}
output_dir=${3:-"../experiments/data"}

# Test experiments
python prepare_eval_data.py -d ${data_dir}/test_data.json -o ${output_dir}/test/eval -t test

# Pilot experiments
# python prepare_eval_data.py -d ${data_dir}/pilot_data.json -o ${output_dir}/pilot/eval -t default