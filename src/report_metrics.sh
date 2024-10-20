#!/bin/bash

report_name=${1:-"run1_report1"}

# # pilot run 1
python report_metrics.py -r "../experiments/outputs/gpt-4/pilot/**/run1" \
                            "../experiments/outputs/gemini-1.5-flash/pilot/**/run1" \
                            "../experiments/outputs/claude-3-5-sonnet-20240620/pilot/**/run1" \
                            "../experiments/outputs/human/pilot/run1" \
                            -o ../experiments/reports/pilot/${report_name} \
                            -c ../experiments/reports/pilot/${report_name}/config.json

# group by sentience
# temperatures=(0.5 0.7 0.9 1)
# top_ps=(0.7 0.9 0.95 1)
# models=("gpt-4" "gemini-1.5-flash" "claude-3-5-sonnet-20240620")
# result_paths=("../experiments/outputs/human/pilot/run1")

# for model in ${models[@]}; do
#     for temperature in ${temperatures[@]}; do
#         for top_p in ${top_ps[@]}; do
#             result_paths+=("../experiments/outputs/${model}/pilot/temp${temperature}_p${top_p}/run1")
#         done
#     done
# done

# python report_metrics.py -r ${result_paths[@]} \
#                             -o ../experiments/reports/pilot/${report_name} \
#                             -c ../experiments/reports/pilot/${report_name}/config.json
