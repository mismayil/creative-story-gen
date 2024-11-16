#!/bin/bash

report_name=${1:-"run1_report1"}

# pilot run 1
python plot_metrics.py -i ../experiments/reports/pilot2/${report_name}/gpt-4/pilot2_gpt-4_${report_name}_metrics.csv \
                          ../experiments/reports/pilot2/${report_name}/gemini-1.5-flash/pilot2_gemini-1.5-flash_${report_name}_metrics.csv \
                          ../experiments/reports/pilot2/${report_name}/claude-3-5-sonnet-20240620/pilot2_claude-3-5-sonnet-20240620_${report_name}_metrics.csv \
                          ../experiments/reports/pilot2/${report_name}/llama-3.1-405b-instruct/pilot2_llama-3.1-405b-instruct_${report_name}_metrics.csv \
                          ../experiments/reports/pilot2/${report_name}/human/pilot2_human_${report_name}_metrics.csv \
                        -o ../experiments/reports/pilot2/${report_name}/figures -f pdf

python plot_metrics.py -i ../experiments/reports/pilot2/${report_name}/gpt-4/pilot2_gpt-4_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot2/${report_name}/gemini-1.5-flash/pilot2_gemini-1.5-flash_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot2/${report_name}/claude-3-5-sonnet-20240620/pilot2_claude-3-5-sonnet-20240620_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot2/${report_name}/llama-3.1-405b-instruct/pilot2_llama-3.1-405b-instruct_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot2/${report_name}/human/pilot2_human_${report_name}_metrics_global.csv \
                        -o ../experiments/reports/pilot2/${report_name}/figures -p n_gram_diversity raw_surprises -f pdf

# machine vs human
# python plot_metrics.py -i ../experiments/reports/pilot/${report_name}/human/pilot_human_${report_name}_metrics.csv \
#                           ../experiments/reports/pilot/${report_name}/machine/pilot_machine_${report_name}_metrics.csv \
#                         -o ../experiments/reports/pilot/${report_name}/figures -f pdf

# python plot_metrics.py -i ../experiments/reports/pilot/${report_name}/human/pilot_human_${report_name}_metrics_global.csv \
#                           ../experiments/reports/pilot/${report_name}/machine/pilot_machine_${report_name}_metrics_global.csv \
#                         -o ../experiments/reports/pilot/${report_name}/figures -p n_gram_diversity raw_surprises -f pdf