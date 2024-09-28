#!/bin/bash

report_name=${1:-"run1_report1"}

# pilot run 1
# python plot_metrics.py -i ../experiments/reports/pilot/${report_name}/gpt-4/pilot_gpt-4_${report_name}_metrics.csv \
#                           ../experiments/reports/pilot/${report_name}/gemini-1.5-flash/pilot_gemini-1.5-flash_${report_name}_metrics.csv \
#                           ../experiments/reports/pilot/${report_name}/claude-3-5-sonnet-20240620/pilot_claude-3-5-sonnet-20240620_${report_name}_metrics.csv \
#                           ../experiments/reports/pilot/${report_name}/human/pilot_human_${report_name}_metrics.csv \
#                         -o ../experiments/reports/pilot/${report_name}/figures

python plot_metrics.py -i ../experiments/reports/pilot/${report_name}/gpt-4/pilot_gpt-4_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot/${report_name}/gemini-1.5-flash/pilot_gemini-1.5-flash_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot/${report_name}/claude-3-5-sonnet-20240620/pilot_claude-3-5-sonnet-20240620_${report_name}_metrics_global.csv \
                          ../experiments/reports/pilot/${report_name}/human/pilot_human_${report_name}_metrics_global.csv \
                        -o ../experiments/reports/pilot/${report_name}/figures -p n_gram_diversity