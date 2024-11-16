#!/bin/bash

report_name=${1:-"run1_report1"}

# pilot run 1
python export_metrics.py -i ../experiments/reports/pilot2/${report_name}/gpt-4 -o pilot2_gpt-4_${report_name}_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot2/${report_name}/gemini-1.5-flash -o pilot2_gemini-1.5-flash_${report_name}_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot2/${report_name}/claude-3-5-sonnet-20240620 -o pilot2_claude-3-5-sonnet-20240620_${report_name}_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot2/${report_name}/llama-3.1-405b-instruct -o pilot2_llama-3.1-405b-instruct_${report_name}_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot2/${report_name}/human -o pilot2_human_${report_name}_metrics.csv

# machine vs human
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/human -o pilot_human_${report_name}_metrics.csv
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/machine -o pilot_machine_${report_name}_metrics.csv