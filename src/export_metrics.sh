#!/bin/bash

report_name=${1:-"run1_report1"}

# pilot run 1
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/gpt-4 -o pilot_gpt-4_${report_name}_metrics.csv
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/gemini-1.5-flash -o pilot_gemini-1.5-flash_${report_name}_metrics.csv
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/claude-3-5-sonnet-20240620 -o pilot_claude-3-5-sonnet-20240620_${report_name}_metrics.csv
# python export_metrics.py -i ../experiments/reports/pilot/${report_name}/human -o pilot_human_${report_name}_metrics.csv

python export_metrics.py -i ../experiments/reports/pilot/${report_name}/human -o pilot_human_${report_name}_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot/${report_name}/machine -o pilot_machine_${report_name}_metrics.csv