#!/bin/bash

report_name=${1:-"run1_report1"}

# pilot run 1
python report_metrics.py -r "../experiments/outputs/gpt-4/pilot/**/run1" -o ../experiments/reports/pilot/${report_name}/gpt-4 -c ../experiments/reports/pilot/${report_name}/config.json
python report_metrics.py -r "../experiments/outputs/gemini-1.5-flash/pilot/**/run1" -o ../experiments/reports/pilot/${report_name}/gemini-1.5-flash -c ../experiments/reports/pilot/${report_name}/config.json
python report_metrics.py -r "../experiments/outputs/claude-3-5-sonnet-20240620/pilot/**/run1" -o ../experiments/reports/pilot/${report_name}/claude-3-5-sonnet-20240620 -c ../experiments/reports/pilot/${report_name}/config.json
python report_metrics.py -r "../experiments/outputs/human/pilot/run1" -o ../experiments/reports/pilot/${report_name}/human -c ../experiments/reports/pilot/${report_name}/config.json