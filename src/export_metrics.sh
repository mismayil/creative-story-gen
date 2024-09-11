#!/bin/bash

# pilot run 1
python export_metrics.py -i ../experiments/reports/pilot/run1_report1/gpt-4 -o pilot_gpt-4_run1_report1_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot/run1_report1/gemini-1.5-flash -o pilot_gemini-1.5-flash_run1_report1_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot/run1_report1/claude-3-5-sonnet-20240620 -o pilot_claude-3-5-sonnet-20240620_run1_report1_metrics.csv
python export_metrics.py -i ../experiments/reports/pilot/run1_report1/human -o pilot_human_run1_report1_metrics.csv