# Creative Story Generation Project

## Setup
Install dependencies
```sh
pip install -r requirements.txt
```

Install spacy packages
```sh
python -m spacy download en_core_web_sm
```

## Generate reports
First, define the report configuration under `experiments/reports/{experiment_name}/{report_name}` (see [`sample config.json`](experiments/reports/pilot/run1_report1/config.json)).
Then you can use this config along with the experiment results to run `report_metrics.py` such as below:
```sh
python report_metrics.py -r "../experiments/outputs/gpt-4/pilot/**/run1" -o ../experiments/reports/{experiment_name}/${report_name}/gpt-4 -c ../experiments/reports/{experiment_name}/${report_name}/config.json
```

See examples in [report_metrics.sh](src/report_metrics.sh) and check out [report_metrics.py](src/report_metrics.py) for further details.