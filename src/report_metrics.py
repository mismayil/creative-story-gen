import argparse
from tqdm import tqdm
import pathlib

from utils import read_json, write_json, find_files, compute_usage


def compute_metrics(results, report_usage=True):
    metrics = {}

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    cost = {
        "input": 0,
        "output": 0,
        "total": 0
    }

    template = results["data"][0]["template"]
    result_map = {}
    ref_response_attr = "reference"
    model_response_attr = "model_output"

    for result in results["data"]:
        result_map[result["id"]] = result

        if model_response_attr in result:
            if report_usage:
                sample_usage, sample_cost = compute_usage(result, results["metadata"]["model"])

                if sample_usage:
                    usage["prompt_tokens"] += sample_usage["prompt_tokens"]
                    usage["completion_tokens"] += sample_usage["completion_tokens"]
                    usage["total_tokens"] += sample_usage["total_tokens"]

                if sample_cost:
                    cost["input"] += sample_cost["input"]
                    cost["output"] += sample_cost["output"]
                    cost["total"] += sample_cost["total"]

    if report_usage:
        metrics["usage"] = usage
        metrics["cost"] = cost

    metrics["num_samples"] = len(results["data"])

    return metrics

def report_metrics(results_files, report_usage=True):
    for results_file in tqdm(results_files, total=len(results_files), desc="Reporting metrics"):
        results = read_json(results_file)
        
        try:
            if "data" in results:
                metrics = compute_metrics(results, report_usage=report_usage)
                results["metrics"] = metrics
                write_json(results, results_file)
        except Exception as e:
            print(results_file)
            raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results-path", type=str, help="Path to evaluation results file in json or directory", required=True)
    parser.add_argument("-u", "--report-usage", action="store_true", help="Report usage metrics", default=True)

    args = parser.parse_args()

    files_to_process = []

    results_path = pathlib.Path(args.results_path)

    if results_path.is_file():
        files_to_process.append(args.results_path)
    else:
        files_to_process.extend(find_files(args.results_path))

    report_metrics(files_to_process, args.report_usage)

if __name__ == "__main__":
    main()