import argparse
import pathlib
import pandas
from tqdm import tqdm

from utils import read_json, find_files

def get_participant_id(sample):
    return sample["metadata"].get("participant_id", sample["result_id"].rsplit("_", 1)[0])

def export_metrics(results):
    export_data = []

    for sample in tqdm(results["data"], total=len(results["data"]), desc="Exporting metrics"):
        if "metrics" in sample:
            id_value = None

            if sample["metadata"]["model"] == "human":
                id_value = f"human_{get_participant_id(sample)}"
            else:
                id_value = f"{sample['metadata']['model']}_t{sample['metadata']['model_args']['temperature']}_p{sample['metadata']['model_args']['top_p']}"
            
            export_data.append({
                "result_id": sample["result_id"],
                "model_id": id_value,
                "item_id": sample["id"],
                **{f"metric_{metric_name}": metric_value for metric_name, metric_value in sample["metrics"].items() if not isinstance(metric_value, dict)}
            })
    
    return export_data

def export_global_metrics(results):
    export_data = []
    sample = results["data"][0]
    model_id = sample["metadata"]["model"]
    
    export_data.append({
        "group_id": results["metadata"]["group_id"],
        "model_id": model_id,
        **{f"metric_{metric_name}": metric_value for metric_name, metric_value in results["metrics"].items() if not isinstance(metric_value, dict)}
    })

    return export_data

def main():
    parser = argparse.ArgumentParser(description="Export metrics to a new format")
    parser.add_argument("-i", "--input-dir", type=str, help="Input directory")
    parser.add_argument("-f", "--output-format", type=str, choices=["csv"], default="csv", help="Output format")
    parser.add_argument("-o", "--output-file", type=str, help="Output filename", default="metrics.csv")

    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    files = find_files(input_dir, "json")

    export_data = []
    export_global_data = []

    for file in files:
        results = read_json(file)
        
        if "data" in results:
            if args.output_format == "csv":
                export_data.extend(export_metrics(results))
                export_global_data.extend(export_global_metrics(results))
            else:
                raise ValueError(f"Unknown output format: {args.output_format}")

    if args.output_format == "csv":
        export_df = pandas.DataFrame(export_data)
        export_df.to_csv(input_dir / args.output_file, index=False)
        export_global_df = pandas.DataFrame(export_global_data)
        export_global_df.to_csv(input_dir / args.output_file.replace(".csv", "_global.csv"), index=False)

if __name__ == '__main__':
    main()