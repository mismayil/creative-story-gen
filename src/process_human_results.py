import argparse
import pandas as pd
import pathlib
from tqdm import tqdm

from utils import write_json, read_json

def main():
    parser = argparse.ArgumentParser(description="Process human results")
    parser.add_argument("-r", "--results-path", type=str, help="Path to the human results file in csv")
    parser.add_argument("-c", "--config-path", type=str, help="Path to the config file in json")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment name")
    
    args = parser.parse_args()

    results_path = pathlib.Path(args.results_path)

    human_results = pd.read_csv(results_path)
    config = read_json(args.config_path)

    metadata = {
        "source": args.results_path,
        "num_participants": len(human_results)
    }
    write_json({"metadata": metadata, "data": human_results.to_dict(orient="records")}, results_path.with_suffix(".json"))

    for _, row in tqdm(human_results.iterrows(), total=len(human_results), desc="Processing human results"):
        if row["Include"] == 0:
            continue
        human_id = row["ResponseId"]
        results = []
        source_data = read_json(config["source_data"])
        source_data_map = {sample["id"]: sample for sample in source_data["data"]}

        for q_index in range(1, 5):
            item_id = config["id_map"][f"Q{q_index}"]
            source_sample = source_data_map[item_id]
            results.append({
                **source_sample,
                "output": row[f"Q{q_index}"],
                "result_id": f"{human_id}_Q{q_index}"
            })
        
        metadata = {
            "source": args.results_path,
            "num_questions": 4,
            "participant_id": human_id,
            "model": "human"
        }

        output_path = results_path.parent / args.experiment / f"{human_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json({"metadata": metadata, "data": results}, output_path)

if __name__ == '__main__':
    main()