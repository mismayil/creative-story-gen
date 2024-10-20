import argparse
import pathlib
from tqdm import tqdm

from utils import read_json, write_json, find_files

def prepare_summarization_data(datapath):
    data = read_json(datapath)
    task_data = []

    for sample in tqdm(data["data"], total=len(data["data"]), desc=f"Preparing summarization data from {datapath}"):
        task_data.append({
            "id": sample["result_id"],
            "item_id": sample["id"],
            "items": sample["items"],
            "semantic_distance": sample["semantic_distance"],
            "story": sample["output"]
        })
    
    return task_data

TASK_MAP = {
    "summarization": prepare_summarization_data
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-paths", type=str, nargs="+", help="Path to input data file(s) in json", required=True)
    parser.add_argument("-t", "--task", type=str, default="summarization", help="Task name")
    parser.add_argument("-s", "--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("-o", "--output-path", type=str, help="Output file path", required=True)

    args = parser.parse_args()
    files_to_process = []

    for input_path in args.input_paths:
        if pathlib.Path(input_path).is_file():
            files_to_process.append(input_path)
        else:
            files_to_process.extend(find_files(input_path))

    task_data = []

    for file_path in files_to_process:
        task_data.extend(TASK_MAP[args.task](file_path))

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "source": args.input_paths,
            "task": args.task,
            "size": len(task_data)
        },
        "data": task_data
    }

    write_json(output_data, output_path.with_suffix(".json"))

if __name__ == "__main__":
    main()