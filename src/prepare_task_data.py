import argparse
import pathlib
from tqdm import tqdm
import random

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

def prepare_llm_judge_data(datapath):
    data = read_json(datapath)
    task_data = []
    data_by_item_by_author = {}
    
    for sample_id, sample in tqdm(data.items(), total=len(data), desc=f"Preparing llm-judge data from {datapath}"):
        item_id = sample["item_id"]
        author = sample["author"]

        if item_id not in data_by_item_by_author:
            data_by_item_by_author[item_id] = {}
        
        if author not in data_by_item_by_author[item_id]:
            data_by_item_by_author[item_id][author] = []
        
        data_by_item_by_author[item_id][author].append(sample)
    
    item_ids = list(data_by_item_by_author.keys())

    for item_id in item_ids:
        ai_queue = data_by_item_by_author[item_id].get("AI", [])
        human_queue = data_by_item_by_author[item_id].get("human", [])
        
        while True:
            item_samples = []

            for _ in range(3):
                if ai_queue:
                    item_samples.append(ai_queue.pop())
                if human_queue:
                    item_samples.append(human_queue.pop())
            
            item_samples = random.sample(item_samples, len(item_samples))

            if item_samples:
                task_data.append({
                    "item_id": item_id,
                    "samples": item_samples
                })
            else:
                break

    return task_data

TASK_MAP = {
    "summarization": prepare_summarization_data,
    "llm-judge": prepare_llm_judge_data
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