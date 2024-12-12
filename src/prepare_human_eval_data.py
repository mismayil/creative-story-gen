from utils import read_json, write_json, find_files
import argparse
import pathlib

def prepare_human_eval_data(datapath):
    contents = read_json(datapath)
    eval_data = {}

    if "data" in contents:
        results = contents["data"]
        model = contents["metadata"]["model"]
        
        for result in results:
            if result["output"]:
                eval_data[result["result_id"]] = {
                    "id": result["result_id"],
                    "text": result["output"],
                    "item_id": result["id"],
                    "author": "human" if model == "human" else "AI"
                }
        
    return eval_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-paths", type=str, nargs="+", help="Path to input data file(s) in json", required=True)
    parser.add_argument("-o", "--output-path", type=str, help="Output file path", required=True)

    args = parser.parse_args()
    files_to_process = []

    for input_path in args.input_paths:
        if pathlib.Path(input_path).is_file():
            files_to_process.append(input_path)
        else:
            files_to_process.extend(find_files(input_path))

    eval_data = {}

    for file_path in files_to_process:
        eval_data = {**eval_data, **prepare_human_eval_data(file_path)}

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(eval_data, output_path.with_suffix(".json"))

if __name__ == "__main__":
    main()