import argparse
import pathlib
from tqdm import tqdm

from utils import read_json, find_files, compute_usage, MODEL_ENCODINGS

def estimate_cost(datapath, default_model="gpt-4", input_attrs=["system_prompt", "user_prompt"], 
                  output_attrs=["output"], max_input_tokens=None, max_output_tokens=None):
    data = read_json(datapath)
    input_cost = 0
    output_cost = 0
    
    if "data" in data:
        metadata = data.get("metadata", {})
        model = metadata.get("model", default_model)
        if model not in MODEL_ENCODINGS:
            model = default_model
        for sample in tqdm(data["data"], total=len(data["data"]), desc=f"Estimating cost for {datapath}"):
            usage, cost = compute_usage(sample, model=model, input_attrs=input_attrs, output_attrs=output_attrs, 
                                  max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens)
            input_cost += cost["input"]
            output_cost += cost["output"]
    
    return input_cost, output_cost

def main():
    parser = argparse.ArgumentParser(description="Estimate cost of using API models")
    parser.add_argument("-i", "--input-paths", type=str, nargs="+", help="Path to input data file(s) in json", required=True)
    parser.add_argument("-m", "--default-model", type=str, default="gpt-4", help="Default model name to use if one not found in the metadata of the input data")
    parser.add_argument("-ia", "--input-attrs", type=str, nargs="+", default=["system_prompt", "user_prompt"], help="Input attribute(s)")
    parser.add_argument("-oa", "--output-attrs", type=str, nargs="+", default=["output"], help="Output attribute(s)")
    parser.add_argument("-mit", "--max-input-tokens", type=int, default=None, help="Maximum input tokens")
    parser.add_argument("-mot", "--max-output-tokens", type=int, default=None, help="Maximum output tokens")

    args = parser.parse_args()
    files_to_process = []

    for input_path in args.input_paths:
        if pathlib.Path(input_path).is_file():
            files_to_process.append(input_path)
        else:
            files_to_process.extend(find_files(input_path))

    total_input_cost = 0
    total_output_cost = 0

    for file_path in files_to_process:
        input_cost, output_cost = estimate_cost(file_path, default_model=args.default_model, input_attrs=args.input_attrs, output_attrs=args.output_attrs, 
                                                max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens)
        print("--------------------------------")
        print(f"File: {file_path}")
        print(f"\tInput cost: ${input_cost:.2f}")
        print(f"\tOutput cost: ${output_cost:.2f}")
        print(f"\tTotal cost: ${input_cost + output_cost:.2f}")

        total_input_cost += input_cost
        total_output_cost += output_cost
    
    print("================================")
    print(f"Total input cost: ${total_input_cost:.2f}")
    print(f"Total output cost: ${total_output_cost:.2f}")
    print(f"Total cost: ${total_input_cost + total_output_cost:.2f}")

if __name__ == "__main__":
    main()