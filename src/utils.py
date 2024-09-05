from string import Formatter
import json
import tiktoken
import uuid
import os
import glob

MODEL_COSTS = {
    "gpt-3.5-turbo": {'input': 0.0000015, 'output': 0.000002},
    "gpt-4": {'input': 0.00003, 'output': 0.00006},
    "gpt-4-0125-preview": {'input': 0.00001, 'output': 0.00003},
    "gpt-4o-2024-08-06": {'input': 2.5e-6, 'output': 10e-6},
    "text-davinci-003": {'input': 0.00002, 'output': 0.00002},
    "gemini-1.5-flash": {'input': 3.5e-7, 'output': 1.05e-6},
    "gemini-1.5-pro": {'input': 3.5e-6, 'output': 10.5e-6},
}

MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "gpt-4o": "o200k_base"
}

def num_tokens_from_string(text, model):
    if model not in MODEL_ENCODINGS:
        return 0
    encoding_name = MODEL_ENCODINGS[model]
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def write_json(data, path, ensure_ascii=True, indent=4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

def generate_unique_id():
    return str(uuid.uuid4()).split("-")[-1]

def find_files(directory, extension="json"):
    return glob.glob(f"{directory}/**/*.{extension}", recursive=True)

def concatenate(lists):
    return [item for sublist in lists for item in sublist]

def levenshtein_distance(s, t):
    m = len(s)
    n = len(t)
    d = [[0] * (n + 1) for i in range(m + 1)]  

    for i in range(1, m + 1):
        d[i][0] = i

    for j in range(1, n + 1):
        d[0][j] = j
    
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                          d[i][j - 1] + 1,      # insertion
                          d[i - 1][j - 1] + cost) # substitution   

    return d[m][n]

def batched(lst, size=4):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def compute_usage(sample, model):
    if model not in MODEL_COSTS:
        return None, None

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    usage = sample.get("usage")

    if not usage and "prompt" in sample and "model_output" in sample:
        prompt_tokens = num_tokens_from_string(sample["prompt"], model)
        completion_tokens = num_tokens_from_string(sample["model_output"], model)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    input_cost = usage["prompt_tokens"] * MODEL_COSTS[model]["input"]
    output_cost = usage["completion_tokens"] * MODEL_COSTS[model]["output"]

    return usage, {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost
    }

def get_template_keys(template):
    return [i[1] for i in Formatter().parse(template) if i[1] is not None]