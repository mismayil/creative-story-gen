from string import Formatter
import json
import tiktoken
import uuid
import os
import glob
import pandas as pd

MODEL_COSTS = {
    "gpt-3.5-turbo": {'input': 0.0000015, 'output': 0.000002},
    "gpt-4": {'input': 30e-6, 'output': 60e-6},
    "gpt-4o": {'input': 2.5e-6, 'output': 10e-6},
    "gpt-4-0125-preview": {'input': 10e-6, 'output': 30e-6},
    "gpt-4o-2024-08-06": {'input': 2.5e-6, 'output': 10e-6},
    "text-davinci-003": {'input': 0.00002, 'output': 0.00002},
    "gemini-1.5-flash": {'input': 3.5e-7, 'output': 1.05e-6},
    "gemini-1.5-pro": {'input': 3.5e-6, 'output': 10.5e-6},
    "claude-3-5-sonnet-20240620": {'input': 3e-6, 'output': 15e-6},
    "claude-3-opus-20240229": {'input': 15e-6, 'output': 75e-6},
    "claude-3-sonnet-20240229": {'input': 3e-6, 'output': 15e-6},
    "claude-3-haiku-20240307": {'input': 0.25e-6, 'output': 1.25e-6}
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

def compute_usage(sample, model, 
                  input_attrs=["system_prompt", "user_prompt"], 
                  output_attrs=["output"],
                  max_input_tokens=None, max_output_tokens=None):
    if model not in MODEL_COSTS:
        return None, None

    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }

    usage = sample.get("usage")

    if not usage:
        input_tokens = 0
        output_tokens = 0

        if max_input_tokens:
            input_tokens = max_input_tokens
        else:
            for attr in input_attrs:
                if attr in sample:
                    input_tokens += num_tokens_from_string(sample[attr], model)

        if max_output_tokens:
            output_tokens = max_output_tokens        
        else:
            for attr in output_attrs:
                if attr in sample:
                    output_tokens += num_tokens_from_string(sample[attr], model)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    input_cost = usage["input_tokens"] * MODEL_COSTS[model]["input"]
    output_cost = usage["output_tokens"] * MODEL_COSTS[model]["output"]

    return usage, {
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost
    }

def get_template_keys(template):
    return [i[1] for i in Formatter().parse(template) if i[1] is not None]

def is_immutable(obj):
    return isinstance(obj, (str, int, float, bool, tuple, type(None)))

def cache(cache_dict):
    def decorator_cache(func):
        def wrapper(*args, **kwargs):
            if all(is_immutable(arg) for arg in args) and all(is_immutable(val) for val in kwargs.values()):
                key = (args, frozenset(kwargs.items()))
                if key in cache_dict:
                    return cache_dict[key]
                result = func(*args, **kwargs)
                cache_dict[key] = result
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_cache

def concat_dfs(df_lst):
    shared_columns = None

    for df in df_lst:
        if shared_columns is None:
            shared_columns = set(df.columns)
        else:
            shared_columns.intersection_update(df.columns)
    
    shared_columns = list(shared_columns)
    return pd.concat([df[shared_columns] for df in df_lst])