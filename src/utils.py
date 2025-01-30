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
    "claude-3-5-haiku-20241022": {"input": 1e-6, "output": 5e-6},
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

# claude, gpt and gemini model sizes based on https://arxiv.org/abs/2412.19260 and https://lifearchitect.substack.com/p/the-memo-special-edition-claude-3
MODEL_SIZE_MAP = {
    "llama-3.2-3b-instruct": 3e9,
    "llama-3.1-8b-instruct": 8e9,
    "llama-3.1-70b-instruct": 70e9,
    "llama-3.1-405b-instruct": 405e9,
    "qwen-2.5-coder-32b-instruct": 32e9,
    "wizardlm-2-8x22b": 22e9,
    "gemma-2-27b": 27e9,
    "gemma-2-9b": 9e9,
    "gemma-2b": 2e9,
    "deepseek-llm-chat-67b": 67e9,
    "mythomax-l2-13b": 13e9,
    "mistral-7b-instruct-v0.3": 7e9,
    "mixtral-8x7b-instruct": 7e9,
    "mixtral-8x22b-instruct": 22e9,
    "nous-hermes-2-mixtral-8x7b-dpo": 7e9,
    "qwen-2.5-7b-instruct": 7e9,
    "qwen-2.5-72b-instruct": 72e9,
    "stripedhyena-nous-7b": 7e9,
    "solar-10.7b-instruct-v1.0": 10.7e9,
    "nemotron-4-340b-instruct": 340e9,
    "yi-large": 34e9,
    "granite-34b-code-instruct": 34e9,
    "granite-8b-code-instruct": 8e9,
    "mistral-nemo-12b-instruct": 12e9,
    "baichuan2-13b-chat": 13e9,
    "nemotron-mini-4b-instruct": 4e9,
    "zamba2-7b-instruct": 7e9,
    "granite-3.0-8b-instruct": 8e9,
    "dbrx-instruct": 132e9,
    "gemma-2-2b-it": 2e9,
    "yi-1.5-34b-chat": 34e9,
    "yi-1.5-9b-chat": 9e9,
    "stablelm-2-12b-chat": 12e9,
    "stablelm-zephyr-3b": 3e9,
    "olmo-2-7b": 7e9,
    "olmo-2-13b": 13e9,
    "persimmon-8b-chat": 8e9,
    "mpt-7b-8k-chat": 7e9,
    "mpt-30b-chat": 30e9,
    "llama-3.2-1b-instruct": 1e9,
    "deepseek-llm-7b-chat": 7e9,
    "baichuan2-7b-chat": 7e9,
    "zamba2-2.7b-instruct": 2.7e9,
    "zamba2-1.2b-instruct": 1.2e9,
    "granite-3.0-2b-instruct": 2e9,
    "gpt-3.5-turbo": 175e9, 
    "grok-beta": 314e9,
    "reka-core": 67e9,
    "reka-edge": 7e9,
    "reka-flash": 21e9,
    "glm-4-0520": 130e9,
    "jamba-1.5-mini": 12e9,
    "jamba-1.5-large": 94e9,
    "Phi-3-mini-4k-instruct": 3.8e9,
    "Phi-3-small-8k-instruct": 7e9,
    "Phi-3-medium-4k-instruct": 14e9,
    "Phi-3.5-MoE-instruct": 6.6e9,
    "command-r-plus": 104e9,
    "c4ai-aya-expanse-8b": 8e9,
    "c4ai-aya-expanse-32b": 32e9,
    "mistral-large-latest": 123e9,
    "ministral-3b-latest": 3e9,
    "ministral-8b-latest": 8e9,
    "mistral-small-latest": 22e9,
    "lfm-40b": 40e9,
    "claude-3-5-haiku-20241022": 20e9,
    "claude-3-5-sonnet-20240620": 175e9,
    "claude-3-opus-20240229": 500e9,
    "gpt-4": 500e9,
    "gpt-4o": 200e9,
    "gemini-1.5-flash": 500e9,
    "gemini-1.5-pro": 500e9
}

def get_model_family(model_name):
    model_name = model_name.lower()
    if model_name.startswith("human"):
        return "Human"
    if "mixtral" in model_name or "MoE" in model_name:
        return "MoE"
    if model_name.startswith("jamba") or model_name.startswith("zamba"):
        return "SSM"
    if model_name.startswith("gpt"):
        return "GPT"
    if model_name.startswith("claude"):
        return "Claude"
    if model_name.startswith("gemini"):
        return "Gemini"
    if model_name.startswith("llama"):
        return "Llama"
    if model_name.startswith("phi"):
        return "Phi"
    if model_name.startswith("reka"):
        return "Reka"
    if model_name.startswith("mistral") or model_name.startswith("ministral"):
        return "Mistral"
    if model_name.startswith("stablelm"):
        return "StableLM"
    if model_name.startswith("yi"):
        return "Yi"
    if model_name.startswith("granite"):
        return "Granite"
    if model_name.startswith("nemotron"):
        return "Nemotron"
    if model_name.startswith("olmo"):
        return "Olmo"
    if model_name.startswith("qwen"):
        return "Qwen"
    if model_name.startswith("gemma"):
        return "Gemma"
    if model_name.startswith("deepseek"):
        return "DeepSeek"
    
    return "Other"

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