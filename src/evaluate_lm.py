import argparse
import math
from openai import AsyncOpenAI, AsyncAzureOpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
import os
from tqdm import tqdm
import pathlib
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import asyncio, dataclasses
from dotenv import load_dotenv
import logging, sys
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
from anthropic import (AsyncAnthropic, RateLimitError as AnthropicRateLimitError, APIConnectionError as AnthropicAPIConnectionError, 
                       APITimeoutError as AnthropicAPITimeoutError, InternalServerError as AnthropicInternalServerError)

logging.basicConfig(stream=sys.stderr, level=logging.WARN)
logger = logging.getLogger(__name__)

from utils import read_json, write_json, generate_unique_id, batched

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview", "gpt-4o-2024-08-06"]
GOOGLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
API_MODELS = OPENAI_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS
HF_MODELS = ["llama-3.1-8b-instruct", "llama-3.1-70b-instruct"]

@dataclasses.dataclass
class ModelResponse:
    text: str
    usage: dict = None
    exception: Exception = None

def get_openai_model_args(model_args):
    openai_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            openai_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            openai_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            openai_model_args["top_p"] = model_args["top_p"]
        if "frequency_penalty" in model_args:
            openai_model_args["frequency_penalty"] = model_args["frequency_penalty"]
        if "presence_penalty" in model_args:
            openai_model_args["presence_penalty"] = model_args["presence_penalty"]

    return openai_model_args

@retry(retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def openai_chat_completion(client, messages, model="gpt-3.5-turbo", model_args=None):
    openai_model_args = get_openai_model_args(model_args)
    text = ""
    exception = None
    response = await client.chat.completions.create(model=model, messages=messages, **openai_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_openai_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await openai_chat_completion(client, messages, model=model, model_args=model_args)

def get_anthropic_model_args(model_args):
    anthropic_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            anthropic_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            anthropic_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            anthropic_model_args["top_p"] = model_args["top_p"]
        if "top_k" in model_args and model_args["top_k"] is not None:
            anthropic_model_args["top_k"] = model_args["top_k"]

    return anthropic_model_args

@retry(retry=retry_if_exception_type((AnthropicAPITimeoutError, AnthropicAPIConnectionError, AnthropicRateLimitError, AnthropicInternalServerError)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def anthropic_chat_completion(client, messages, system_prompt=None, model="claude-3-5-sonnet-20240620", model_args=None):
    anthropic_model_args = get_anthropic_model_args(model_args)
    exception = None

    try:
        response = await client.messages.create(model=model, messages=messages, system=system_prompt, **anthropic_model_args)
        text = response.content[0].text.strip()
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    except (AttributeError, ValueError) as e:
        text = ""
        usage = {}
        exception = e
    
    return ModelResponse(text, usage, exception)

async def evaluate_anthropic_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = [{"role": "user", "content": user_prompt.strip()}]
    return await anthropic_chat_completion(client, messages, system_prompt=system_prompt, model=model, model_args=model_args)

def get_google_model_args(model_args):
    google_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            google_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            google_model_args["max_output_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            google_model_args["top_p"] = model_args["top_p"]
        if "top_k" in model_args:
            google_model_args["top_k"] = model_args["top_k"]

    return google_model_args

@retry(retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, DeadlineExceeded)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def evaluate_google_model(client, model, user_prompt, system_prompt=None, model_args=None):
    text = ""
    exception = None

    model = genai.GenerativeModel(model, system_instruction=system_prompt.strip())
    google_model_args = get_google_model_args(model_args)
    config = genai.GenerationConfig(**google_model_args)
    response = model.generate_content(user_prompt.strip(), generation_config=config)
    
    try:
        text = response.text.strip()
    except ValueError as e:
        exception = f"Finish reason: {str(response.candidates[0].finish_reason)}"

    return ModelResponse(text, None, exception)

def load_model(model_path="gpt2", tokenizer_path="gpt2", model_args=None, cache_dir=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 cache_dir=cache_dir,
                                                 device_map=device,
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer

def get_hf_model_args(model_args):
    hf_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            hf_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            hf_model_args["max_new_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            hf_model_args["top_p"] = model_args["top_p"]
        if "top_k" in model_args:
            hf_model_args["top_k"] = model_args["top_k"]
        if hf_model_args["top_p"] == 1 and not hf_model_args.get("top_k"):
            hf_model_args["do_sample"] = False
        else:
            hf_model_args["do_sample"] = True
    return hf_model_args

def evaluate_hf_model(model, tokenizer, batch, model_args=None, device="cuda"):
    hf_model_args = get_hf_model_args(model_args)

    responses = []

    for sample in batch:
        messages = []

        if sample["system_prompt"]:
            messages.append({"role": "system", "content": sample["system_prompt"].strip()})
        
        messages.append({"role": "user", "content": sample["user_prompt"].strip()})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            **hf_model_args
        )
        response = outputs[0][input_ids.shape[-1]:]
        responses.append(ModelResponse(text=tokenizer.decode(response, skip_special_tokens=True)))
    
    return responses

def evaluate_model(batch, model_name, model, tokenizer, model_args=None, device="cuda"):
    if model_name in HF_MODELS:
        return evaluate_hf_model(batch, model, tokenizer, model_args=model_args, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported")

async def evaluate_api_model(client, model, batch, model_args=None):
    tasks = []
    
    for sample in batch:
        if model in OPENAI_MODELS:
            tasks.append(asyncio.create_task(evaluate_openai_model(client, model, sample["user_prompt"], sample["system_prompt"], model_args=model_args)))
        elif model in GOOGLE_MODELS:
            tasks.append(asyncio.create_task(evaluate_google_model(client, model, sample["user_prompt"], sample["system_prompt"], model_args=model_args)))
        elif model in ANTHROPIC_MODELS:
            tasks.append(asyncio.create_task(evaluate_anthropic_model(client, model, sample["user_prompt"], sample["system_prompt"], model_args=model_args)))
        else:
            raise ValueError(f"Model {model} not supported")
    
    results = await asyncio.gather(*tasks)

    return results

def configure_openai_client(api_key, is_openai_azure=False):
    if is_openai_azure:
        endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT", "https://sigturk-openai.openai.azure.com/")
        client = AsyncAzureOpenAI(
            api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY"),
            api_version = '2024-02-15-preview',
            azure_endpoint=endpoint
        )
    else:
        client = AsyncOpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
    
    return client

def configure_google_client(api_key):
    genai.configure(api_key=api_key if api_key is not None else os.getenv("GOOGLE_API_KEY"))
    return None

def configure_anthropic_client(api_key):
    return AsyncAnthropic(api_key=api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY"))

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def _write_error(error_path, sample, exception):
    with open(error_path, "a") as error_file:
        error_file.write(f"Error for sample {sample['id']}: {str(exception)}\n")
        error = "".join(traceback.format_exception(type(exception), value=exception, tb=exception.__traceback__))
        error_file.write(error)
        error_file.write("\n")
    
async def main():
    load_dotenv() 

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, help="Path to evaluation data in json", required=True)
    parser.add_argument("-a", "--api-key", type=str, help="Model API Key")
    parser.add_argument("-oa", "--openai-azure", action="store_true", help="If OpenAI on Azure")
    parser.add_argument("-m", "--model", type=str, help="Model to use for evaluation", default="gpt-4")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for generation", default=0.0)
    parser.add_argument("-g", "--max-tokens", type=none_or_int, help="Max tokens for generation", default=None)
    parser.add_argument("-p", "--top-p", type=float, help="Top-p for generation", default=1)
    parser.add_argument("-k", "--top-k", type=float, help="Top-k for generation", default=None)
    parser.add_argument("-fp", "--frequency-penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("-pp", "--presence-penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory for evaluation results", default="outputs")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to evaluate", default=0)
    parser.add_argument("-c", "--cache-dir", type=str, help="Cache directory for model", default="~/.cache")
    parser.add_argument("-mp", "--model-path", type=str, help="Model path to use for evaluation", default=None)
    parser.add_argument("-tp", "--tokenizer-path", type=str, help="Tokenizer path to use for evaluation", default=None)
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size for evaluation", default=1)
    parser.add_argument("-r", "--resume", action="store_true", help="Resume evaluation from the current file")
    
    args = parser.parse_args()
    client = None

    if args.model in API_MODELS:
        if args.model in OPENAI_MODELS:
            client = configure_openai_client(args.api_key, args.openai_azure)
        elif args.model in GOOGLE_MODELS:
            client = configure_google_client(args.api_key)
        elif args.model in ANTHROPIC_MODELS:
            client = configure_anthropic_client(args.api_key)
        
    input_data = read_json(args.datapath)
    data = input_data["data"]

    if args.num_samples > 0:
        data = data[:int(args.num_samples)]

    outputs = {
        "metadata": {
            **input_data["metadata"],
            "source": args.datapath,
            "size": len(data),
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "cache_dir": args.cache_dir,
            "ignore_path": args.ignore_path,
            "batch_size": args.batch_size,
            "openai_azure": args.openai_azure,
            "num_samples": args.num_samples,
            "resume": args.resume,
            "model_args": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty
            }
        },
        "metrics": {},
        "data": data
    }

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datapath = pathlib.Path(args.datapath)

    if args.resume:
        output_path = args.datapath
        error_path = args.datapath.replace(".json", "_errors.txt")
    else:
        unique_id = generate_unique_id()
        output_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_{unique_id}.json")
        error_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_{unique_id}_errors.txt")

    print(f"Writing to {output_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model not in API_MODELS and args.model_path:
        model, tokenizer = load_model(model_path=args.model_path, tokenizer_path=args.tokenizer_path, cache_dir=args.cache_dir, device=device)

    model_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty
    }

    for batch in tqdm(batched(data, size=args.batch_size), total=math.ceil(len(data)/args.batch_size)):
        try:
            filtered_batch = []

            for sample in batch:
                if "output" in sample:
                    continue
                
                filtered_batch.append(sample)

            results = []

            if args.model in API_MODELS:
                results = await evaluate_api_model(client, args.model, filtered_batch, model_args)
            else:
                results = evaluate_model(args.model, model, tokenizer, filtered_batch, model_args=model_args, device=device)

            for sample, result in zip(filtered_batch, results):
                sample["output"] = result.text
                sample["usage"] = result.usage
                sample["result_id"] = generate_unique_id()
                if result.exception is not None:
                    sample["exception"] = str(result.exception)
            
            write_json(outputs, output_path, ensure_ascii=False)
        except Exception as e:
            _write_error(error_path, sample, e)

    write_json(outputs, output_path, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())
