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
from reka.client import AsyncReka
from zhipuai import ZhipuAI
from ai21 import AI21Client
from ai21.models.chat import SystemMessage as AI21SystemMessage, UserMessage as AI21UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage, UserMessage as AzureUserMessage
from azure.core.credentials import AzureKeyCredential
from cohere import ClientV2 as CohereClient
from mistralai import Mistral

logging.basicConfig(stream=sys.stderr, level=logging.WARN)
logger = logging.getLogger(__name__)

from utils import read_json, write_json, generate_unique_id, batched

TOGETHER_MODEL_MAP = {
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-3.1-405b-instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "qwen-2.5-coder-32b-instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "wizardlm-2-8x22b": "microsoft/WizardLM-2-8x22B",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma-2b": "google/gemma-2b-it",
    "deepseek-llm-chat-67b": "deepseek-ai/deepseek-llm-67b-chat",
    "mythomax-l2-13b": "Gryphe/MythoMax-L2-13b",
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b-instruct": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "nous-hermes-2-mixtral-8x7b-dpo": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "qwen-2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "stripedhyena-nous-7b": "togethercomputer/StripedHyena-Nous-7B",
    "solar-10.7b-instruct-v1.0": "upstage/SOLAR-10.7B-Instruct-v1.0"
}

HF_MODEL_MAP = {
    "yi-1.5-34b-chat": "01-ai/Yi-1.5-34B-Chat",
    "yi-1.5-9b-chat": "01-ai/Yi-1.5-9B-Chat",
}

NVIDIA_MODEL_MAP = {
    "nemotron-4-340b-instruct": "nvidia/nemotron-4-340b-instruct",
    "yi-large": "01-ai/yi-large",
    "granite-34b-code-instruct": "ibm/granite-34b-code-instruct",
    "granite-8b-code-instruct": "ibm/granite-8b-code-instruct",
    "mistral-nemo-12b-instruct": "nv-mistralai/mistral-nemo-12b-instruct",
    "baichuan2-13b-chat": "baichuan-inc/baichuan2-13b-chat",
    "nemotron-mini-4b-instruct": "nvidia/nemotron-mini-4b-instruct",
    "zamba2-7b-instruct": "zyphra/zamba2-7b-instruct",
    "granite-3.0-8b-instruct": "ibm/granite-3.0-8b-instruct",
    "dbrx-instruct": "databricks/dbrx-instruct",
    "gemma-2-2b-it": "google/gemma-2-2b-it"
}

MODEL_MAP = {**TOGETHER_MODEL_MAP, **HF_MODEL_MAP, **NVIDIA_MODEL_MAP}

OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview", "gpt-4o-2024-08-06", "gpt-4o"]
GOOGLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-exp-1121"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
TOGETHER_MODELS = list(TOGETHER_MODEL_MAP.keys())
XAI_MODELS = ["grok-beta"]
HF_INFERENCE_MODELS = ["yi-1.5-34b-chat", "yi-1.5-9b-chat"]
REKA_MODELS = ["reka-core", "reka-edge", "reka-flash"]
ZHIPU_MODELS = ["glm-4-0520"]
AI21_MODELS = ["jamba-1.5-mini", "jamba-1.5-large"]
AZURE_MODELS = ["Phi-3-mini-4k-instruct", "Phi-3-small-8k-instruct", "Phi-3-medium-4k-instruct", "Phi-3.5-MoE-instruct"]
COHERE_MODELS = ["command-r-plus", "c4ai-aya-expanse-8b", "c4ai-aya-expanse-32b"]
MISTRAL_MODELS = ["mistral-large-latest", "ministral-3b-latest", "ministral-8b-latest", "mistral-small-latest"]
NVIDIA_MODELS = list(NVIDIA_MODEL_MAP.keys())

OPENAI_COMPATIBLE_MODELS = OPENAI_MODELS + TOGETHER_MODELS + XAI_MODELS + HF_INFERENCE_MODELS + NVIDIA_MODELS
API_MODELS = OPENAI_COMPATIBLE_MODELS + GOOGLE_MODELS + ANTHROPIC_MODELS + REKA_MODELS + ZHIPU_MODELS + AI21_MODELS + AZURE_MODELS + COHERE_MODELS + MISTRAL_MODELS
HF_MODELS = []

@dataclasses.dataclass
class ModelResponse:
    text: str
    usage: dict = None
    exception: Exception = None

def get_openai_model_args(model_args, model=None):
    openai_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            openai_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            openai_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            openai_model_args["top_p"] = model_args["top_p"]
        if "frequency_penalty" in model_args and (not model or model not in list(NVIDIA_MODEL_MAP.values())):
            openai_model_args["frequency_penalty"] = model_args["frequency_penalty"]
        if "presence_penalty" in model_args and (not model or model not in list(NVIDIA_MODEL_MAP.values())):
            openai_model_args["presence_penalty"] = model_args["presence_penalty"]
        if "stop" in model_args and model_args["stop"] is not None and (not model or model not in list(NVIDIA_MODEL_MAP.values())):
            openai_model_args["stop"] = [model_args["stop"]]

    return openai_model_args

@retry(retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=before_sleep_log(logger, logging.DEBUG))
async def openai_chat_completion(client, messages, model="gpt-3.5-turbo", model_args=None):
    openai_model_args = get_openai_model_args(model_args, model)
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
    model = MODEL_MAP.get(model, model)
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await openai_chat_completion(client, messages, model=model, model_args=model_args)

async def reka_chat_completion(client, messages, model="reka-core", model_args=None):
    reka_model_args = get_openai_model_args(model_args)
    text = ""
    exception = None
    response = await client.chat.create(model=model, messages=messages, **reka_model_args)
    content = response.responses[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.responses[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_reka_model(client, model, user_prompt, system_prompt=None, model_args=None):
    content = ""
    messages = []

    if system_prompt:
        content = system_prompt.strip()
    
    content = content + "\n" + user_prompt.strip()
    messages.append({"role": "user", "content": content.strip()})

    return await reka_chat_completion(client, messages, model=model, model_args=model_args)

def get_zhipu_model_args(model_args):
    zhipu_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            zhipu_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            zhipu_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            zhipu_model_args["top_p"] = model_args["top_p"]
        if "stop" in model_args and model_args["stop"] is not None:
            zhipu_model_args["stop"] = [model_args["stop"]]

    return zhipu_model_args

async def zhipu_chat_completion(client, messages, model="glm-4", model_args=None):
    zhipu_model_args = get_zhipu_model_args(model_args)
    text = ""
    exception = None
    response = client.chat.completions.create(model=model, messages=messages, **zhipu_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_zhipu_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await zhipu_chat_completion(client, messages, model=model, model_args=model_args)

def get_ai21_model_args(model_args):
    ai21_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            ai21_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            ai21_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            ai21_model_args["top_p"] = model_args["top_p"]
        if "stop" in model_args and model_args["stop"] is not None:
            ai21_model_args["stop"] = [model_args["stop"]]

    return ai21_model_args

async def ai21_chat_completion(client, messages, model="jamba-1.5-mini", model_args=None):
    ai21_model_args = get_ai21_model_args(model_args)
    text = ""
    exception = None
    response = client.chat.completions.create(model=model, messages=messages, **ai21_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_ai21_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append(AI21SystemMessage(content=system_prompt.strip()))
    
    messages.append(AI21UserMessage(content=user_prompt.strip()))

    return await ai21_chat_completion(client, messages, model=model, model_args=model_args)

def get_azure_model_args(model_args):
    azure_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            azure_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            azure_model_args["max_tokens"] = model_args["max_tokens"]
        if "top_p" in model_args:
            azure_model_args["top_p"] = model_args["top_p"]
        if "stop" in model_args and model_args["stop"] is not None:
            azure_model_args["stop"] = [model_args["stop"]]

    return azure_model_args

async def azure_chat_completion(client, messages, model="Phi-3-small-instruct-8k", model_args=None):
    azure_model_args = get_azure_model_args(model_args)
    text = ""
    exception = None
    response = client.complete(model=model, messages=messages, **azure_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_azure_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append(AzureSystemMessage(content=system_prompt.strip()))
    
    messages.append(AzureUserMessage(content=user_prompt.strip()))

    return await azure_chat_completion(client, messages, model=model, model_args=model_args)

def get_cohere_model_args(model_args):
    cohere_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            cohere_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            cohere_model_args["max_tokens"] = model_args["max_tokens"]
        if "frequency_penalty" in model_args:
            cohere_model_args["frequency_penalty"] = model_args["frequency_penalty"]
        if "presence_penalty" in model_args:
            cohere_model_args["presence_penalty"] = model_args["presence_penalty"]
        if "top_p" in model_args:
            cohere_model_args["p"] = model_args["top_p"]
        if "stop" in model_args and model_args["stop"] is not None:
            cohere_model_args["stop_sequences"] = [model_args["stop"]]

    return cohere_model_args

async def cohere_chat_completion(client, messages, model="Phi-3-small-instruct-8k", model_args=None):
    cohere_model_args = get_cohere_model_args(model_args)
    text = ""
    exception = None
    response = client.chat(model=model, messages=messages, **cohere_model_args)
    content = response.message.content[0].text
    
    if content is None:
        exception = f"Finish reason: {response.finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.tokens.input_tokens, "output_tokens": response.usage.tokens.output_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_cohere_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await cohere_chat_completion(client, messages, model=model, model_args=model_args)

def get_mistral_model_args(model_args):
    mistral_model_args = {}

    if model_args is not None:
        if "temperature" in model_args:
            mistral_model_args["temperature"] = model_args["temperature"]
        if "max_tokens" in model_args:
            mistral_model_args["max_tokens"] = model_args["max_tokens"]
        if "frequency_penalty" in model_args:
            mistral_model_args["frequency_penalty"] = model_args["frequency_penalty"]
        if "presence_penalty" in model_args:
            mistral_model_args["presence_penalty"] = model_args["presence_penalty"]
        if "top_p" in model_args:
            mistral_model_args["top_p"] = model_args["top_p"]
        if "stop" in model_args and model_args["stop"] is not None:
            mistral_model_args["stop"] = [model_args["stop"]]

    return mistral_model_args

async def mistral_chat_completion(client, messages, model="mistral-small-latest", model_args=None):
    mistral_model_args = get_mistral_model_args(model_args)
    text = ""
    exception = None
    response = client.chat.complete(model=model, messages=messages, **mistral_model_args)
    content = response.choices[0].message.content
    
    if content is None:
        exception = f"Finish reason: {response.choices[0].finish_reason}"
        usage = None
    else:
        text = content.strip()
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    
    return ModelResponse(text, usage, exception)

async def evaluate_mistral_model(client, model, user_prompt, system_prompt=None, model_args=None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    messages.append({"role": "user", "content": user_prompt.strip()})

    return await mistral_chat_completion(client, messages, model=model, model_args=model_args)

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

        if sample.get("system_prompt"):
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
        if model in OPENAI_COMPATIBLE_MODELS:
            tasks.append(asyncio.create_task(evaluate_openai_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in GOOGLE_MODELS:
            tasks.append(asyncio.create_task(evaluate_google_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in ANTHROPIC_MODELS:
            tasks.append(asyncio.create_task(evaluate_anthropic_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in REKA_MODELS:
            tasks.append(asyncio.create_task(evaluate_reka_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in ZHIPU_MODELS:
            tasks.append(asyncio.create_task(evaluate_zhipu_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in AI21_MODELS:
            tasks.append(asyncio.create_task(evaluate_ai21_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in AZURE_MODELS:
            tasks.append(asyncio.create_task(evaluate_azure_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in COHERE_MODELS:
            tasks.append(asyncio.create_task(evaluate_cohere_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        elif model in MISTRAL_MODELS:
            tasks.append(asyncio.create_task(evaluate_mistral_model(client, model, sample["user_prompt"], sample.get("system_prompt"), model_args=model_args)))
        else:
            raise ValueError(f"Model {model} not supported")
    
    results = await asyncio.gather(*tasks)

    return results

def configure_openai_client(model, api_key, is_openai_azure=False):
    if is_openai_azure:
        endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT", "https://sigturk-openai.openai.azure.com/")
        client = AsyncAzureOpenAI(
            api_key = api_key if api_key is not None else os.getenv("AZURE_OPENAI_API_KEY"),
            api_version = '2024-02-15-preview',
            azure_endpoint=endpoint
        )
    else:
        if model in HF_INFERENCE_MODELS:
            client = AsyncOpenAI(base_url="https://api-inference.huggingface.co/v1/", api_key=api_key if api_key is not None else os.getenv("HF_API_KEY"))
        elif model in TOGETHER_MODELS:
            client = AsyncOpenAI(base_url="https://api.together.xyz/v1", api_key=api_key if api_key is not None else os.getenv("TOGETHER_API_KEY"))
        elif model in XAI_MODELS:
            client = AsyncOpenAI(base_url="https://api.x.ai/v1", api_key=api_key if api_key is not None else os.getenv("XAI_API_KEY"))
        elif model in NVIDIA_MODELS:
            client = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key if api_key is not None else os.getenv("NVIDIA_API_KEY"))
        else:
            client = AsyncOpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
    
    return client

def configure_google_client(api_key):
    genai.configure(api_key=api_key if api_key is not None else os.getenv("GOOGLE_API_KEY"))
    return None

def configure_anthropic_client(api_key):
    return AsyncAnthropic(api_key=api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY"))

def configure_reka_client(api_key):
    return AsyncReka(api_key=api_key if api_key is not None else os.getenv("REKA_API_KEY"))

def configure_zhipu_client(api_key):
    return ZhipuAI(api_key=api_key if api_key is not None else os.getenv("ZHIPU_API_KEY"))

def configure_ai21_client(api_key):
    return AI21Client(api_key=api_key if api_key is not None else os.getenv("AI21_API_KEY"))

def configure_azure_client(api_key):
    return ChatCompletionsClient(endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT"),
                                 credential=AzureKeyCredential(api_key if api_key is not None else os.getenv("AZURE_INFERENCE_API_KEY")))

def configure_cohere_client(api_key):
    return CohereClient(api_key=api_key if api_key is not None else os.getenv("COHERE_API_KEY"))

def configure_mistral_client(api_key):
    return Mistral(api_key=api_key if api_key is not None else os.getenv("MISTRAL_API_KEY"))

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
    parser.add_argument("-s", "--stop", type=str, help="Stop token for generation", default=None)
    
    args = parser.parse_args()
    client = None

    if args.model in API_MODELS:
        if args.model in OPENAI_COMPATIBLE_MODELS:
            client = configure_openai_client(args.model, args.api_key, args.openai_azure)
        elif args.model in GOOGLE_MODELS:
            client = configure_google_client(args.api_key)
        elif args.model in ANTHROPIC_MODELS:
            client = configure_anthropic_client(args.api_key)
        elif args.model in REKA_MODELS:
            client = configure_reka_client(args.api_key)
        elif args.model in ZHIPU_MODELS:
            client = configure_zhipu_client(args.api_key)
        elif args.model in AI21_MODELS:
            client = configure_ai21_client(args.api_key)
        elif args.model in AZURE_MODELS:
            client = configure_azure_client(args.api_key)
        elif args.model in COHERE_MODELS:
            client = configure_cohere_client(args.api_key)
        elif args.model in MISTRAL_MODELS:
            client = configure_mistral_client(args.api_key)
        else:
            raise ValueError(f"Model {args.model} not supported")
        
    input_data = read_json(args.datapath)
    data = input_data["data"]

    if args.num_samples > 0:
        data = data[:int(args.num_samples)]
    
    stop_token = args.stop.replace("\\n", "\n") if args.stop else None

    outputs = {
        "metadata": {
            **input_data["metadata"],
            "source": args.datapath,
            "size": len(data),
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "cache_dir": args.cache_dir,
            "batch_size": args.batch_size,
            "openai_azure": args.openai_azure,
            "num_samples": args.num_samples,
            "resume": args.resume,
            "stop": stop_token,
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
        "presence_penalty": args.presence_penalty,
        "stop": stop_token
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
