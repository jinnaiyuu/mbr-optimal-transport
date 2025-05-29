import os

import json
from collections.abc import Iterable
from glob import glob

import numpy as np
import pandas as pd

import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import PaliGemmaForConditionalGeneration
import datasets
from peft import PeftModel

from gpt4api import GPT4API

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# TODO: this wanna be set by base directory
dataset_dir = "./dataset"
sample_dir = "./samples"
score_dir = "./score"  # not used
output_dir = "./output"  # not used
evaluate_dir = "./evaluate"
prompt_dir = "./prompts"
result_dir = "./results"
matrix_dir = "./matrix"
embed_dir = "./embed"
instruct_dir = "./instruct"

# approx_dir = './approx'
# diverse_dir = "./diverse"

reward_dir = "./reward"

HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
if HF_READ_TOKEN is None:
    print("HF_READ_TOKEN is not set. Please set it in your environment.")


def load_model(dataset, torch_device, model_name, quantize=-1):
    # TODO: It is important to specify "None" when generating texts using sequence-to-sequence models.
    # Otherwise, it will generate texts using language models.
    q4 = quantize == 4
    q8 = quantize == 8

    stop_tokens = []
    if model_name == "None":
        if "wmt19" in dataset:
            mname = "facebook/wmt19-" + dataset.split(".")[1]
            tokenizer = FSMTTokenizer.from_pretrained(mname)
            model = FSMTForConditionalGeneration.from_pretrained(mname)
            model.to(torch_device)
        elif "wmt21" in dataset:
            if "wmt21fs" in dataset:
                src_lang = dataset.split(".")[1].split("-")[0]
                if src_lang == "en":
                    mname = "facebook/wmt21-dense-24-wide-en-x"
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        mname, load_in_4bit=True, device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(mname)
                    model.config.forced_bos_token_id = tokenizer.get_lang_id(
                        dataset.split(".")[1].split("-")[1]
                    )
                else:
                    mname = "facebook/wmt21-dense-24-wide-x-en"
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        mname, load_in_4bit=True, device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(mname)
                    tokenizer.src_lang = dataset.split(".")[1].split("-")[0]

            else:
                mname = "facebook/m2m100_418M"
                tokenizer = M2M100Tokenizer.from_pretrained(mname)
                tokenizer.src_lang = dataset.split(".")[1].split("-")[0]
                model = M2M100ForConditionalGeneration.from_pretrained(mname)
                model.to(torch_device)
                model.config.forced_bos_token_id = tokenizer.get_lang_id(
                    dataset.split(".")[1].split("-")[1]
                )
        elif dataset in ["xsum", "cnndm", "samsum"]:
            if dataset == "samsum":
                mname = "philschmid/bart-large-cnn-samsum"
            else:
                mname = "facebook/bart-large-" + dataset
            model = BartForConditionalGeneration.from_pretrained(mname)
            tokenizer = BartTokenizer.from_pretrained(mname)
            model.to(torch_device)
        elif dataset in ["nocaps", "mscoco", "mscoco-ft"]:
            if dataset == "mscoco-ft":
                mname = "Salesforce/blip2-flan-t5-xl-coco"
            else:
                mname = "Salesforce/blip2-flan-t5-xl"
            tokenizer = AutoProcessor.from_pretrained(mname)
            model = Blip2ForConditionalGeneration.from_pretrained(
                mname, load_in_8bit=True, device_map="auto"
            )
        elif dataset in ["chatbot_arena_instructions-en-ja-s2s", "ethics-en-ja"]:
            mname = "facebook/wmt21-dense-24-wide-en-x"
            model = AutoModelForSeq2SeqLM.from_pretrained(mname, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(mname)
            model.config.forced_bos_token_id = tokenizer.get_lang_id("ja")
        elif dataset in ["jcm-ja-en"]:
            mname = "facebook/wmt21-dense-24-wide-x-en"
            model = AutoModelForSeq2SeqLM.from_pretrained(mname, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(mname)
            tokenizer.src_lang = "ja"
        else:
            assert False
    else:
        model_n = os.path.basename(model_name)

        if model_name in ["gpt-4o", "gpt-4o-mini"]:
            return None, GPT4API(model_name), model_name, None

        # TODO: Load the base model name from adapter_config.json.
        if "Mixtral-8x7B-Instruct-ja-en" in model_n:
            # LoRA models needs to load the base model first
            base_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif "ddyuudd/dpo_20240122-153719" in model_name:
            # TODO: We should be able to extract the base model name from
            # adapter_config.json.
            base_model_name = "line-corporation/japanese-large-lm-3.6b-instruction-sft"
        elif ("ddyuudd/dpo" in model_name) or ("ddyuudd/calm2" in model_name):
            base_model_name = "cyberagent/calm2-7b-chat"
        elif "ddyuudd/mistral" in model_name:
            base_model_name = "HuggingFaceH4/mistral-7b-sft-beta"
        elif "ddyuudd/line" in model_name:
            base_model_name = "line-corporation/japanese-large-lm-1.7b-instruction-sft"
        elif "ddyuudd/llmjp-3-13b-instruct" in model_name:
            base_model_name = "llm-jp/llmjp-3-13b-instruct"
        elif "ddyuudd/Llama-3.1-Swallow-8B-Instruct-v0.3" in model_name:
            base_model_name = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
        elif "ddyuudd/llmjp-13b" in model_name:
            base_model_name = "llm-jp/llm-jp-13b-instruct-full-jaster-v1.0"
        elif "ddyuudd/Swallow" in model_name:
            base_model_name = "tokyotech-llm/Swallow-7b-instruct-v0.1"
        elif "webbigdata/C3TR-Adapter" in model_name:
            base_model_name = "unsloth/gemma-7b-bnb-4bit"
        else:
            base_model_name = model_name

        if (
            ("polylm" in model_n)
            or ("japanese-large-lm-" in model_n)
            or ("line-" in model_n)
            or ("Llama-3.1-Swallow" in model_n)
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                legacy=False,
                use_fast=False,
                padding_side="left",
                token=HF_READ_TOKEN,
            )
            if "polylm" in model_n:
                stop_tokens = [213]  # new line token \n
        elif "llm-jp-3" in model_n:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, padding_side="left", token=HF_READ_TOKEN
            )
        elif "paligemma" in model_n:
            # Vision and Language model
            tokenizer = AutoProcessor.from_pretrained(base_model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                padding_side="left",
                use_fast=True,
                trust_remote_code=True,
                token=HF_READ_TOKEN,
            )
            if "falcon" in model_n:
                # TODO: it doesn't fully solve the problem. There seems to be
                # random newline tokens.
                stop_tokens = [193, 1001]
            elif "bloomz" in model_n:
                stop_tokens = [2]
            else:
                print("Stop token is not set for", model_n)
                stop_tokens = []

        if "paligemma" not in model_n:
            # seems paligemma does not have eos token.
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if "calm2-7b-chat" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: '  + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"

        if "cba1m-calm3-sft_train" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: '  + message['content'] + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"

        if "pythia" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|prompter|>' + message['content'] + '<|endoftext|> '}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"

        if "stablelm" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|USER|>' + message['content']}}{% endfor %}{% if add_generation_prompt %}{{ '<|ASSISTANT|>' }}{% endif %}"

        if "42dot" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<human>: ' + message['content'] + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<bot>: ' }}{% endif %}"

        if "japanese-large-lm" in base_model_name:
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'ユーザー: ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'システム: '  + message['content'] + '\n' + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'システム: ' }}{% endif %}"

        # TODO: List models which needs to be quantized to 4 bits.
        # Actually there is only a marginal drop in performance by 4 bit quants
        # compared to 8bit.
        if "Mixtral" in model_n:
            # TODO: Flast attention is not used yet as the cuda version is old.
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                device_map="auto",
                token=HF_READ_TOKEN,
            )
        elif (
            ("calm2-7b-chat" in base_model_name)
            or ("llm-jp" in model_n)
            or ("japanese-large-lm-3.6b-instruction-sft" in model_n)
        ):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                device_map="auto",
                torch_dtype="auto",
                token=HF_READ_TOKEN,
            )
        elif (
            ("Mistral" in base_model_name)
            or ("zephyr" in base_model_name)
            or ("mistral-7b-sft-beta" in base_model_name)
        ):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                device_map="auto",
                token=HF_READ_TOKEN,
            )
        elif "opus-mt-en-jap" in base_model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                token=HF_READ_TOKEN,
            )
            model.to(torch_device)
        elif "paligemma" in base_model_name:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                device_map="auto",
                torch_dtype="auto",
                token=HF_READ_TOKEN,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                load_in_4bit=q4,
                load_in_8bit=q8,
                device_map="auto",
                torch_dtype="auto",
                token=HF_READ_TOKEN,
            )

        if base_model_name != model_name:
            model = PeftModel.from_pretrained(
                model=model, model_id=model_name, token=HF_READ_TOKEN
            )

        model.eval()

    return tokenizer, model, model_name, stop_tokens


def load_dataset(dataset, ref=False, raw_text=False):
    """Load dataset for the given dataset name.

    This function is simplified to only handle the following datasets:
    - wmt24doc.*
    - jados
    - pixelprose-*

    Args:
        dataset: Name of the dataset to load
        ref: Whether to load reference data
        raw_text: Whether to return raw text

    Returns:
        Iterable of data lines
    """
    subdir_name = "hf"  # Using huggingface dataset

    if "wmt24doc" in dataset:
        if "en-de" in dataset:
            dss = datasets.load_dataset("ddyuudd/wmttest2024.en-de")["train"]
        elif "en-ja" in dataset:
            dss = datasets.load_dataset("ddyuudd/wmttest2024.en-ja")["train"]
        else:
            assert False
        if not ref:
            lines = [d["src"] for d in dss]
        else:
            lines = [d["ref"] for d in dss]
    elif dataset == "jados":
        dss = datasets.load_dataset("ddyuudd/jados")["train"]
        if not ref:
            lines = [d["src"] for d in dss]
        else:
            lines = [d["ref"] for d in dss]
    elif "pixelprose" in dataset:
        # 260 samples in each split.
        split = dataset.split("-")[1]
        dataset = datasets.load_dataset("tomg-group-umd/pixelprose")[split]
        if not ref:
            lines = []
            for i in range(len(dataset)):
                try:
                    img = dataset[i]["image"]
                    # print('img format: ', img.format)
                    imgg = img.convert("RGB")
                    lines.append(imgg)
                except BaseException:
                    print("image error: ", i)
        else:
            lines = []
            for i in range(len(dataset)):
                try:
                    img = dataset[i]["image"]
                    # print('img format: ', img.format)
                    imgg = img.convert("RGB")
                    # lines.append(imgg)
                    lines.append([dataset["caption"][i]])
                except BaseException:
                    print("image error: ", i)
                    pass
    else:
        assert False, f"Dataset {dataset} not supported in this simplified version"

    assert isinstance(lines, Iterable)
    return lines


def load_kwargs(dataset):
    model_kwargs = dict(max_new_tokens=500)
    return model_kwargs


def load_matrix(target_matrix_dir, filename, sim, n_samples):
    """
    Load matrix from the target_matrix_dir.
    The matrix is saved as a text file.
    The matrix stored may be larger than n_samples. In this case, it is truncated.
    If no files found, it returns None.
    """

    matrix_base = os.path.join(target_matrix_dir, filename + "_" + sim + "_")
    matrix_paths = glob(matrix_base + "*")

    cached_nsamples = [int(f[len(matrix_base) :]) for f in matrix_paths]
    larger_cahces = [c for c in cached_nsamples if c >= n_samples]

    if len(larger_cahces) == 0:
        return None

    # Load the smallest matrix larger than n_samples.
    # TODO: We should be able to clear matrix cache.
    min_nsamples = min(larger_cahces)

    matrix = np.loadtxt(matrix_base + str(min_nsamples))
    matrix = matrix[:n_samples, :n_samples]

    return matrix


def load_samples(dataset, sample_id, eps):
    # This function is old. not accepting input from recent sample files
    print("utils.load_samples: not maintained")
    sample_path = os.path.join(
        sample_dir, dataset, "{:04d}_eps-{:.2f}".format(sample_id, eps)
    )

    with open(sample_path) as f:
        samples = f.read().splitlines()

    return samples


def load_samples_from_file(
    files, epsilon, topk, topp, do_sample, diverse_k, divpen, temperature=1.0
):
    # TODO: Clean it up (backward compatibility)
    # To keep backward compatibility to the old format, it needs two steps.
    # First it loads in current format and it no files found, it loads in old format.
    # TODO: double negative is difficult to maintain. refactor to make it
    # intuitively understandable.
    filtered_files = []

    if do_sample > 0:
        for filename in files:
            isnt_eps = not "eps-{:.2f}".format(epsilon) in filename

            # If topk is set to negative (e.g. -1), then it means that "topk"
            # should not be in the filename.
            if topk < 0:
                isnt_topk = "topk" in filename
            else:
                isnt_topk = not "topk-{:02d}".format(topk) in filename

            if topp < 0:
                isnt_topp = "topp" in filename
            else:
                isnt_topp = not "topp-{:.2f}".format(topp) in filename

            # if (temperature > 0.99) and (temperature < 1.01):
            #     if not (isnt_eps or isnt_topk or isnt_topp):
            #         filtered_files.append(filename)
            # else:
            #     isnt_tmp = not "tmp-{:.2f}".format(temperature) in filename
            #     if not (isnt_eps or isnt_topk or isnt_topp or isnt_tmp):
            #         filtered_files.append(filename)

            if not (isnt_eps or isnt_topk or isnt_topp):
                filtered_files.append(filename)
        filtered_files.sort(key=lambda x: int(x.split("_eps")[0]))
    elif do_sample == 0:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = "divpen-{:.2f}".format(divpen) in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)
    else:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = "divpen" not in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)

    return filtered_files


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count += (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def list_to_text(words):
    text = words[0]
    for w in words[1:]:
        text = text + " " + w
    return text
