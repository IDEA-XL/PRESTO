from typing import Dict, List, Any, Union, Optional
from collections import Counter
from functools import cache
import contextlib
import tempfile
import shutil
import random
import subprocess
import json
import re
import io
import os

import torch
import requests
import transformers
import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image

from presto.constants import IGNORE_INDEX
from presto.modalities.base_modality import Modality

def encode_chat(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List[Modality],
) -> Dict:
    messages = list(item["messages"])
    chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

    token_to_modality = {m.token: m for m in modalities}
    modality_token_counts = Counter()
    modality_instance_counts = Counter()
    instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
    pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

    chat_part = re.split(instruct_pattern, chat_as_string)
    input_ids = []
    labels = []
    
    data_dict = dict()
    for m in modalities:
        data_dict[m.name] = m.preprocess_rows([item])[0]

    for part in chat_part:
        if "[INST]" in part:
            is_instruction = True
        else:
            is_instruction = False
        for subpart in re.split(pattern, part):
            if not subpart:
                continue
            if subpart in token_to_modality:
                assert (
                    is_instruction
                ), "There should be no modality tokens outside of instructions"
                m = token_to_modality[subpart]
                m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
                modality_instance_counts[m.name] += 1
                modality_token_counts[m.name] += m_token_width
                input_ids.extend([m.token_idx] * m_token_width)
                labels.extend([IGNORE_INDEX] * m_token_width)
            elif is_instruction:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend([IGNORE_INDEX] * len(part_ids))
            else:
                part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                input_ids.extend(part_ids)
                labels.extend(part_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    data_dict.update({"input_ids": input_ids, "labels": labels})
    return data_dict
    
    
def encode_interleaved_data(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List[Modality],
):  
    token_to_modality = {m.token: m for m in modalities}
    pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
    input_ids = []
    labels = []
    data_dict = dict()
    modality_instance_counts = Counter()
    modality_token_counts = Counter()
    
    for m in modalities:
        # ensure the item has key like "smiles" or "selfies"
        data_dict[m.name] = m.preprocess_rows([item])[0]
        
    # convert the multi-turns "messages" into a single string
    if "messages" in item and "text" not in item:
        text_str = ""
        for turn in item["messages"]:
            text_str += turn["content"]
        item["text"] = text_str
    for subpart in re.split(pattern, item["text"]):
        if not subpart:
            continue
        if subpart in token_to_modality:
            m = token_to_modality[subpart]
            m_token_width = data_dict[m.name][modality_instance_counts[m.name]][0].shape[0]
            modality_instance_counts[m.name] += 1
            modality_token_counts[m.name] += m_token_width
            input_ids.extend([m.token_idx] * m_token_width)
            labels.extend([IGNORE_INDEX] * m_token_width)
        else:
            part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
            input_ids.extend(part_ids)
            labels.extend(part_ids)
    
    # for m in modalities:
    #     assert modality_instance_counts[m.name] == len(data_dict[m.name]), f"Expected {len(data_dict[m.name])} instances for {m.name}, got {modality_instance_counts[m.name]}"
    #     m_token_total = sum(data_dict[m.name][i][0].shape[0] for i in range(len(data_dict[m.name])))
    #     assert modality_token_counts[m.name] == m_token_total, f"Expected {m_token_total} tokens for {m.name}, got {modality_token_counts[m.name]}"
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    data_dict.update({"input_ids": input_ids, "labels": labels})
    return data_dict


def parse_chat_output(output: str, style: str = "base") -> Dict:
    if style == "base":
        pattern_thoughts = r"Thoughts:(?:\n| )([\s\S]*?)\n"
        pattern_output = r"Output:(?:\n| )([\s\S]*)"
        thoughts = re.search(pattern_thoughts, output)
        if thoughts:
            thoughts = thoughts.group(1).strip()
        else:
            thoughts = None
        output = re.search(pattern_output, output).group(1).strip()
        return {"output": output, "thoughts": thoughts}
    elif style == "classification":
        # extract int from output
        thoughts = None # temporarily set to None
        output = int(re.search(r"\d+", output).group())
        return {"output": output, "thoughts": thoughts}
    elif style == "regression":
        # extract float from output
        thoughts = None # temporarily set to None
        try:
            output = float(re.search(r"\d+\.\d+", output).group())
        except:
            output = float(re.search(r"\d+", output).group())
        return {"output": output, "thoughts": thoughts}
    else:
        raise ValueError(f"Invalid style: {style}")
        

@contextlib.contextmanager
def with_local_files(fn_or_urls: List[Any]):
    local_fns = []
    fps = []
    for fn_or_url in fn_or_urls:
        if isinstance(fn_or_url, Image.Image):
            fp = tempfile.NamedTemporaryFile(suffix=".png", mode="wb")
            fn_or_url.convert("RGB").save(fp)
            fps.append(fp)
            local_fns.append(fp.name)
        elif fn_or_url.startswith("http://") or fn_or_url.startswith("https://"):
            suffix = os.path.splitext(fn_or_url)[-1]
            with requests.get(fn_or_url, stream=True) as r:
                fp = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb")
                shutil.copyfileobj(r.raw, fp)
                fps.append(fp)
                local_fns.append(fp.name)
        else:
            local_fns.append(fn_or_url)
    try:
        yield local_fns
    finally:
        for fp in fps:
            fp.close()


@cache
def _get_dataset(dataset_args: str) -> Dataset:
    return load_dataset(**json.loads(dataset_args))


def get_dataset_cached(dataset_args: Dict) -> Dataset:
    return _get_dataset(json.dumps(dataset_args))