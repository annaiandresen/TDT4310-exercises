"""Exercise 4 - GPT-2 fine-tuning and generation
Using the Hugging Face library transformers4, fine-tune a generalized GPT2-model to generate sentences
based on all chapters of Chamber of Secrets except chapter 1. Then generate sentences based on the first
few words in the original sentences of chapter 1. Explain your results.
There are a lot of resources on the topic here. Use this an opportunity to learn something cool and
avoid copy-pasting directly.
"""
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model, pipeline, set_seed, GPT2Config, AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}




if __name__ == '__main__':
    context_length = 128

    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

    raw_datasets = DatasetDict(
        {
            "train": ds_train,  # .shuffle().select(range(50000)),
            "valid": ds_valid,  # .shuffle().select(range(500))
        }
    )

    for key in raw_datasets["train"][0]:
        print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    print(tokenized_datasets)
