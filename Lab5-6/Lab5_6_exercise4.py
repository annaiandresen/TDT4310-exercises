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
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset, load_dataset, DatasetDict
from Lab5_6_exercise1 import separate_into_chapters
import torch

# https://discuss.tensorflow.org/t/attributeerror-tensorflow-python-framework-ops-eagertensor-object-has-no-attribute-to-tensor/5044/3
# https://huggingface.co/course/chapter7/6

tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='<pad>')
context_length = 128
PATH = "models/model.pt"


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


def create_dataframe(sentences: list):
    """
    Creates a Pandas dataframe and convert to to Hugging Face dataset
    :param sentences: Sentences in a list
    :return: Hugging Face Dataset
    """
    df = pd.DataFrame({'Text': sentences})
    train_ds, val_ds, test_ds = get_dataset_partitions_pd(df)
    return Dataset.from_pandas(train_ds), Dataset.from_pandas(val_ds), Dataset.from_pandas(test_ds)


def tokenize_element(dataset: Dataset) -> Dataset:
    """
    Tokenizes each sentence in dataset
    :param dataset: a Dataset object.
    :return: a tokenized dataset.
    """
    dataset_tok = dataset.map(lambda x: tokenizer(x['Text'], truncation=True, padding=True,
                              max_length=context_length))
    dataset_tok = dataset_tok.remove_columns(['Text', '__index_level_0__']).with_format('torch').remove_columns(
        ['attention_mask'])
    return dataset_tok


# def tokenize(element):
#     outputs = tokenizer(
#         element["Text"],
#         truncation=True,
#         max_length=context_length,
#         return_overflowing_tokens=True,
#         return_length=True,
#     )
#     input_batch = []
#     for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#         if length == context_length:
#             input_batch.append(input_ids)
#     return {"input_ids": input_batch}


if __name__ == '__main__':
    chapters = separate_into_chapters('chamber_of_secrets.txt')

    train_chapters = chapters[1:]
    train_chapters = nltk.sent_tokenize(' '.join(train_chapters))

    # test_chapter = chapters[0]

    # Data prep and analysis
    train_df, test_df, val_df, = create_dataframe(train_chapters)

    # https://huggingface.co/docs/transformers/main/preprocessing

    # Tokenize
    train_df = tokenize_element(train_df)
    val_df = tokenize_element(val_df)

    tokenized_datasets = DatasetDict(
        {
            "train": train_df,
            "valid": val_df
        }
    )

    print(tokenized_datasets)


    # Training model
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    args = TrainingArguments(
        output_dir="models",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()
    torch.save(trainer.model.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH))
    model.eval()


    pipe = pipeline(
        "text-generation", model=trainer.model, tokenizer=trainer.tokenizer
    )
    result = pipe("Harry Potter was", num_return_sequences=5)
    for res in result:
        print(res['generated_text'])

