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
from Lab5_6_exercise1 import separate_into_chapters


# https://discuss.tensorflow.org/t/attributeerror-tensorflow-python-framework-ops-eagertensor-object-has-no-attribute-to-tensor/5044/3

# # dummy sentences
# sentences = ['the house is blue and big', 'this is fun stuff', 'what a horrible thing to say']
#
# # create a pandas dataframe and conver to to Hugging Face dataset
# df = pd.DataFrame({'Text': sentences})
# dataset = Dataset.from_pandas(df)
#
# # download bert tokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#
# # tokenize each sentence in dataset
# dataset_tok = dataset.map(lambda x: tokenizer(x['Text'], truncation=True, padding=True, max_length=10), batched=True)
#
# # remove original text column and set format
# dataset_tok = dataset_tok.remove_columns(['Text']).with_format('tensorflow')
#
# # extract features
# features = {x: dataset_tok[x].to_tensor() for x in tokenizer.model_input_names}

def create_dataframe(sentences: list) -> Dataset:
    """
    Creates a Pandas dataframe and convert to to Hugging Face dataset
    :param sentences: Sentences in a list
    :return: Hugging Face Dataset
    """
    df = pd.DataFrame({'Text': sentences})
    return Dataset.from_pandas(df)


def tokenize(dataset: Dataset, tokenizer: GPT2Tokenizer):
    """
    Tokenizes each sentence in dataset
    :param dataset:
    :return:
    """
    dataset_tok = dataset.map(lambda x: tokenizer(x['Text'], truncation=True, padding=True, max_length=10),
                              batched=True)
    dataset_tok = dataset_tok.remove_columns(['Text']).with_format('tensorflow')
    features = extract_features(dataset_tok, tokenizer)
    return dataset_tok, features


def extract_features(dataset: Dataset, tokenizer: GPT2Tokenizer) -> dict:
    """
    :param dataset: Tokenized dataset
    :param tokenizer: A GPT2 tokenizer
    :return: A dictionary with features
    """
    return {x: dataset[x] for x in tokenizer.model_input_names}

def train(dataset, model, tokenizer):
    pass

def save_model(model):
    pass


if __name__ == '__main__':
    chapters = separate_into_chapters('chamber_of_secrets.txt')

    train_chapters = chapters[1:]
    # train_chapters = [nltk.sent_tokenize(chapter) for chapter in train_chapters]
    test_chapter = chapters[0]
    test_chapter = nltk.sent_tokenize(test_chapter)
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
    context_length = 128

    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="train")

    raw_datasets = DatasetDict(
        {
            "train": ds_train,  # .shuffle().select(range(50000)),
            "valid": ds_valid,  # .shuffle().select(range(500))
        }
    )

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<sos>', eos_token='<eos>', pad_token='<pad>')
    # encoded_input = tokenizer(train_chapters, return_tensors='tf', padding=True, truncation=True)
    # config = GPT2Config(
    #     vocab_size=tokenizer.vocab_size,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id
    # )
    # model = TFGPT2Model(config)
    # model.resize_token_embeddings(len(tokenizer))
    # output = model(encoded_input)
    # generator = pipeline('text-generation', model='gpt2')
    # set_seed(42)
    # result = generator("Anna was", max_length=15, num_return_sequences=5)
    # for dic in result:
    #     print(dic['generated_text'])

    # df = create_dataframe(train_chapters[0])
    # dataset_tok, features = tokenize(df, tokenizer)
    # print(features)



    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = TFGPT2Model.from_pretrained('gpt2')
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='tf')
    # output = model(encoded_input)
    #
    # generator = pipeline('text-generation', model='gpt2')
    # set_seed(42)
    # result = generator("Anna was", max_length=15, num_return_sequences=5)
    # for dic in result:
    #     print(dic['generated_text'])
