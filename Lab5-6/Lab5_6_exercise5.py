"""Exercise 5 - Preparing for your project with Transformers
By now you have probably decided on your topic for the main project. Use this task as an introductory part,
whether that is to explore entity recognition, dependency parsing, sentiment analysis, summarization, or
anything else. Explore the library and experiment with already fine-tuned models related to your problem.
See e.g. https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads.
Note down your results here and what you may have found :-)
"""

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import pandas as pd


def load_dataset(file_path="nli_dataset.csv"):
    # Dataset from http://corefl.learnercorpora.com/search_simple
    dataset = pd.read_csv(file_path, engine='python', encoding="utf-8", sep='\t', usecols=['L1', 'Text'])
    return dataset

def create_model():
    pass


if __name__ == '__main__':
    dataset = load_dataset()
    print(dataset)
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)
    # print(output)
