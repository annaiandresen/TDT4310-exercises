import torch
from sklearn.metrics import accuracy_score
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, BertTokenizerFast, \
    BertForSequenceClassification, pipeline
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from HFDataset import HFDataSet

PATH: str = "/Users/annaandresen/Documents/ntnu/tdt4310/TDT4310-exercises/project/model/bert/checkpoint-1500"


# https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_training_arguments():
    return TrainingArguments(
        output_dir=PATH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def get_prediction(text, tokenizer):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs.argmax()

class BertModel:
    def __init__(self, ds: HFDataSet, path: str = PATH, is_trained: bool = False):
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = path
        self.ds = ds
        print('Using device:', self.device)
        self.tokenizer = hf.get_tokenizer()
        if is_trained:
            self.model = BertForSequenceClassification.from_pretrained(self.path).to(device)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
        self.trainer = self.build_trainer()

    @staticmethod
    def get_training_arguments(learning_rate=2e-5, train_batch_size=16, eval_batch_size=16, epochs=1):
        return TrainingArguments(
            output_dir=PATH,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
        )

    def build_trainer(self):
        train_args = self.get_training_arguments()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return Trainer(
            model=self.model,  # the instantiated Transformers model to be trained
            args=train_args,  # training arguments, defined above
            train_dataset=ds["train"],  # training dataset
            eval_dataset=ds["val"],  # evaluation dataset
            compute_metrics=self.compute_metrics,  # the callback that computes metrics of interest
            data_collator=data_collator
        )

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    def train_model(self):
        self.trainer.train()
        self.trainer.save_model(PATH)

    def evaluate_model(self):
        return self.trainer.predict(self.ds["test"])

if __name__ == '__main__':
    hf = HFDataSet()
    ds = hf.get_dataset()
    tokenizer = hf.get_tokenizer()
    # print(ds)
    set_seed(1)
    model_name = "bert-base-cased"

    # the model we gonna train, base uncased BERT
    # check text classification models here: https://huggingface.co/models?filter=text-classification
    # model_name = "bert-base-uncased"
    # max sequence length for each document/sentence sample
    # max_length = 512

    train_args = get_training_arguments()
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)  # .to("cuda")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=train_args,  # training arguments, defined above
        train_dataset=ds["train"],  # training dataset
        eval_dataset=ds["val"],  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(PATH)

    # Evaluate model
    trainer.predict(ds["test"])
    pipe = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )


