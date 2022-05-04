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


if __name__ == '__main__':
    hf = HFDataSet()
    ds = hf.get_dataset()
    tokenizer = hf.get_tokenizer()
    # print(ds)
    set_seed(1)
    model_name = "bert-base-uncased"

    # the model we gonna train, base uncased BERT
    # check text classification models here: https://huggingface.co/models?filter=text-classification
    # model_name = "bert-base-uncased"
    # max sequence length for each document/sentence sample
    # max_length = 512

    train_args = get_training_arguments()
    model = BertForSequenceClassification.from_pretrained(PATH, num_labels=4)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # trainer = Trainer(
    #     model=model,  # the instantiated Transformers model to be trained
    #     args=train_args,  # training arguments, defined above
    #     train_dataset=ds["train"],  # training dataset
    #     eval_dataset=ds["val"],  # evaluation dataset
    #     compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    #     data_collator=data_collator
    # )
    #trainer.train()
    #torch.save(trainer.model.state_dict(), PATH)
    #model.load_state_dict(torch.load(PATH))
    model.eval()

    test_set = ds["test"]

    finnish = "it did n't really seem reasonable as he was better than both xizt and LOC ."
    french = "the thing is that ORG 's story has nothing to do with sexuality according to his own admission ."
    norwegian = "i replied i did n't know what spec was , and left in shame ."  # from norway
    russian = "working as a programmer specialising in a field that requires CARDINAL to study for ages -- CARDINAL goddamn majors !"
    pipe = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    print(pipe)

    print(pipe(finnish))
    print(pipe(french))
    print(pipe(norwegian))
    print(pipe(russian))

