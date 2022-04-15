"""Exercise 3 - Sentiment Analysis with Keras
Implement a neural network with LSTM with dropout. Download a commonly used dataset for sentiment
analysis: IMDB Reviews https://www.tensorflow.org/datasets/catalog/imdb_reviews. This is found
in the “tensorflow-datasets” package, installable with pip, or via keras https://keras.io/api/datasets/
imdb/.
1. Instantiate the dataset and clean it as you see fit. Encode the labels as you want (e.g. 0 for negative).
2. Setup a shallow feed-forward NN to give your setup an initial benchmark (no LSTM)
3. Setup a NN with LSTM. Feel free to follow the code from the ATAP book. If you wish, you can
implement GloVe embeddings in the initial layer.
4. Test out a few different model setups, different dropouts, etc. This is heavily based on your available
hardware.
5. Verify the model on a sample of texts from Chamber of Secrets. Explain your initial thoughts and
reflect on how you could create a dataset more suited to the domain of fantasy books."""

import os
import re
import shutil
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

from keras_preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM


# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
# https://www.tensorflow.org/tutorials/keras/text_classification

def load_dataset(name="imdb_reviews"):
    train_data, validation_data, test_data = tfds.load(
        name=name,
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    return train_data, validation_data, test_data


# The labels to predict are either 0 or 1.

def create_keras_layer(train_examples_batch):
    # Based on NNLM with two hidden layers.
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])
    model = Sequential()
    model.add(hub_layer)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.summary()


def build_shallow_nn():
    """
    Creates a shallow neural network with one (DENSE) hidden layer.
    Based on example in ATAP.
    :return: a Sequential NN model
    """
    N_FEATURES = 100000
    N_CLASSES = 2
    model = Sequential()

    # Layers
    model.add(Dense(500, activation='relu', input_shape=(N_FEATURES,)))  # Hidden layer
    model.add(Dense(N_CLASSES, activation='softmax'))  # Outer layer

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def build_lstm():
    N_FEATURES = 100000
    N_CLASSES = 2
    DOC_LEN = 60
    model = Sequential()
    model.add(Embedding(N_FEATURES, 128, input_length=DOC_LEN))
    model.add(Dropout(0.4))
    model.add(LSTM(units=200, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(N_CLASSES, activation='sigmoid'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    # Part 1 - Loading dataset
    train_data, validation_data, test_data = load_dataset()
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

    # Part 2 - Shallow NN Model
    model = build_shallow_nn()
    print(model.summary())

    # Part 3 - LSTM
    lstm = build_lstm()

    # reviews = list()
    # for i in range(len(train_examples_batch)):
    #     rev = (train_examples_batch.numpy()[i], train_labels_batch.numpy()[i])
    #     reviews.append(rev)
    # print(reviews)

    # url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    #
    # dataset = tf.keras.utils.get_file("aclImdb_v1", url,
    #                                   untar=True, cache_dir='.',
    #                                   cache_subdir='')
    #
    # dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    # print(os.listdir(dataset_dir))
    # train_dir = os.path.join(dataset_dir, 'train')
    # os.listdir(train_dir)
    # sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    # with open(sample_file) as f:
    #     print(f.read())
    #
    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)
    #
    # batch_size = 32
    # seed = 42
    #
    # raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    #     'aclImdb/train',
    #     batch_size=batch_size,
    #     validation_split=0.2,
    #     subset='training',
    #     seed=seed)
