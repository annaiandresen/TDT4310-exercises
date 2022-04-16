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

import re
import string
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# SOURCES
# https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
# https://www.tensorflow.org/tutorials/keras/text_classification
# https://www.kaggle.com/code/muhammed3ly/sentiment-analysis-using-lstm/notebook

def load_dataset(name="imdb_reviews"):
    train_data, validation_data, test_data = tfds.load(
        name=name,
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    return train_data, validation_data, test_data


def build_shallow_nn(shape):
    """
    Creates a shallow neural network with one (DENSE) hidden layer.
    Based on example in ATAP.
    :return: a Sequential NN model
    """
    N_CLASSES = 1
    model = Sequential()
    # Layers
    model.add(Dense(units=1024, activation='relu', input_dim=shape))
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
                                    '').numpy().decode("utf-8")

def build_lstm(shape):
    N_FEATURES = 100000
    N_CLASSES = 2
    model = Sequential()
    model.add(Embedding(N_FEATURES, 128, input_length=shape))
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
    # Part 1a - Loading dataset
    train_data, validation_data, test_data = load_dataset()
    train_data = [(custom_standardization(example.numpy()), label.numpy()) for example, label in train_data]
    df = pd.DataFrame(train_data, columns=['text', 'label'])

    # Part 1b - Preprocessing, vectorizing data
    split_ratio = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=split_ratio, random_state=4310)

    pos_df = df.loc[df['label'] == 1].reset_index(drop=True)
    neg_df = df.loc[df['label'] == 0].reset_index(drop=True)

    # Using TFIDvectorizer
    vectorizer = TfidfVectorizer(binary=False, ngram_range=(1, 3))
    vectorizer.fit(X_train)

    x_train_tv = vectorizer.transform(X_train)
    x_test_tv = vectorizer.transform(X_test)

    print(x_test_tv.shape)  # (3000, 2843628)

    # Part 2 - Shallow NN
    model = build_shallow_nn(x_train_tv.shape[1])
    model.summary()
    history = model.fit(x_train_tv, y_train, epochs=1, validation_data=(x_test_tv, y_test))
    print(history)

    # Part 3 - NN with LSTM
    lstm = build_lstm(x_train_tv.shape[1])
    lstm.summary()
    history2 = lstm.fit(x_train_tv, y_train, epochs=1, validation_data=(x_test_tv, y_test))
    print(history2)
