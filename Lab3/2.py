"""
## Exercise 2 - Spam or ham
Spam or ham is referred to a mail being spam or regular ("ham"). Follow the instructions and implement the `TODOs`
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords, wordnet
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def getData():
    spam = pd.read_csv(
        'spam.csv',
        usecols=["v1", "v2"],
        encoding="latin-1"
    ).rename(columns={"v1": "label", "v2": "text"})

    # Replace column names with numerical values
    spam.label = spam.label.replace('ham', 0)
    spam.label = spam.label.replace('spam', 1)
    return spam


class TextCleaner:
    def __init__(self, text):
        self.text = text
        self.stemmer = nltk.stem.SnowballStemmer("english")
        self.stopwords = stopwords.words("english")
        self.lem = nltk.wordnet.WordNetLemmatizer()

    def tokenize(self):
        self.text = nltk.word_tokenize(self.text)

    def lowercase(self):
        self.text = [w.lower() for w in self.text]

    def clean(self, stem=False, lem=False) -> str:
        """
        A function that cleans this object's text.
        :param stem: Whether to use a stemmer
        :param lem:  Whether to use a lemmatizer
        :return: The cleaned text in a string
        """
        self.tokenize()
        self.lowercase()
        self.remove_stopwords()
        self.remove_punctuation()
        if stem:
            self.stem()
        elif lem:
            self.lemmatize()
        return " ".join(self.text)

    def remove_stopwords(self):
        self.text = [w for w in self.text if w not in self.stopwords]

    def remove_punctuation(self):
        self.text = [re.sub("[^A-Za-z0-9]+", '', w) for w in self.text]

    def lemmatize(self):
        """
        Taken from https://www.holisticseo.digital/python-seo/nltk/lemmatize
        :return: None, sets self.text to the lemmatized sentence
        """
        nltk_tagged = nltk.pos_tag(self.text)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_pos_tagger(x[1])), nltk_tagged)
        lemmatized_sentence = []

        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(self.lem.lemmatize(word, tag))
        self.text = lemmatized_sentence

    def stem(self):
        self.text = [self.stemmer.stem(w) for w in self.text]

    @staticmethod
    def nltk_pos_tagger(nltk_tag):
        """
        Taken from https://www.holisticseo.digital/python-seo/nltk/lemmatize
        :param nltk_tag: a tag that matches nltk tags
        :return:wordnet tag
        """
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


def predict(model, vectorizer, data, all_predictions=False):
    data = vectorizer.transform(data)
    if all_predictions:
        return model.predict_proba(data)
    else:
        return model.predict(data)


def print_examples(data, probs, label1, label2, n=10):
    percent = lambda x: "{}%".format(round(x * 100, 1))

    for text, pred in list(zip(data, probs))[:n]:
        print("{}\n{}: {} / {}: {}\n{}".format(
            text,
            label1,
            percent(pred[0]),
            label2,
            percent(pred[1]),
            "-" * 100  # to print a line
        ))


if __name__ == '__main__':
    # Part 1 - Data Prep
    spam = getData()
    clean = lambda text: TextCleaner(text).clean()
    spam.text = spam.text.apply(clean)

    # Part 2 and 3 - Data analysis and training
    split_ratio = 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        spam.text, spam.label, test_size=split_ratio, random_state=4310)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)  # fit_transform on training data

    # Creating a classifier and training
    classifier = LogisticRegression()
    if classifier:
        classifier.fit(X_train, y_train)

    # Part 4 - Metrics
    y_probas = predict(classifier, vectorizer, X_test, all_predictions=True)
    print_examples(X_test, y_probas, "ham", "spam", 15)
    y_pred = predict(classifier, vectorizer, X_test)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)

    # Show precision and recall in a confusion matrix
    tn, fp, fn, tp = confusion_mat.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print("Recall={}\nPrecision={}".format(round(recall, 2), round(precision, 2)))
