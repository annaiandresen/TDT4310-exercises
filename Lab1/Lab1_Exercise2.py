from nltk.corpus import brown, stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt
import nltk
import re

'''Exercise 2 – Reading a corpus
In NLTK, several corpora are available1
. Load the Brown corpus and extract data (i.e. words) from at
least two categories of your choice (or all)2
. The goal is to find basic trends or identifiers for each category
(i.e. spotting differences in the data). Do the following:
(a) Look at the top 5 most common words. Do you notice any similarities? Explain your findings
(b) Filter stopwords and repeat (a). Describe at least two new techniques you would use to further
improve the result
(c) Implement at least one technique of your choice and repeat (a)
(d) Plot a feature you find interesting (e.g. lexical diversity). NLTK has several built-in methods for
plotting (make sure you have Matplotlib installed!)
'''


def find_5_most_common_words(words):
    fdist = FreqDist(words)
    return fdist.most_common(5)


def remove_stopwords(words):
    stop_words = stopwords.words('english')
    return [w.lower() for w in words if w.lower() not in stop_words]


def remove_punctuation(words):
    return [re.sub("[^A-Za-z0-9]+", '', w) for w in words]


def remove_modals(words):
    modals = ['can', 'could', 'may', 'might', 'must', 'will', 'would', 'shall']
    return [w for w in words if w not in modals]


def remove_all(words):
    new_list = remove_stopwords(words)
    new_list = remove_punctuation(new_list)
    new_list = [w for w in new_list if w != '']
    return new_list


def print_words(words):
    i = 1
    for token in words:
        print(str(i) + ": '" + token[0] + "' with " + str(token[1]) + " occurrences")
        i += 1

def lexical_diversity(text):
    return len(set(text)) / len(text)


def create_plot():
    fileids = brown.fileids()
    cfd = nltk.ConditionalFreqDist(
        (target, fileid)
        for fileid in fileids
        for w in brown.words(fileid)
        for target in ['world', 'men']
        if w.lower() == target)
    print(len(fileids))
    # cfd.plot()


if __name__ == '__main__':

    words = brown.words()
    '''
    print("Here are the 5 most common words in the Brown Corpus:")
    print_words(find_5_most_common_words(words))

    print("\nHere are the 5 most common words excluding stopwords: ")
    print_words(find_5_most_common_words(remove_stopwords(words)))

    print("\nHere are the 5 most common words excluding stopwords, punctuation and whitespace:")
    print_words(find_5_most_common_words(remove_all(words)))
    '''
    new = remove_all(words)
    print("\nHere are the 5 most common words excluding stopwords, punctuation, whitespaces and modals:")
    print_words(find_5_most_common_words(remove_modals(new)))


    # TODO exercise 2d
    create_plot()
