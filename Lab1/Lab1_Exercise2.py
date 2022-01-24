from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
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

def remove_stopwords_punctuation(words):
    stop_words = stopwords.words('english')
    # Remove stopwords
    words_without_stopwords = [w.lower() for w in words if w.lower() not in stop_words]
    # Remove punctuation
    words_without_stopwords = [re.sub("[^A-Za-z0-9]+", '', w) for w in words_without_stopwords]
    # Remove whitespaces
    words_without_stopwords = [w for w in words_without_stopwords if w != '']
    return words_without_stopwords



if __name__ == '__main__':
    words = brown.words()
    print("Here are the 5 most common words in the Brown Corpus:")
    print(find_5_most_common_words(words))

    print("Here are the 5 most common words excluding whitespace, punctuation and stopwords: ")
    print(find_5_most_common_words(remove_stopwords_punctuation(words)))