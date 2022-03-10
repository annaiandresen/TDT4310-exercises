"""
## Exercise 3 - Word features
Word features can be very useful for performing document classification,
since the words that appear in a document give a strong indication of what its semantic content is.
However, many words occur very infrequently, and some of the most informative words in a document may never have
occurred in our training data. One solution is to make use of a lexicon, which describes how different words relate to
each other.

Your task:
- Use the WordNet lexicon and augment the movie review document classifier (See NLTK book, Ch. 6, section 1.3) to use
features that generalize the words that appear in a document, making it more likely that they will match words found in
the training data.
"""
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet as wn
import random


def word_to_syn(word) -> str:
    synonyms = []
    for syn in wn.synsets(word):  # Look up synonyms
        for lemma in syn.lemma_names():  # For each lemma
            synonyms.append(lemma.lower())  # Add these lemmas to our output list

    if len(synonyms) > 0:
        index = random.randint(0, len(synonyms) - 1)  # Get a random synonym if available
        output = synonyms[index]
    else:
        output = word  # Return input word if there are no synonyms
    return output


def lexicon_features(reviews):
    review_words = set(reviews)
    features = {}
    for word in expanded_word_features:
        if word not in word_features:
            features['synset({})'.format(word)] = (word in review_words)
        features['contains({})'.format(word)] = (word in review_words)

    return features


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def synset_expansion(words) -> list:
    new_feat = []
    for word in words:
        new_feat.append(word)
        for synset in wn.synsets(word):
            for lemma in synset.lemma_names():
                new_feat.append(lemma.lower())
    return list(set(new_feat))


def synset_test():
    assert sorted(synset_expansion(["pc"])) == ["microcomputer", "pc", "personal_computer"]
    assert sorted(synset_expansion(["programming", "coder"])) == [
        'coder',
        'computer_programing',
        'computer_programmer',
        'computer_programming',
        'program',
        'programing',
        'programme',
        'programmer',
        'programming',
        'scheduling',
        'software_engineer'
    ]


if __name__ == '__main__':
    # Generating documents
    documents = [([word_to_syn(word) for word in list(movie_reviews.words(fileid))], category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # Shuffle
    random.shuffle(documents)

    # Get feature sets
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    n_most_freq = 2000
    word_features = list(all_words)[:n_most_freq]
    featuresets = [(document_features(d), c) for (d, c) in documents]

    # Choosing split ratio and generating train and test sets
    split_ratio = 0.8
    train_set, test_set = train_test_split(featuresets, test_size=split_ratio)

    # Choosing naive bayes as my classifier for this task
    classifier = nltk.NaiveBayesClassifier
    model = classifier.train(train_set)

    # Test synset expansion
    synset_test()

    # Expanding the features with the synset_expansion function
    expanded_word_features = synset_expansion(word_features)

    doc_featuresets = [(document_features(d), c) for (d, c) in documents]
    doc_train_set, doc_test_set = train_test_split(doc_featuresets, test_size=0.1)

    doc_model = model.train(doc_train_set)
    doc_model.show_most_informative_features(5)
    print("Accuracy: ", nltk.classify.accuracy(doc_model, doc_test_set))


    # With synsets
    lex_featuresets = [(lexicon_features(d), c) for (d, c) in documents]
    lex_train_set, lex_test_set = train_test_split(lex_featuresets, test_size=0.1)
    lex_model = model.train(lex_train_set)  # the same classifier as you defined above
    lex_model.show_most_informative_features()
    print("Accuracy: ", nltk.classify.accuracy(lex_model, lex_test_set))


