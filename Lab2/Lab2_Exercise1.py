"""1 – Ambiguity
Use the Brown Corpus and...
(a) Print the 5 most frequent tags in the corpus
(b) How many words are ambiguous, in the sense that they appear with more than two tags?
(c) Print the percentage of ambiguous words in the corpus1
(d) Find the top 5 words (longer than 4 characters) with the highest number of distinct tags. Select one
of them and print out a sentence with the word in its different forms.
(e) Discuss and think about how you would attack the problem of resolving ambiguous words for a
predictive smartphone keyboard
"""

from nltk.corpus import brown
from nltk import FreqDist, ConditionalFreqDist


def n_most_frequent_tags(n, tags):
    tag_fd = FreqDist(tag for (word, tag) in tags)
    return tag_fd.most_common(n)


def get_ambiguous_words(tags):
    data = ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
    ambiguous_words = [w for w in sorted(data.conditions()) if
                       len(data[w]) >= 2]  # words that appear with at least two tags
    return len(ambiguous_words)


def get_percentage_of_ambiguous_words(tags):
    data = ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
    count = get_ambiguous_words(tags)
    return round((count / len(data) * 100), 2)


if __name__ == '__main__':
    tags = brown.tagged_words()

    print("HERE ARE THE 5 MOST FREQUENT TAGS IN THE CORPUS")
    print(n_most_frequent_tags(5, tags))

    print("\nNUMBER OF AMBIGUOUS WORDS:")
    print(get_ambiguous_words(tags))

    print("\nPERCENTAGE OF AMBIGUOUS WORDS:")
    print(get_percentage_of_ambiguous_words(tags))
