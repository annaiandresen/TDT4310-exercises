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
from nltk import FreqDist, ConditionalFreqDist, trigrams
import random


def n_most_frequent_tags(n, tags):
    tag_fd = FreqDist(tag for (word, tag) in tags)
    return tag_fd.most_common(n)


def get_ambiguous_words(tags, n):
    data = ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
    return [w for w in sorted(data.conditions()) if len(data[w]) > n]


def get_percentage_of_ambiguous_words(tags, n):
    data = ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
    count = len(get_ambiguous_words(tags, n))
    return round((count / len(data) * 100), 2)


def get_frequent_tags(tags):
    list_ambiguous_words = []
    data = ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
    for word in sorted(data.conditions()):
        if len(data[word]) > 3 and len(word) > 4:  # Filter out words with less than 3 tags
            tags = [tag for (tag, _) in data[word].most_common()]
            list_ambiguous_words.append((word, ' '.join(tags)))
    list_ambiguous_words = list_ambiguous_words[:5]
    return list_ambiguous_words


def print_tuples_nicely(liste):
    for entry in liste:
        print(entry[0])


def process(word, sentence, tag):
    for (w1, t1), (w2, t2), (w3, t3) in trigrams(sentence):
        if (w1.lower() == word and t1 == tag) or (w2.lower() == word and t2 == tag) or (w3.lower() == word and t3 == tag):
            print("\n'"+word + "' as " + tag)
            print_tagged_sents(sentence)
            return True

def print_tagged_sents(tagged_sent):
    liste = [w for (w, tup) in tagged_sent]
    print(' '.join(liste))


def check_tagged_sents(word, tag):
    for tagged_sent in brown.tagged_sents(tagset="universal"):
        if process(word, tagged_sent, tag):
            break


if __name__ == '__main__':
    tags = brown.tagged_words(tagset="universal")

    print("HERE ARE THE 5 MOST FREQUENT TAGS IN THE CORPUS")
    print(n_most_frequent_tags(5, tags))

    print("\nNUMBER OF AMBIGUOUS WORDS:")
    print(len(get_ambiguous_words(tags, 2)))

    print("\nPERCENTAGE OF AMBIGUOUS WORDS:")
    print(get_percentage_of_ambiguous_words(tags, 2))

    frequent_tags = get_frequent_tags(tags)

    print("\n5 words with 3 or more tags:".upper())
    print_tuples_nicely(frequent_tags)

    random_index = random.randint(0, len(frequent_tags)-1)
    tup = frequent_tags[random_index]
    tags = tup[1].split()
    word = tup[0].lower()

    print("\nPRINTING SENTENCES CONTAINING "+word+" WITH DIFFERENT TAGS")
    for tag in tags:
        print(check_tagged_sents(word, tag))

