"""2 – Training a tagger
Explore the performance of a tagger using the Brown Corpus and NPS Chat Corpus as data sources, with
different ratios of train/test data. Use the following ratios:
• Brown 90%/NPS 10%
• Brown 50%/NPS 50%
• NPS 90%/Brown 10%
• NPS 50%/Brown 50%
Create the taggers listed below and comment your results.
(a) Create a DefaultTagger using the most common tag in each corpus as the default tag.
(b) Create a combined tagger with the RegEx tagger (see Ch. 5, sec. 4.2) with an initial backoff using
the most common default tag. Then, use n-gram taggers as backoff taggers (e.g., UnigramTagger,
BigramTagger, TrigramTagger). The ordering is up to you, but justify your choice. Calculate the
accuracy of each of the four train/test permutations.
(c) Select a dataset split of your choice and print a table containing the precision, recall and f-measure
for the top 5 most common tags (look up truncate in the documentation) and sort each score by
count. Do this for all your chosen variations of backoffs (e.g., DefaultTagger, UnigramTagger and
BigramTagger).
(d) Using the Brown Corpus, create a baseline tagger (e.g. Unigram) with a lookup model (see Ch. 5,
sec. 4.3). The model should handle the most 200 common words and store the tags. Evaluate the
accuracy on the above permutations of train/test data.
(e) With an arbitrary text from another corpus (or an article you scraped in Lab 1), use the tagger you
just created and print a few tagged sentences.
(f) Experiment with different ratios and using only one dataset with a train/test split. Explain your
findings"""

from nltk.corpus import brown, nps_chat, gutenberg
from sklearn.model_selection import train_test_split as split
from nltk.tag import DefaultTagger, UnigramTagger, RegexpTagger, BigramTagger
from Lab2_Exercise1 import n_most_frequent_tags
from nltk import word_tokenize, FreqDist, ConditionalFreqDist

# Regex patterns for Regextagger
patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]


def get_most_common_tag(tags):
    return n_most_frequent_tags(1, tags)[0][0]


def split_data(tagged_sents, tagged_sents2, r1, r2):
    train1, test1 = split(tagged_sents, train_size=r1)
    train2, test2 = split(tagged_sents2, test_size=r2)
    return train1, test1, train2, test2


def create_tagger_program(tags1, tags2, r1, r2, most_common_tag):
    train_set1, test_set1, train_set2, test_set2 = split_data(tags1, tags2, r1, r2)
    b_tag = create_taggers(train_set1, test_set2, most_common_tag)
    print("\nNow printing accuracy with ratio reversed")
    b_tag = create_taggers(train_set2, test_set1, most_common_tag)
    return train_set1, test_set1, train_set2, test_set2, b_tag


def create_taggers(train, test, tag):
    # Default tagger
    default = DefaultTagger(tag)
    default.tag(train)

    # Regex tagger
    r_tagger = RegexpTagger(patterns, backoff=default)

    # Unigram tagger
    u_tagger = UnigramTagger(train=train, backoff=r_tagger)

    # Bigram tagger
    b_tagger = BigramTagger(train=train, backoff=u_tagger)

    print("Default Tagger accuracy: ", round(default.accuracy(test), 4))
    print("Regex tagger accuracy with default tagger as backoff: ", round(r_tagger.accuracy(test), 4))
    print("Unigram tagger accuracy with regex tagger as backoff: ", round(u_tagger.accuracy(test), 4))
    print("Bigram tagger accuracy with bigram tagger as backoff: ", round(b_tagger.accuracy(test), 4))
    return b_tagger


def create_lookup_tagger():
    """
    Shamelessly copied from NLTK: https://www.nltk.org/book/ch05.html
    :param tagged_sents:
    :return: A Unigram tagger created from most frequent words
    """
    fd = FreqDist(brown.words(categories='fiction'))
    cfd = ConditionalFreqDist(brown.tagged_words(categories='fiction'))
    most_freq_words = fd.most_common(200)
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    baseline_tagger = UnigramTagger(model=likely_tags)
    return baseline_tagger


if __name__ == '__main__':
    # Brown
    brown_tags = brown.tagged_words()
    most_common_tag = get_most_common_tag(brown_tags)
    print(most_common_tag + " IS THE MOST COMMON TAG IN THE BROWN CORPUS")
    brown_tags = brown.tagged_sents()
    nps_tags = nps_chat.tagged_posts()
    # nps_tags = nps_chat.tagged_words()
    # most_common_tag = get_most_common_tag(nps_tags) # (we already know that Noun is the most common tag in both corpora)
    # print(most_common_tag + " IS THE MOST COMMON TAG IN THE NPS_CHAT CORPUS")

    print("\nPRINTING ACCURACY WITH A BROWN 90 % / NPS 10 % RATIO")
    brown_train90, brown_test10, nps_train90, nps_test10, b_tagger = create_tagger_program(brown_tags, nps_tags, 0.9,
                                                                                           0.1, most_common_tag)

    print("\nPRINTING ACCURACY WITH A BROWN 50 % / NPS 50 % RATIO")
    brown_train50, brown_test50, nps_train50, nps_test50, b_tagger2 = create_tagger_program(brown_tags, nps_tags, 0.5,
                                                                                            0.5, most_common_tag)

    print("TABLE CONTAINING PRECISION")
    print(b_tagger.evaluate_per_tag(nps_test50, truncate=5, sort_by_count=True))

    print("CREATING LOOKUP TAGGER")
    lookup_tagger = create_lookup_tagger()
    print("EVALUATING LOOKUP TAGGER WITH 50% TRAIN SET FROM NPS CHAT")
    print(lookup_tagger.accuracy(nps_test50))

    print("TESTING TAGGER ON EMMA BY JANE AUSTEN IN THE GUTENBERG CORPUS")
    text = gutenberg.raw('austen-emma.txt')
    tokenized = word_tokenize(text)
    tagged_text = lookup_tagger.tag(tokenized)

    print("PRINTING RANDOM SENTENCES FROM THE TEXT")
    for i in range(21, 40):
        print(tagged_text[i], end=" ")

    print("TESTING BIGRAM TAGGER ON DIFFERENT SETS")
    print("NPS TEST 50% :", b_tagger.accuracy(nps_test50))
    print("BROWN TEST 50% :", b_tagger.accuracy(brown_test50))
    print("NPS TEST 10% :", b_tagger.accuracy(nps_test10))
    print("BROWN TEST 90% :", b_tagger.accuracy(brown_test10))
