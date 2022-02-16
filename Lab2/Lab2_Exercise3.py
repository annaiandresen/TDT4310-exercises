"""3 – Tagging with probabilities
Hidden Makrov Models (HMMs) can be used to solve Part-of-Speech (POS) tagging. Use HMMs to
calculate probabilities for words and tags, using the appended code.
(a) Implement the missing pieces of the function task3a() found in the appended code. Also found on
the next page for reference.
(b) Print the probablity of...2
• a verb (VB) being “run”
• a preposition (PP) being followed by a verb
A template is found in the code under task3b()
(c) Print the 10 most common words for each of the tags NN, VB, JJ
(d) Print the probability of the tag sequence PP VB VB DT NN for the sentence “I can code some
code”
"""

import nltk
from nltk.corpus import brown
from Lab2_Exercise1 import print_tuples_nicely

# see https://nltk.readthedocs.io/en/latest/api/nltk.html
# define distinguishable start/end tuples of tag/word
# used to mark sentences
START = ("START", "START")
END = ("END", "END")


def get_tags(corpus):
    tags_words = []
    for sent in corpus.tagged_sents():
        # Mark the start
        tags_words.append(START)

        # shorten tags to 2 characters each for simplicity
        tags_words.extend([(tag[:2], word) for (word, tag) in sent])
        # Mark end of tags
        tags_words.append(END)

    return tags_words


def probDist(corpus, probability_distribution, tag_observation_fn):
    tag_words = get_tags(corpus)
    tags = [tag for (tag, _) in tag_words]

    # conditional frequency distribution over tag/word
    cfd_tagwords = nltk.ConditionalFreqDist(tag_words)
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, probability_distribution)

    # conditional frequency distribution of observations:
    cfd_tags = nltk.ConditionalFreqDist(tag_observation_fn(tags))
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, probability_distribution)

    return cpd_tagwords, cpd_tags


def task3a():
    corpus = brown
    # maximum likelihood estimate to create a probability distribution
    probability_distribution = nltk.MLEProbDist
    # The maximum likelihood estimate for the probability distribution of the experiment used to generate a frequency
    # distribution. The maximum likelihood estimate approximates the probability of each sample as the frequency of
    # that sample in the frequency distribution.

    # a function to create tag observations.
    tag_observation_fn = nltk.bigrams

    return probDist(corpus, probability_distribution, tag_observation_fn)


def prettify(prob):
    return "{}%".format(round(prob * 100, 4))


def task3b():
    tagwords, tags = task3a()
    prob_verb_is_run = tagwords["VB"].prob("run")
    prob_v_follows_p = tags["PP"].prob("VB")
    print("Prob. of a Verb(VB) being 'run' is", prettify(prob_verb_is_run))
    print("Prob. of a Preposition(PP) being followed by a Verb(VB) is", prettify(prob_v_follows_p))

def task3c():
    print_most_common("NN")
    print_most_common("VB")
    print_most_common("JJ")

def print_most_common(tag):
    tagwords, tags = task3a()
    fdist = tagwords[tag].freqdist()
    print("\nPRINTING THE 10 MOST COMMON WORDS FOR TAG "+tag+":")
    print_tuples_nicely(fdist.most_common(10))


def task3d():
    # text = "I can code some code"
    # PP VB VB DT NN
    tagwords, tags = task3a()
    probability = tags["START"].prob("PP") * \
                  tagwords["PP"].prob("I") * \
                  tags["PP"].prob("VB") * \
                  tagwords["VB"].prob("can") * \
                  tags["VB"].prob("VB") * \
                  tagwords["VB"].prob("code") * \
                  tags["VB"].prob("DT") * \
                  tagwords["DT"].prob("some") * \
                  tags["DT"].prob("NN") * \
                  tagwords["NN"].prob("code") * \
                  tags["NN"].prob("END")
    print("Prob. of 'I can code some code' to have the sequence PP VB VB DT NN:", prettify(probability))
    print(probability)

if __name__ == '__main__':
    task3b()
    task3c()
    task3d()

