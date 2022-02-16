import nltk
from nltk.corpus import brown
# see https://nltk.readthedocs.io/en/latest/api/nltk.html
# define distinguishable start/end tuples of tag/word
# used to mark sentences
START = ("START", "START")
END = ("END", "END")

def get_tags(corpus):
    tags_words = []
    for sent in corpus.tagged_sents():
        tags_words.append(START)
        # shorten tags to 2 characters each for simplicity
        tags_words.extend([(tag[:2], word) for (word, tag) in sent])
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
    probability_distribution = None  # IMPLEMENT
    # a function to create tag observations. Hint: can we observe anything with unigrams?
    tag_observation_fn = None  # IMPLEMENT
    return probDist(corpus, probability_distribution, tag_observation_fn)
    
def prettify(prob):
    return "{}%".format(round(prob * 100, 4))
    
def task3b():
    tagwords, tags = task3a()
    prob_verb_is_run = None  # IMPLEMENT
    prob_v_follows_p = None  # IMPLEMENT
    print("Prob. of a Verb(VB) being 'run' is", prettify(prob_verb_is_run))
    print("Prob. of a Preposition(PP) being followed by a Verb(VB) is", prettify(prob_v_follows_p))