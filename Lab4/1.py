import nltk

# the following can be of use when identifying tags:
# nltk.help.upenn_tagset()

"""
# Exercise 1 - Introduction to chunking

## 1a) Make your own noun phrase (NP) chunker, detecting noun phrases and a clause, for which verbs (VB) are followed 
by a preposition (IN) and/or a noun phrase. """


"""
## 1b)
Convert a POS tagged text to a list of tuples, where each tuple consists of a verb followed by a sequence of noun phrases and prepositions.
Example: “the little cat sat on the mat” becomes (‘sat’, ‘on’, ‘NP’) . . . 
"""


def chunks_to_verb_NP_tuples(tagged_sents, cp):
    tuples = set()
    """
    iterate the trees and subtrees of your parser.
    add all chunks starting with a verb (CLAUSE) to the set of tuples
    """
    for sent in tagged_sents:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CLAUSE':
                tuples.add((subtree[0][0], subtree[1][0], "NP"))
    return list(tuples)


# check your output :-)
import random


"""## 1c) Using the pre-annotated test set from wall street journal data (conll2000 in nltk), experiment with 
different grammars to get the highest possible F-measure. There is no evaluation of this task, but rather a motivator 
to learn something about grammars. """

if __name__ == '__main__':
    # Let's use the familiar brown corpus to begin with. Get the POS-tagged sentences.

    # a)
    sents = nltk.corpus.brown.tagged_sents()
    grammar = r"""
    NP: {<DT>?<JJ>*<NN>} # NP consists of optional DT, zero or many adjectives and one noun
    CLAUSE: {<VB>(<IN>|<NP>)+}  # main verb -> prep or noun phrase, infinite iterations 
    """

    # Parser
    chunk_parser = nltk.RegexpParser(grammar)

    # test your parser!
    test_sentence = sents[400][:10]  # just an example sentence, using the first 10 words
    chunks = chunk_parser.parse(test_sentence)
    print(chunks)


    vb_np = chunks_to_verb_NP_tuples(sents, chunk_parser)
    random.shuffle(vb_np)
    print(vb_np[:20])


    #c)

    # Testing the grammar written in nltk ch 07
    grammar2 = r"""
            NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
            PP: {<IN><NP>}               # Chunk prepositions followed by NP
            VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
            CLAUSE: {<NP><VP>}           # Chunk NP, VP
            """
    wsj = nltk.corpus.conll2000

    # new parser
    chunk_parser = nltk.RegexpParser(grammar2)
    test_sents = wsj.chunked_sents('test.txt', chunk_types=['NP'])
    print(chunk_parser.accuracy(test_sents))
