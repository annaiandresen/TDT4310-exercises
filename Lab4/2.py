import nltk

"""
# Exercise 2 - Making use of chunks

## 2a)
With the following grammar rules:
```
1. proper noun singular
2. determiner followed by an adjective, followed by any noun
3. two consecutive nouns
```
Create a `RegexpParser` chunker
"""


def task2a():
    grammar = r"""
        NP: {<DT>?<JJ.*>*<NN.*>+} # Determiner followed (optionally) by one or more adjectives, followed by any noun
            {<NN><NN>} # Chunk two consecutive nouns
            {<NNP>+}  # Proper noun singular
        """
    return nltk.RegexpParser(grammar)


"""
## 2b)

Read the file `starlink.txt` and perform the following operations on the text:
- sentence tokenize
- word tokenize
- pos tag

"""

def get_pos_tags_from_file(file):
    with open(file, 'r') as f:
        sample = f.read()
    sents = nltk.sent_tokenize(sample)
    f.close()
    return tokenize_and_tag(sents)


def tokenize_and_tag(sents):
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sents]
    return [nltk.pos_tag(sentence) for sentence in tokenized_sentences]


"""## 2c) From all found subtrees in the text, print out the text from all the leaves on the form of `DT -> JJ -> NN` 
(i.e. the CONSECUTIVE chunk you defined above) """


def get_descriptive_nouns(tagged_sents, cp):
    return [parse_sent(sent, cp) for sent in tagged_sents]

def parse_sent(sent, cp):
    return cp.parse(sent)

def print_dtjjnn_chunks(list_of_trees):
    for tree in list_of_trees:
        for subtree in tree.subtrees():
            if subtree.label() == "NP" and subtree[0][1] == 'DT' and subtree[1][1] == 'JJ' and subtree[2][1] == 'NN' and len(subtree) == 3:
                # A DT followed by ADJ followed by NN
                print(subtree.leaves())

"""
## 2d)
Create a custom rule for a combination of 3 or more tags, similarly to task c).

Do you see any practical uses for chunking with this rule, or in general?
"""

def print_n_chunks(list_of_trees):
    for tree in list_of_trees:
        for subtree in tree.subtrees():
            if subtree.label() == "NP" and subtree[0][1] == 'DT' and subtree[1][1] == 'JJ' and len(subtree) > 3:
                print(subtree.leaves())


def print_nn_chunks(list_of_trees, n):
    for tree in list_of_trees:
        for subtree in tree.subtrees():
            if subtree.label() == "NP" and len(subtree) >= n: # A DT followed by ADJ has to followed by (optionally)
                # more modifiers and ended with a noun
                print(subtree.leaves())

if __name__ == '__main__':
    cp = task2a()  # task 2a

    # 2b
    filename = 'starlink.txt'
    starlink_tagged = get_pos_tags_from_file(filename)
    # for sent in starlink_tagged:
    #    print(sent, end="\n")

    # 2c
    list_of_trees = get_descriptive_nouns(starlink_tagged, cp)
    print_dtjjnn_chunks(list_of_trees)
    print("")

    # 2d
    print_n_chunks(list_of_trees)
