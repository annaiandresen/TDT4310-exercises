"""Exercise 1 - Named entity recognition with Harry Potter
You will find the text for ‘Chamber of Secrets‘ on GitHub, which you are to use for all tasks in this
exercise. Load the file and separate chapters into a proper data structure. Each chapter should be a single
string of text. For visualization purposes, use a few sentences from chapter 1, or whatever subset which
makes sense.
1. With your knowledge of POS tags and chunking, attempt to fetch entities from text. Print out the
NLTK trees.
2. Chunk pronouns in the same text, and reflect on how you would attack the problem of attributing
a pronoun to an entity.
3. Use NLTKs built-in functionality for NER. Print the top 10 most frequent entities in chapter 2.
4. Now implement spaCy. See code on GitHub. Perform the previous task and discuss the results.
5. Using the results from spaCy, plot the frequency of characters (e.g. PERSONs) in the entire book.
6. Visualize the dependency tree using spaCy, discuss how you could utilize the results to improve upon
what you figured out in task 2."""

import nltk
import random
import spacy
from spacy import displacy
from spacy.cli import download
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from collections import defaultdict


def separate_into_chapters(file) -> list:
    """
    Load the file and separate chapters into a proper data structure.
    Splits the input string on 'CHAPTER'
    :param file: name of file
    :return: a list with chapters
    """
    with open(file, 'r') as f:
        sample = f.read()
    chapters = sample.split("CHAPTER ")[1:]
    chapters = [chapter.replace('"', '').replace("\n", " ").strip() for chapter in chapters]
    f.close()
    return chapters


class Parser:
    def __init__(self, text):
        self.text = text
        self.parser = self.create_chunk_parser()
        self.tags = self.get_pos_tags()

    @staticmethod
    def create_chunk_parser():
        grammar = r"""
            ENITITY: {(<NP>|<DT>|<NNP>)(<NNP>|<NNS>)} #Mr/ms... OR The Dursleys, cant catch "the NN" 
                     {<NNP.*>} # Names
            PRONOUN: {<PRP>} # Pronouns, but incorrectly labels 'it'...
            """
        return nltk.RegexpParser(grammar)

    def get_pos_tags(self):
        text = nltk.sent_tokenize(self.text)
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in text]
        return [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    def generate_trees(self):
        return [self.parse_sent(sent) for sent in self.tags if self.tags is not None]

    def parse_sent(self, sent):
        return self.parser.parse(sent)

    def generate_ner(self):
        return [nltk.ne_chunk(sent, binary=True) for sent in self.tags]

    def get_entities(self):
        entities = []
        for tree in self.generate_ner():
            for subtree in tree.subtrees():
                if subtree.label() == "NE":
                    entities.append(" ".join([w for w, pos in list(subtree)]))
        return entities

    def get_most_common_entity(self, n=10):
        entities = self.get_entities()
        return nltk.FreqDist(entities).most_common(n)


class Ner:
    def __init__(self, text):
        # download('en_core_web_sm')
        self.text = text

        # Load English tokenizer, tagger, parser and NER
        self.nlp = spacy.load("en_core_web_sm")
        self.doc = self.nlp(text)

    def get_entities(self):
        return self.doc.ents

    def render_entities(self, i=0):
        sent = list(self.doc.sents)[:i]
        return displacy.serve(sent, style="ent")

    def render_dependencies(self, i=None):
        sent = list(self.doc.sents) if i is None else list(self.doc.sents)[i]
        return displacy.serve(sent, style="dep")

    def sort_entities(self):
        dict = {}
        entities = self.get_entities()
        for entity in entities:
            self.set_key(dict, entity.label_, entity.text)
        return dict

    @staticmethod
    def set_key(dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = value
        elif type(dictionary[key]) == list:
            dictionary[key].append(value)
        else:
            dictionary[key] = [dictionary[key], value]


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer')


def create_masked_wc(data):
    wc = WordCloud(
        max_words=150,
        max_font_size=70,
        contour_width=0,
        width=1000,
        height=1000)

    wc.generate(" ".join(data))

    # show
    plt.figure(figsize=(15, 15))
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def flat_map(xs):
    ys = []
    for x in xs:
        ys.extend(x)
    return ys


def clean_sent(sent):
    escapes = ['\n', '\t', '\r']

    def valid(tok):
        return tok not in escapes

    for e in escapes:
        sent = sent.replace(e, ' ')
    return sent


def get_sentences(data):
    MIN_LEN = 20
    sents = []
    for sent in data:
        # clean the raw text
        cleaned = clean_sent(sent).strip()

        # parse possible sub-sentences using spacy
        doc = nlp(cleaned)
        sub_sents = [s.text for s in doc.sents if len(s.text) > MIN_LEN]
        sents.append(sub_sents)
    return flat_map(sents)


def ent_info(idx, ent):
    return {
        "sent_idx": idx,
        "start": ent.start_char,
        "end": ent.end_char,
        "label": ent.label_
    }


if __name__ == '__main__':
    # Part 1
    chapters = separate_into_chapters(
        'chamber_of_secrets.txt')  # Lazy implementation as it removes the word 'chapter', could have used regex here.
    chapter_one = chapters[0][24:]  # Cutting off chapter title

    # Parser
    cp = Parser(chapter_one)

    # Trees
    trees = cp.generate_trees()
    cp_ner = cp.generate_ner()

    # Printing trees from random snippets
    random_integer = random.randint(0, len(cp_ner) - 3)
    for i in range(random_integer, random_integer + 1):
        print("Printing tree produced with chunker \n", trees[i], end="+\n")
        print("Printing tree produced with nltk named entity recognizer: \n", cp_ner[i])

    # Part 2
    chapter_two = chapters[1][25:].strip()
    cp_2 = Parser(chapter_two)
    print("Printing 10 most commmon entities in chapter 2")
    print(cp_2.get_most_common_entity())
    # ner = Ner(chapter_one)
    # print(ner.render_entities(2))
    # print(ner.render_dependencies(2))
    # d = ner.sort_entities()
    # print(d)
    # sents = get_sentences(chapters)
    # ents = defaultdict(list)
    # only_ent_names = []
    # for idx, sent in enumerate(sents):
    #     doc = nlp(sent)
    #     for ent in doc.ents:
    #         ents[str(ent).lower()].append(ent_info(idx, ent))
    #         only_ent_names.append(str(ent))
    #
    # create_masked_wc(only_ent_names)
