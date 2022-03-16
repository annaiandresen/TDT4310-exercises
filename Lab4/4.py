"""
# Exercise 4 - Tweet like Trump! Now that he's banned
Using the provided file "realDonaldTrump.json", you will build a language model to generate Trump-esque tweets using n-grams.

Hint: make use of "padded_everygram_pipeline" supported in nltk.lm. This creates all ngrams up to the specified N-parameter with padding:

Example:
```
('<s>',)
('<s>', '<s>')
('<s>', '<s>', '<s>')
('<s>', '<s>', '<s>', '<s>')
('<s>', '<s>', '<s>', '<s>', 'i')
('<s>',)
('<s>', '<s>')
('<s>', '<s>', '<s>')
('<s>', '<s>', '<s>', 'i')
('<s>', '<s>', '<s>', 'i', 'am')
('<s>',)
('<s>', '<s>')
('<s>', '<s>', 'i')
('<s>', '<s>', 'i', 'am')
('<s>', '<s>', 'i', 'am', 'asking')
('<s>',)
('<s>', 'i')
('<s>', 'i', 'am')
('<s>', 'i', 'am', 'asking')
('<s>', 'i', 'am', 'asking', 'for')
('i',)
('i', 'am')
('i', 'am', 'asking')
('i', 'am', 'asking', 'for')
('i', 'am', 'asking', 'for', 'everyone')
```
"""
import string
import nltk
import json
import pickle
import os
from nltk import lm
from nltk.lm.preprocessing import padded_everygram_pipeline



# Finish the train_model function
def train_model(data, pickle_path="model.pkl"):
    tokenizer = nltk.TweetTokenizer()
    tokenized = [tokenizer.tokenize(sent.lower()) for sent in data]
    n = 5
    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized)
    model = lm.models.MLE(n)  # we only have to specify highest ngram, nice simple solution<3
    model.fit(train_data, padded_vocab)
    # save the model if you want to :-) then we can load it in the next step without retraining!
    with open(pickle_path, "wb") as fp:
        pickle.dump(model, fp)
    return model


def generate_sentence(model, txt):
    """
    :param txt: string that the output sentence should start with
    :param model: The Language model that will generate a sentence
    Generates Trump-esque tweet that starts with '"+txt+"'\n\n")
    :return: a new Trump-esque sentence as string
    """
    txt = nltk.word_tokenize(txt.lower())
    while True:
        next_word = model.generate(text_seed=txt, random_seed=42)
        if next_word == '</s>':
            break
        txt.append(next_word)

    def filter_fn(txt):
        no_http = "http" not in txt
        some_other_rule = True

        return no_http or some_other_rule

    return " ".join([t for t in txt if filter_fn(t)])


"""
## 4b)
Create a grammar to chunk some typical trump statements.

There are multiple approaches to this. One way would be to use your own input to the model and look at the resulting 
outputs and their POS tags. Another possible approach is to use the training data to group together e.g. 5-grams of 
POS tags to look at the most frequently occurring POS tag groupings. The aim is to have a chunker that groups 
utterances like "so sad", "make america great again!" and so forth. 

Show your results using the outputs from your model with these inputs: 
- "clinton will"
- "obama is"
- "build a"
- "so sad"
"""

def find_pos_tag_combinations(sentence):
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)


def get_only_pos_tags(tagged_sentence):
    pos_tags = [pos for (w, pos) in tagged_sentence]
    return ' '.join(pos_tags)


if __name__ == '__main__':
    model = None
    # nltk.help.upenn_tagset()
    pickle_path = "model.pkl"
    print("Checking if model exists...")
    if os.path.exists(pickle_path):
        print("Model exists, loading model...")
        with open(pickle_path, "rb") as fp:
            model = pickle.load(fp)
    else:
        print("No model found, generating new model...")
        with open("realDonaldTrump.json", encoding='UTF-8') as fp:
            tweets = json.load(fp)
        texts = list(map(lambda x: x.get("text"), tweets))
        model = train_model(texts)

    words_to_generate_trump_sentences = ["clinton will", "obama is", "so sad", "build a"]

    # Generate sentences and remove punctuation
    generated_sentences = [generate_sentence(model, w) for w in words_to_generate_trump_sentences]
    generated_sentences = [w.translate((str.maketrans('', '', string.punctuation))) for w in generated_sentences]

    # POS tag and find combinations
    pos_tagged_examples = [find_pos_tag_combinations(sent) for sent in generated_sentences]
    pos_tag_combinations = [get_only_pos_tags(sent) for sent in pos_tagged_examples]

    print("\nPrinting sentences with their respective POS tags")
    for i in range(len(pos_tag_combinations)):
        print('')
        print(generated_sentences[i])
        print(pos_tag_combinations[i])

    # Creating Trump grammar
    trump_grammar = r"""
            CHUNKS:
            {<RB><JJ><NN>*<NNS>?} # So sad .., not presidental material. Optionally add 'thanks', congrats' etc.
            {<VB><JJ>} # Look foolish!
            {<IN><DT><JJ><NN>}
            {<WP><VBD><PRP><RB>*}
            {<NN>*<JJ><RB>?} # Make america great again!, not sure why 'make' is NN...
            {<VB><DT><NN>} # Build a wall

            """

    # Creating a regex parser with the specified grammar
    cp = nltk.RegexpParser(trump_grammar)

    # Printing chunks
    for sent in pos_tagged_examples:
        print(cp.parse(sent), end="\n\n")

