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

from nltk.corpus import brown, nps_chat
from Lab2_Exercise1 import n_most_frequent_tags

if __name__ == '__main__':
    brown_tags = brown.tagged_words()
    most_common_tag = n_most_frequent_tags(1, brown_tags)[0][0]

    print(most_common_tag+" IS THE MOST COMMON TAG IN THE BROWN CORPUS")

    nps_tags = nps_chat.tagged_words()
    most_common_tag = n_most_frequent_tags(1, nps_tags)[0][0]
    print(most_common_tag + " IS THE MOST COMMON TAG IN THE NPS_CHAT CORPUS")

