'''3 - Building your own corpus
A real-world project needs updated data. Not all services provide proper APIs to access their data,
and must thus be mined. In this task, you need to build a web scraper for your favorite news site,
separating each article. Norwegian sites can also be used, you can find a list of stopwords here: https:
//gist.github.com/kmelve/8869818.
The following data should be extracted:
1. Headline
2. Ingress
3. Several sentences from the text body
4. Published date
5. Topic, if it is easily accessible. Several news sites allow you to browse news by category.
6. URL
Try to use at least 50 articles for somewhat interesting results. However, feel free to create your own huge
dataset for future use! The data may be saved in any format of your choice (e.g., .txt, .json), as long as
you are able to load the content into a Python program. If you managed to separate on topics, feel free
to do the tasks d-f below separately for each topic (easy once you have the code for one)
(a) Build the corpus. Chapter 2, section 1.9 (NLTK) and Chapter 2 (ATAP) explain of how this can be
done using NLTK (e.g., using a custom HTMLCorpusReader), but you are free to load the text as
you prefer. As long as the code is readable!
(b) Clean the data using what you have just learned. There are likely several new problems to handle.
Select one and explain how you would solve it – no need for code
(c) Separate the text body (sentences) into tokens (words) by splitting on spaces. Find and implement
one improvement on this tokenization process
(d) Print the 10 most common words
(e) Create bigrams from the texts that do not contain stopwords. Print the 10 most common bigrams
(f) Find the headline that contains the highest number of most frequently used words (from the common
words in task d, but not limited to top 10). If you want to, you are free to explore more advanced
techniques if desired, e.g., generating a new headline based on the frequency of words, etc.
'''