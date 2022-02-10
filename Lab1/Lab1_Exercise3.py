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
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk import bigrams
from norwegian_stopwords import generate_norwegian_stopwords
from Lab1_Exercise2 import clean_words, find_n_most_common_words, print_words

driver = webdriver.Chrome(ChromeDriverManager().install())


def scrape_article(link):
    """
    Scrapes a specific NRK article using selenium
    :param article link:
    :return article body as string in a list, date posted as string:
    """
    driver.get(link)
    whole_article = driver.page_source
    newSoup = BeautifulSoup(whole_article, features="html.parser")

    # empty list to store snippets from article body
    snippets = []

    for data in newSoup.find("div", attrs={'class', 'article-body'}):
        tag = data.name
        if tag == 'p':
            snippets.append(data.text)

    dateTag = newSoup.find('time')
    if dateTag.name == 'time':
        try:
            date = dateTag['datetime']
        except:
            date = "no date"
    return snippets, date


def generate_corpus():
    """
    Generates a corpus in dictionary format
    The corpus exists of articles from NRK marked 'urix'
    The key is the article ID
    The value is a dictionary consisting of date, headline, preamble, link, text
    :return: a dictionary item with news
    """
    driver.get('https://www.nrk.no/urix')

    # Need to click '+' to display more articles....
    # 30 clicks = ~60 articles
    for i in range(30):
        driver.find_element(By.CSS_SELECTOR, ".nrk-pagination>.nrk-button:only-child").click()

    # Raw HTML
    content = driver.page_source

    # Parse the HTML in bs4
    soup = BeautifulSoup(content, features="html.parser")

    news = dict()
    for data in soup.find_all(class_='autonomous lp_plug'):
        tag = data.name
        if tag == 'a':
            link = data['href']
            article_id = data['data-id']
            headline = data.find("h2").text
            preamble = data.find("p", attrs={'class', 'plug-preamble'}).text

            text, date = scrape_article(link)
            serialized = serialize(date, headline, preamble, link, text)
            news[article_id] = serialized

    driver.quit()
    return news


def serialize(date, headline, preamble, link, body):
    data = {
        'date': date,
        'headline': headline,
        'preamble': preamble,
        'link': link,
        'body': body
    }
    return data


def write_to_file(dic):
    with open("news.json", 'w', encoding='utf8') as file:
        serialized = json.dumps(dic, sort_keys=True, ensure_ascii=False, indent=4)
        json.dump(serialized, file)
    file.close()

def deserialize_json():
    with open("news.json", 'r') as file:
        corpus = json.load(file)
        corpus = json.loads(corpus)
    file.close()
    return corpus


def combine_all_words(news):
    words = []
    for key in news:
        element = news.get(key)
        body_as_string = ''.join(element['body'])
        words.append(body_as_string)
    return ''.join(words)


def find_n_most_common_words_from_raw(words, n):
    # Creating tokens
    tokens = word_tokenize(words)

    # Remove stopwords, punctuation and whitespace
    words_without_stopwords = clean_words(tokens, 'no')
    return find_n_most_common_words(words_without_stopwords, n)


def generate_bigrams(text):
    li = [w.lower() for w in text.split(" ") if w not in generate_norwegian_stopwords()]
    return bigrams(li)


def get_all_headlines(dic):
    words = []
    for key in dic:
        element = dic.get(key)
        headline = ''.join(element['headline'])
        words.append(headline)
    return words


def get_most_common_words(most_common):
    """
    Get the most common words from a fdist list created from bigrams.
    :param most_common: n most common words in a list with tuples
    :return: a list with the most frequent words.
    """
    return [w for w, freq in most_common]


def find_headline_with_most_frequent_word(headlines, words):
    freq_headline = ''
    occurrences = 0
    for headline in headlines:
        count = count_occurrences(words, headline)
        if count > occurrences:
            occurrences = count
            freq_headline = headline
    return freq_headline, occurrences


def count_occurrences(words, sentence):
    i = 0
    lst = sentence.lower().split()
    for word in words:
        i += lst.count(word)
    return i


def print_most_common_bigrams(list_bigram):
    i = 1
    for tup in list_bigram:
        bigram = tup[0]
        occurrence = tup[-1]
        print(str(i) + ": (" + bigram[0] + "," + bigram[-1] + ") with " + str(occurrence) + " occurrences")
        i += 1


if __name__ == '__main__':
    news = generate_corpus()

    # Write the news to a file called news.json in json format
    write_to_file(news)

    # Finding 10 most common words in data set
    news = deserialize_json()
    text = combine_all_words(news)
    most_common = find_n_most_common_words_from_raw(text, 10)

    number_of_articles = len(deserialize_json())

    print("Scraping " + str(number_of_articles) + " articles resulted in these 10 most common words: ")
    print_words(most_common)

    # Finding bigrams
    bigrams = generate_bigrams(text)

    print("\nThe set had these 10 most common bigrams")
    most_common_bigrams = find_n_most_common_words(bigrams, 10)
    print_most_common_bigrams(most_common_bigrams)

    # Working with headlines
    headlines = get_all_headlines(news)
    most_common_words = get_most_common_words(most_common)
    frequent_headline, count = find_headline_with_most_frequent_word(headlines, most_common_words)

    print("\nThe headline that contains the highest number of most frequently used words:".upper())
    print("'" + frequent_headline + "'"" with " + str(count) + " words")
