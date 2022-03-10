"""
# Lab 3
"""
import nltk
from sklearn.model_selection import train_test_split
import random

# feel free to import from modules of sklearn and nltk later
# e.g., from sklearn.model_selection import train_test_split


"""
## Exercise 1 - Gender detection of names
In NLTK you'll find the corpus `corpus.names`. A set of 5000 male and 3000 female names.
1) Select a ratio of train/test data (based on experiences from previous labs perhaps?)
2) Build a feature extractor function
3) Build two classifiers:
    - Decision tree
    - Naïve bayes
    
Finally, write code to evaluate the classifiers. Explain your results, and what do you think would change if you 
altered your feature extractor? """


class GenderDataset:
    def __init__(self):
        self.names = nltk.corpus.names
        self.data = None
        self.build()

    def make_labels(self, gender):
        """
        this function is to help you get started
        based on the passed gender, as you can fetch from the file ids,
        we return a tuple of (name, gender) for each name
        
        use this in `build` below, or do your own thing completely :)
        """
        return [(n, gender) for n in self.names.words(gender + ".txt")]

    def build(self):
        """
        combine the data in "male" and "female" into one
        remember to randomize the order
        """
        female = self.make_labels('female')
        male = self.make_labels('male')
        labeled_names = female + male

        # Shuffle
        random.shuffle(labeled_names)
        self.data = labeled_names

    def split(self, ratio):
        return train_test_split(self.data, train_size=ratio)


class Classifier:
    def __init__(self, classifier):
        if classifier == 'dt':
            self.classifier = nltk.DecisionTreeClassifier
        else:
            self.classifier = nltk.NaiveBayesClassifier
        self.model = None

    def train(self, data):
        trained = self.classifier.train(data)
        self.model = trained

    def test(self, data):
        return nltk.classify.accuracy(self.classifier, data)

    def train_and_evaluate(self, train, test):
        self.train(train)
        return nltk.classify.accuracy(self.classifier, test)

    def show_features(self):
        return self.classifier.show_most_informative_features(5)


class FeatureExtractor:
    def __init__(self, data):
        self.data = data
        self.features = []
        self.build()

    @staticmethod
    def text_to_features(name):
        return {
            "name": name,
            "last_letter": name[-1],
            "length": len(name)
        }

    def gender_features(self, tup):
        """
        Some code taken from nltk ch 6.1.1
        :return: a dictionary with features (last letter, word length)
        """
        return {'suffix': tup[0][-1].lower(), 'suffix2': tup[0][-2:].lower(), 'length': len(tup[0])}

    def build(self):
        for tup in self.data:
            features = [self.gender_features(tup), tup[1]]
            self.features.append(features)


if __name__ == '__main__':
    split_ratio = 0.9  # this seemed like a good ratio from last lab
    train, test = GenderDataset().split(ratio=split_ratio)

    train_set = FeatureExtractor(train).features
    test_set = FeatureExtractor(test).features

    # Could not make my classifier class work, so had to go with this solution
    dt_classifier = nltk.DecisionTreeClassifier.train(train_set)
    print("Printing accuracy for Decision Tree Classifier:", nltk.classify.accuracy(dt_classifier, test_set))

    nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Printing accuracy for Naive Bayes Classifier:", nltk.classify.accuracy(nb_classifier, test_set))

    nb_classifier.show_most_informative_features(5)
