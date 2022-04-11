"""Exercise 3 - Sentiment Analysis with Keras
Implement a neural network with LSTM with dropout. Download a commonly used dataset for sentiment
analysis: IMDB Reviews https://www.tensorflow.org/datasets/catalog/imdb_reviews. This is found
in the “tensorflow-datasets” package, installable with pip, or via keras https://keras.io/api/datasets/
imdb/.
1. Instantiate the dataset and clean it as you see fit. Encode the labels as you want (e.g. 0 for negative).
2. Setup a shallow feed-forward NN to give your setup an initial benchmark (no LSTM)
3. Setup a NN with LSTM. Feel free to follow the code from the ATAP book. If you wish, you can
implement GloVe embeddings in the initial layer.
4. Test out a few different model setups, different dropouts, etc. This is heavily based on your available
hardware.
5. Verify the model on a sample of texts from Chamber of Secrets. Explain your initial thoughts and
reflect on how you could create a dataset more suited to the domain of fantasy books."""