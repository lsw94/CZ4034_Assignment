import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z0-9 ]')


def remove_non_alphanumeric(string):
    return regex.sub("", string)


def remove_stop_words(tokens):
    tokens_filtered = []
    for token in tokens:
        if token not in stop_words:
            tokens_filtered.append(token)
    return tokens_filtered


def tokenize(string):
    return nltk.word_tokenize(string)


def lemmatize(tokens):
    terms = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return terms


def process_string(string):
    string = remove_non_alphanumeric(string)
    tokens = tokenize(string)
    tokens = remove_stop_words(tokens)
    terms = lemmatize(tokens)
    return terms


def process_documents(documents):
    

