import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from Backend.Objects.TermList import TermList

stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
regex = re.compile('[^a-zA-Z0-9 ]')


def remove_non_alphanumeric_and_lowercase(string):
    string = string.lower()
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
    string = remove_non_alphanumeric_and_lowercase(string)
    tokens = tokenize(string)
    tokens = remove_stop_words(tokens)
    terms = lemmatize(tokens)
    return terms


def process_documents(documents):
    tokens = []
    for n in range(len(documents)):
        tokens.extend(process_string(documents[n].title))
        tokens.extend(process_string(documents[n].description))
        tokens.extend(process_string(documents[n].content))
        tokens.extend(process_string(documents[n].source))
    tokens = np.asarray(tokens)
    tokens_unique = np.unique(tokens)

    tokens_unique_documents = []
    tokens_unique_frequency = []
    for token_unique in tokens_unique:
        indices = [i for i, token in enumerate(tokens) if token == token_unique]
        tokens_unique_documents.append(indices)
        tokens_unique_frequency.append(len(indices))

    terms = TermList()
    for n, token in enumerate(tokens_unique):
        terms.add_term(token, tokens_unique_documents[n], tokens_unique_frequency[n])
    return terms




