import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
        else:
            tokens_filtered.append("")
    return tokens_filtered


def tokenize(string):
    return nltk.word_tokenize(string)


def lemmatize(tokens):
    terms = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return terms


def process_string(string):
    string = remove_non_alphanumeric_and_lowercase(string)
    tokens_raw = tokenize(string)
    tokens = remove_stop_words(tokens_raw)
    terms = lemmatize(tokens)
    return terms


def process_documents(documents):
    tokens = []
    for n in range(len(documents)):
        source_tokens = process_string(documents[n].source)
        title_tokens = process_string(documents[n].title)
        description_tokens = process_string(documents[n].description)
        content_tokens = process_string(documents[n].content)

        documents[n].set_processed_strings(source_tokens, title_tokens, description_tokens,
                                           content_tokens)

        tokens.extend(source_tokens)
        tokens.extend(title_tokens)
        tokens.extend(description_tokens)
        tokens.extend(content_tokens)

    tokens_unique = np.unique(np.asarray(tokens))
    tokens_unique = np.delete(tokens_unique, np.where(tokens_unique == ""))
    tokens_unique_frequency = []
    for token_unique in tokens_unique:
        indices = [i for i, token in enumerate(tokens) if token == token_unique]
        tokens_unique_frequency.append(len(indices))

    terms = TermList()
    for n, token in enumerate(tokens_unique):
        terms.add_term(token, tokens_unique_frequency[n])

    return terms
