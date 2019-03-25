import Backend.crawler as crawler
import Backend.lemmatizer as lemmatizer
import Backend.ranking as ranking
from Backend.Objects.TermDocumentSimilarity import TermDocumentSimilarity
from Backend.Objects.TermList import TermList
from gensim import models
from gensim import corpora
import numpy as np
documents = None
terms = None


def initialize():
    global documents, terms
    documents, terms = crawler.query_news(True)
    precompute_tfidf()


def search_string(string):
    global documents, terms
    search_terms = lemmatizer.process_string(string)
    found_terms = []
    for term in search_terms:
        found = terms.get_term(term)
        found_terms.append(found)
    similar_positional_index = find_similar_documents(found_terms)
    ranking_tfidf(found_terms)
    for n, spis in enumerate(similar_positional_index):
        print("------" + str(n+2) + "------")
        for spi in spis:
            for m, sdi in enumerate(spi.similar_document_ids):
                if spi.similar_document_in_order[m] is True:
                    print(sdi)
            print("...")


def find_similar_documents(found_terms):
    number_of_terms = len(found_terms)
    term_similarity_list_num_words = []
    for n in range(2, number_of_terms+1):
        term_similarity_list = []
        for m in range(len(found_terms)-n+1):
            terms = []
            for o in range(n):
                terms.append(found_terms[m+o])
            term_similarity_list.append(TermDocumentSimilarity(terms))
        term_similarity_list_num_words.append(term_similarity_list)
    return term_similarity_list_num_words


def precompute_tfidf():
    global documents, terms
    ranking.get_document_tf_idf(documents, terms)


def ranking_tfidf(found_terms):
    global documents
    ranking.calculate_query_tf_idf(found_terms, documents)


initialize()
search_string("The investigation reportedly centered")
