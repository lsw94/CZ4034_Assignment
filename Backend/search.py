import time

import Backend.crawler as crawler
import Backend.lemmatizer as lemmatizer
from Backend.Objects.TermDocumentSimilarity import TermDocumentSimilarity

documents = None
terms = None


def initialize():
    start_t = time.time()
    global documents, terms
    documents, terms = crawler.query_news(False)
    end_t = time.time()
    print("Initialize: %.2fs\n" % (end_t - start_t))


def search_string(string):
    start_t = time.time()
    global documents, terms
    search_terms = lemmatizer.process_string(string)
    found_terms = []
    for term in search_terms:
        found = terms.get_term(term)
        found_terms.append(found)
    similar_positional_index = find_similar_documents(found_terms)
    end_t = time.time()
    print("Search: %.2fs" % (end_t - start_t))
    print("------Results-------")
    for n, spis in enumerate(similar_positional_index):
        print("------" + str(n + 1) + "------")
        for spi in spis:
            for m, sdi in enumerate(spi.similar_document_ids):
                if spi.similar_document_in_order[m] is True:
                    print(documents.get_document(sdi).title)
                    # print(sdi)
            print("...")


def find_similar_documents(found_terms):
    number_of_terms = len(found_terms)
    term_similarity_list_num_words = []
    for n in range(1, number_of_terms + 1):
        term_similarity_list = []
        for m in range(len(found_terms) - n + 1):
            terms = []
            for o in range(n):
                terms.append(found_terms[m + o])
            term_similarity_list.append(TermDocumentSimilarity(terms))
        term_similarity_list_num_words.append(term_similarity_list)
    return term_similarity_list_num_words


initialize()
search_string("Christchurch mosque shootings")
