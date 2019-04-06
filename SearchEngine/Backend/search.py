import math
import time

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import nltk
# from flask import request
import SearchEngine.Backend.crawler as crawler
import SearchEngine.Backend.lemmatizer as lemmatizer
from SearchEngine.Backend.Objects.TermDocumentSimilarity import TermDocumentSimilarity
from SearchEngine.Backend.categorize import calculate_fscore

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
    similar_positional_index, relevent_documents_ids = find_similar_documents(found_terms)
    documents_tfidf_dict = get_document_query_tfidf_score(found_terms, relevent_documents_ids)
    end_t = time.time()
    doc_ids = []
    tfidf_score = []

    if documents_tfidf_dict is None:
        document_return = []
        for id in relevent_documents_ids:
            doc = documents.get_document(id)
            document_return.append(doc)
        return document_return

    for k, value in documents_tfidf_dict.items():
        doc_ids.append(k)
        tfidf_score.append(value)
    tfidf_score, doc_ids = zip(*sorted(zip(tfidf_score, doc_ids)))
    print("Search: %.2fs" % (end_t - start_t))
    document_return = []
    for id in reversed(doc_ids):
        doc = documents.get_document(id)
        document_return.append(doc)
        print(doc.title)
    return document_return

    # print("------Results-------")
    # for n, spis in enumerate(similar_positional_index):
    #     print("------" + str(n + 1) + "------")
    #     for spi in spis:
    #         for m, sdi in enumerate(spi.similar_document_ids):
    #             if spi.similar_document_in_order[m] is True:
    #                 print(documents.get_document(sdi).title)
    #                 # print(sdi)
    #         print("...")


def find_similar_documents(found_terms):
    number_of_terms = len(found_terms)
    term_similarity_list_num_words = []
    relevant_documents_id = []
    for n in range(1, number_of_terms + 1):
        term_similarity_list = []
        for m in range(len(found_terms) - n + 1):
            terms = []
            for o in range(n):
                if n == 1:
                    typ = nltk.pos_tag([found_terms[m + o].term])[0][1]
                    if "VB" in typ:
                        continue
                terms.append(found_terms[m + o])
            tds = TermDocumentSimilarity(terms)
            term_similarity_list.append(tds)
            relevant_documents_id.extend(tds.similar_document_ids)
        term_similarity_list_num_words.append(term_similarity_list)
    return term_similarity_list_num_words, list(np.unique(np.asarray(relevant_documents_id)))


def get_document_query_tfidf_score(query, relevant_document_ids):
    query_filtered = [x for x in query if x is not None]
    if len(query_filtered) < 2:
        return None
    query_text = []
    for q in query_filtered:
        query_text.append(q.term)
    query_terms, query_terms_count = np.unique(np.asarray(query_text), return_counts=True)
    query_terms_tfidf = []
    query_terms_tfidf_norm = []
    query_terms_id = []
    for i, query_term in enumerate(query_terms):
        query_tf = 1 + math.log(query_terms_count[i], 10)
        for t_o in query_filtered:
            if t_o.term == query_term:
                query_object = t_o
                break
        query_terms_tfidf.append(query_tf * query_object.idf)
        query_terms_id.append(query_object.id)

    query_normalization = 0
    for tfidf in query_terms_tfidf:
        query_normalization += math.pow(tfidf, 2)
    query_normalization = math.sqrt(query_normalization)

    for tfidf in query_terms_tfidf:
        query_terms_tfidf_norm.append(tfidf / query_normalization)

    document_tfidf_scores = {}
    for doc_id in relevant_document_ids:
        document = documents.get_document(doc_id)
        doc_tfidf_score = 0
        for n, query_term in enumerate(query_terms):
            ttfidf_norm = document.tfidfs.get_tfidf_norm_of_term_id(query_terms_id[n])
            if ttfidf_norm is not None:
                term_tfidf = query_terms_tfidf_norm[n] * ttfidf_norm
                doc_tfidf_score += term_tfidf
        document_tfidf_scores[doc_id] = doc_tfidf_score
    return document_tfidf_scores


initialize()
calculate_fscore(documents)
# categorize_document()
# search_string("Donald Trump America Safety")
# search_string("Christchuch shooting")
# search_string("MH370 Found Malaysia")
