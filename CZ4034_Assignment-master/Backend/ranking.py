import math
import numpy as np


def get_document_tf_idf(documents, terms):
    document_term_tf_idf_weight = []
    document_tfidf_score = []
    for term in terms:
        # each term (idf)
        idf = math.log((len(documents)/term.document_fequency), 2)
        term.set_term_idf(idf)
    for document in documents:
        for term_1 in terms:
            tf = term_1.get_positional_index(document.id)
            if tf is None:
                document_term_tf_idf_weight.append(0)
            else:
                document_tf = 1 + math.log(len(tf), 2)
                document_term_tf_idf_weight.append(document_tf * term.get_term_idf)
        term_length = np.asarray(document_term_tf_idf_weight)
        term_length = term_length[term_length != 0]
        term_length = term_length[term_length ** 2]
        doc_length = math.sqrt(sum(term_length))
        for weight in document_term_tf_idf_weight:
            document_tfidf_score.append(weight/doc_length)
        document.set_document_tfidf_score(document_tfidf_score)


def calculate_query_tf_idf(query, documents):
    if len(query) < 2:
        return
    unique_terms, count = np.unique(np.asarray(query), return_counts=True)
    query_tf_idf_weight = []
    i = 0
    for term in unique_terms:
        query_tf = 1 + math.log(count[i], 2)
        query_tf_idf_weight.append(query_tf * term.get_term_idf())
        ++i
    query_length = 0
    query_tfidf = []
    for weight in query_tf_idf_weight:
        query_length += math.pow(weight, 2)
    query_length = math.sqrt(query_length)
    for weight in query_tf_idf_weight:
        query_tfidf.append(weight/query_length)
    document_term_tfidf_score = []
    for document in documents:
        j = 0
        tfidf_score = 0
        for term in unique_terms:
            document_term_tfidf = document.get_document_tfidf_score()
            tfidf_score += query_tfidf[j] * document_term_tfidf[term.id]
            ++j
        document_term_tfidf_score.append(tfidf_score)
    return document_term_tfidf_score


