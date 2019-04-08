import math
import re
import time
from nltk.corpus import wordnet
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
# from pattern.en import suggest
from spellchecker import SpellChecker

import SearchEngine.Backend.crawler as crawler
import SearchEngine.Backend.lemmatizer as lemmatizer
from SearchEngine.Backend.Objects.TermDocumentSimilarity import TermDocumentSimilarity
from SearchEngine.Backend.lemmatizer import tokenize

documents = None
terms = None

# pattern = re.compile(r"(.)\1{2,}")
spell = SpellChecker()
num_words_search = 2


def initialize():
    start_t = time.time()
    global documents, terms
    documents, terms = crawler.query_news(False)
    end_t = time.time()
    print("Initialize: %.2fs\n" % (end_t - start_t))


def search_string(string):
    start_t = time.time()
    global documents, terms
    global num_words_search
    one_word_seach = False
    fixed_search = False
    if "\"" in string:
        start_end_position = [pos for pos, char in enumerate(string) if char == "\""]
        if len(start_end_position) == 2:
            string = string[start_end_position[0] + 1:start_end_position[1]]
            fixed_search = True
    search_terms = lemmatizer.process_string(string)

    print("Original Search: ")
    print(string)
    suggested_terms = spelling_check(tokenize(string))
    if suggested_terms is None:
        suggested_search = ""
    else:
        suggested_search = " ".join(suggested_terms)
        print("Corrected Search: ")
        print(suggested_search)
        if suggested_search.lower().strip() == string.lower().strip():
            suggested_search = ""

    if len(search_terms) < num_words_search:
        one_word_seach = True
    found_terms = []
    for term in search_terms:
        found = terms.get_term(term)
        found_terms.append(found)
    similar_positional_index, relevent_documents_ids = find_similar_documents(found_terms, fixed_search, one_word_seach)
    if fixed_search:
        similar_pi = similar_positional_index[len(search_terms) - num_words_search][0]
        in_order_position = [pos for pos, io in enumerate(similar_pi.similar_document_in_order) if io is True]
        relevent_documents_ids = list(np.asarray(similar_pi.similar_document_ids)[in_order_position])
    documents_tfidf_dict = get_document_query_tfidf_score(found_terms, relevent_documents_ids)

    doc_ids = []
    tfidf_score = []

    if documents_tfidf_dict is None or len(documents_tfidf_dict) == 0:
        document_return = []
        for id in relevent_documents_ids:
            doc = documents.get_document(id)
            document_return.append(doc)

        end_t = time.time()
        print("Search: %.4fs" % (end_t - start_t))
        print("Number of document: " + str(len(document_return)))
        return document_return, suggested_search, (end_t - start_t)

    for k, value in documents_tfidf_dict.items():
        doc_ids.append(k)
        tfidf_score.append(value)
    tfidf_score, doc_ids = zip(*sorted(zip(tfidf_score, doc_ids)))
    document_return = []
    for id in reversed(doc_ids):
        doc = documents.get_document(id)
        document_return.append(doc)
        # print(doc.title)
    end_t = time.time()
    print("Search: %.4fs" % (end_t - start_t))
    print("Number of document: " + str(len(document_return)))
    return document_return, suggested_search, (end_t - start_t)

    # print("------Results-------")
    # for n, spis in enumerate(similar_positional_index):
    #     print("------" + str(n + 1) + "------")
    #     for spi in spis:
    #         for m, sdi in enumerate(spi.similar_document_ids):
    #             if spi.similar_document_in_order[m] is True:
    #                 print(documents.get_document(sdi).title)
    #                 # print(sdi)
    #         print("...")


def find_similar_documents(found_terms, fixed_search, one_word_search):
    number_of_terms = len(found_terms)
    term_similarity_list_num_words = []
    relevant_documents_id = []
    if one_word_search:
        start = 1
    elif fixed_search:
        start = number_of_terms
    else:
        start = num_words_search
    for n in range(start, number_of_terms + 1):
        term_similarity_list = []
        for m in range(len(found_terms) - n + 1):
            terms = []
            for o in range(n):
                # if n == 1 and found_terms[m + o] is not None:
                #     typ = nltk.pos_tag([found_terms[m + o].term])[0][1]
                #     if "VB" in typ:
                #         print("Verb Ignored: " + found_terms[m + o].term)
                #         continue # Enable here for verb filtering
                terms.append(found_terms[m + o])
            # if len(terms) == 0:
            #     continue
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


# def spelling_check(words):
#     try:
#         correct_words = []
#         for word in words:
#             word_wlf = pattern.sub(r"\1\1", word)
#             suggestions = suggest(word_wlf)
#             if suggestions[0][1] > 0.65:
#                 correct_word = suggestions[0][0]
#             else:
#                 correct_word = word
#             correct_words.append(correct_word)
#
#         return correct_words
#     except Exception:
#         return None


def spelling_check(words):
    try:
        correct_words = []
        for word in words:
            word_process = spell.unknown([word])
            if len(word_process) == 0:
                correct_words.append(word)
            else:
                corrected = spell.correction(next(iter(word_process)))
                correct_words.append(corrected)
        return correct_words
    except Exception:
        return None


print("Initializing Backend...")
initialize()
print("Done Initializing Backend...")
# calculate_fscore(documents)
# search_string("Laplace Trnsform")
# search_string("\  "Men Killed\"")
# search_string("\"Men LA What\"")
# search_string("Killlled")
