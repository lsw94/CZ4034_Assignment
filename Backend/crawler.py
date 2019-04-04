import json
import os

import jsonpickle
import requests

from Backend.Objects.DocumentList import DocumentList
from Backend.lemmatizer import process_documents

api_keys = ["6209cba3f204447aa019713ad53decf5", "a99a17c2950f4e66bcc4ad161b02f292"]
# sources = ["the-economist", "bbc-news", "al-jazeera-english", "nbc-news", "cbs-news", "reuters", "vice-news",
#            "bloomberg", "msnbc", "daily-mail", "associated-press", "fox-news", "the-huffington-post",
#            "the-verge", "business-insider", "cbc-news", "ign", "buzzfeed", "newsweek", "new-scientist"]
sources = ["the-economist", "bbc-news"]
url_everything = "https://newsapi.org/v2/everything?pageSize=100&apiKey=a99a17c2950f4e66bcc4ad161b02f292&sources="
data_folder = "Data"
raw_path = os.path.join(data_folder, "raw.json")
documents_path = os.path.join(data_folder, "documents.json")
documents_processed_path = os.path.join(data_folder, "documents_processed.json")
terms_path = os.path.join(data_folder, "terms.json")
maximum_number_of_news = 1000


def query_news(requery):
    check_create_folder(data_folder)
    data_list = []
    if requery:
        for m, source in enumerate(sources):
            for n in range(maximum_number_of_news):
                response = requests.get(
                    url_everything + source + "&page=" + str(n + 1))
                data = response.json()
                if data["status"] != "ok":
                    break
                save_json(data, raw_path.replace(".json", "_" + str(m) + "_" + str(n + 1) + ".json"))
                data_list.append(data)
                if len(data["articles"]) != 100:
                    break
    else:
        for m, source in enumerate(sources):
            for n in range(maximum_number_of_news):
                if not os.path.exists(raw_path.replace(".json", "_" + str(m) + "_" + str(n + 1) + ".json")):
                    break
                data = load_json(raw_path.replace(".json", "_" + str(m) + "_" + str(n + 1) + ".json"))
                data_list.append(data)
    if os.path.exists(documents_path) and not requery:
        documents = load_json_list(documents_path)
    else:
        documents = process_jsons(data_list)
        save_json(jsonpickle.encode(documents), documents_path)

    if os.path.exists(terms_path) and os.path.exists(documents_processed_path) and not requery:
        terms = load_json_list(terms_path)
        documents = load_json_list(documents_processed_path)
    else:
        terms, documents = process_documents(documents)

        save_json(jsonpickle.encode(terms), terms_path)
        save_json(jsonpickle.encode(documents), documents_processed_path)


    print("-----Database size-----")
    print("Number of documents: " + str(len(documents)))
    print("Number of terms: " + str(len(terms)))
    return documents, terms


def save_json(data, file):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def load_json_list(file):
    data = load_json(file)
    return jsonpickle.decode(data)


def check_create_folder(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def process_jsons(datas):
    documents = DocumentList()
    for data in datas:
        articles = data["articles"]
        for article in articles:
            documents.add_document(article["source"]["name"], article["title"], article["description"], article["url"],
                                   article["urlToImage"], article["publishedAt"], article["content"])
    documents.sort_by_document_id()
    documents.generate_doc_id_length_stop_positions()
    return documents
