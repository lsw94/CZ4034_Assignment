import json
import os

import jsonpickle
import requests

from Backend.Objects.DocumentList import DocumentList
from Backend.lemmatizer import process_documents

url_everything = "https://newsapi.org/v2/everything?q=*&apiKey=a99a17c2950f4e66bcc4ad161b02f292&pageSize=100"
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
        for n in range(maximum_number_of_news):
            response = requests.get(url_everything + "&page=" + str(n + 1))
            data = response.json()
            if data["status"] != "ok":
                break
            save_json(data, raw_path.replace(".json", str(n + 1) + ".json"))
            data_list.append(data)
    else:
        for n in range(maximum_number_of_news):
            if not os.path.exists(raw_path.replace(".json", str(n + 1) + ".json")):
                break
            data = load_json(raw_path.replace(".json", str(n + 1) + ".json"))
            data_list.append(data)

    if os.path.exists(documents_path) and not requery:
        documents = load_json_list(documents_path)
    else:
        documents = process_jsons(data_list)
        save_json(jsonpickle.encode(documents), documents_path)

    if os.path.exists(terms_path) and not requery:
        terms = load_json_list(terms_path)
    else:
        terms = process_documents(documents)
        for term in terms:
            positional_index = documents.get_positional_index(term.term)
            if len(positional_index) != term.frequency:
                print("Error")
                exit()
            term.add_positional_index(positional_index)
        save_json(jsonpickle.encode(terms), terms_path)
        save_json(jsonpickle.encode(documents), documents_processed_path)

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
    return documents

