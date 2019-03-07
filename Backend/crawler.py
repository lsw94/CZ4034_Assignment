import json
import os

import requests

from Backend.Objects.Document import Document
from Backend.Objects.DocumentList import DocumentList

url_everything = "https://newsapi.org/v2/everything?q=*&apiKey=a99a17c2950f4e66bcc4ad161b02f292"
data_folder = "Data"
requery = False
raw_path = os.path.join(data_folder, "raw.json")
objects_path = os.path.join(data_folder, "raw_preprocessed.json")


def query_news():
    check_create_folder(data_folder)
    if requery or not os.path.exists(raw_path):
        response = requests.get(url_everything)
        data = response.json()
        save_json(data, raw_path)
    else:
        data = load_json(raw_path)
    documents = process_json(data)
    # save_json(documents.__dict__, objects_path)


def save_json(data, file):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def check_create_folder(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def process_json(data):
    articles = data["articles"]
    documents = DocumentList()
    for article in articles:
        document = Document(article["source"]["name"], article["title"], article["description"], article["url"],
                            article["urlToImage"], article["publishedAt"], article["content"])
        documents.append(document)
    return documents


query_news()
