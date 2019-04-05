from SearchEngine.Backend.crawler import query_news
from SearchEngine.Backend.lemmatizer import process_string
import os

storage_folder = "Storage"
storage_path = os.path.join(storage_folder, "storage.json")


def initialize_storage():
    check_create_folder(storage_folder)
    documents = query_news()


def check_create_folder(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)



