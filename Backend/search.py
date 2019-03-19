import Backend.crawler as crawler
import Backend.lemmatizer as lemmatizer
documents = None
terms = None


def initialize():
    global documents, terms
    documents, terms = crawler.query_news(False)


def search_string(string):
    terms = lemmatizer.process_string(string)
    


