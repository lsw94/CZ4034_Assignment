class Document:

    def __init__(self, source, title, description, url, image_url, publish_time, content, id):

        self.id = id
        self.source_processed = ""
        self.title_processed = ""
        self.description_processed = ""
        self.content_processed = ""
        self.document_tfidf_score = []

        if source is None:
            self.source = ""
        else:
            self.source = source

        if title is None:
            self.title = ""
        else:
            self.title = title

        if description is None:
            self.description = ""
        else:
            self.description = description

        if url is None:
            self.url = ""
        else:
            self.url = url

        if image_url is None:
            self.image_url = ""
        else:
            self.image_url = image_url

        if publish_time is None:
            self.publish_time = ""
        else:
            self.publish_time = publish_time

        if content is None:
            self.content = ""
        else:
            self.content = content

    def set_processed_strings(self, source, title, description, content):
        self.source_processed = source
        self.title_processed = title
        self.description_processed = description
        self.content_processed = content

    def get_positions_of_term(self, term):
        position_source = [i for i, word in enumerate(self.source_processed) if word == term]
        position_title = [i for i, word in enumerate(self.title_processed) if word == term]
        position_description = [i for i, word in enumerate(self.description_processed) if word == term]
        position_content = [i for i, word in enumerate(self.content_processed) if word == term]

        return position_source, position_title, position_description, position_content

    def __len__(self):
        return len(self.source_processed) + len(self.title_processed) + len(self.description_processed) + len(
            self.content_processed)

    def set_document_tfidf_score(self, document_tfidf_score):
        self.document_tfidf_score = document_tfidf_score

    def get_document_tfidf_score(self):
        return self.document_tfidf_score

