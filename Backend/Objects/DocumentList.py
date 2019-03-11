from Backend.Objects.Document import Document


class DocumentList:

    def __init__(self):
        self.idx = 0
        self.document_list = []

    def add_document(self, source, title, description, url, image_url, publish_time, content):
        self.document_list.append(
            Document(source, title, description, url, image_url, publish_time, content, len(self.document_list)))

    def __len__(self):
        return len(self.document_list)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.document_list[item.start:item.stop]
        else:
            return self.document_list[item]
