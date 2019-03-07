

class DocumentList:

    def __init__(self):
        self.document_list = []

    def append(self, document):
        self.document_list.append(document)

    def __len__(self):
        return len(self.document_list)
