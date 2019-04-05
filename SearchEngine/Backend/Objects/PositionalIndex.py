class PositionalIndex:

    def __init__(self, document, source, title, description, content):
        self.document = document
        self.position_source = source
        self.position_title = title
        self.position_description = description
        self.position_content = content

    def __len__(self):
        return len(self.position_source) + len(self.position_title) + len(self.position_description) + len(
            self.position_content)

