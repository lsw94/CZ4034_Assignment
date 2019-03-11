class Document:

    def __init__(self, source, title, description, url, image_url, publish_time, content, id):

        self.id = id

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
