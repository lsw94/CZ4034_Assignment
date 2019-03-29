from Backend.Objects.Document import Document
from Backend.Objects.PositionalIndex import PositionalIndex
from Backend.Objects.PositionalIndexList import PositionalIndexList


class DocumentList:

    def __init__(self):
        self.document_list = []
        self.doc_id_length_stop_position_dict = {}

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

    def get_positional_index(self, term):
        positional_index_list = PositionalIndexList()
        for document in self.document_list:
            source_index, title_index, description_index, content_index = document.get_positions_of_term(term)
            if len(source_index) == 0 and len(title_index) == 0 and len(description_index) == 0 and len(
                    content_index) == 0:
                continue
            positional_index_list.add_positional_index(
                PositionalIndex(document.id, source_index, title_index, description_index, content_index))
        positional_index_list.sort_by_document_id()
        return positional_index_list

    def get_document(self, id):
        if str(len(str(id))) not in self.doc_id_length_stop_position_dict:
            return None
        l, r = self.doc_id_length_stop_position_dict[str(len(str(id)))]
        while l <= r:
            mid = l + (r - l) // 2
            if self.document_list[mid].id == id:
                return self.document_list[mid]
            elif self.document_list[mid].id < id:
                l = mid + 1
            else:
                r = mid - 1
        return None

    def get_total_number_of_terms(self):
        total = 0
        for document in self.document_list:
            total = total + len(document)
        return total

    def get_all_source_as_token(self):
        source_list = []
        for document in self.document_list:
            source_list.append(document.source_processed)
        return source_list

    def get_all_title_as_token(self):
        title_list = []
        for document in self.document_list:
            title_list.append(document.title_processed)
        return title_list

    def get_all_description_as_token(self):
        description_list = []
        for document in self.document_list:
            description_list.append(document.description_processed)
        return description_list

    def get_all_content_as_token(self):
        content_list = []
        for document in self.document_list:
            content_list.append(document.content_processed)
        return content_list

    def sort_by_document_id(self):
        self.document_list.sort(key=lambda x: x.id)

    def generate_doc_id_length_stop_positions(self):
        current_length = 1
        start_position = 0
        for n, doc in enumerate(self.document_list):
            if len(str(doc.id)) > current_length:
                end_position = n
                self.doc_id_length_stop_position_dict[str(current_length)] = (start_position, end_position)
                start_position = n
                current_length = len(str(doc.id))
        self.doc_id_length_stop_position_dict[current_length] = (start_position, len(self.document_list) - 1)