from Backend.Objects.PositionalIndexList import PositionalIndexList


class Term:

    def __init__(self, term, id, frequency):
        self.id = id
        self.term = term
        self.term_frequency = frequency
        self.document_frequency = 0
        self.positional_indexes = PositionalIndexList()
        self.term_idf = 0

    def add_positional_index(self, positional_index):
        self.positional_indexes = positional_index
        self.document_frequency = self.positional_indexes.document_frequency()

    def __len__(self):
        return len(self.term)

    def get_positional_index(self, doc_id):
        return self.positional_indexes.get_positional_index_by_doc_id(doc_id)

    def set_term_idf(self, term_idf):
        self.term_idf = term_idf

    def get_term_idf(self):
        return self.term_idf

