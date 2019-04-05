from SearchEngine.Backend.Objects.PositionalIndexList import PositionalIndexList


class TermDocumentSimilarity:

    def __init__(self, terms):
        self.terms = terms
        if len(self.terms) == 1:
            if self.terms[0] is None:
                self.similar_document_ids = []
                self.similar_document_in_order = []
            else:
                self.similar_document_ids = self.terms[0].positional_indexes.get_document_id_list()
                self.similar_document_in_order = [True] * len(self.similar_document_ids)
        else:
            self.similar_document_ids = self.find_similar_positional_index()
            self.similar_document_in_order = []
            self.find_if_term_in_order()

    def find_similar_positional_index(self):
        positional_index_list = []
        for found_term in self.terms:
            if found_term is None:
                positional_index_list.append(None)
            else:
                positional_index_list.append(found_term.positional_indexes)

        for n in range(0, len(positional_index_list) - 1):
            positional_index_a = positional_index_list[n]
            positional_index_b = positional_index_list[n + 1]
            if positional_index_a is None or positional_index_b is None:
                return []
            temp_positional_index = PositionalIndexList()
            for pi_a in positional_index_a:
                if positional_index_b.is_doc_id_in_list(pi_a.document):
                    temp_positional_index.add_positional_index(pi_a)
            positional_index_list[n + 1] = temp_positional_index
        return positional_index_list[len(positional_index_list) - 1].get_document_id_list()

    def find_if_term_in_order(self):
        for similar_document_id in self.similar_document_ids:
            positional_indexes = []
            falsed = False
            for term in self.terms:
                positional_indexes.append(term.get_positional_index_of_doc(similar_document_id))
            for n in range(len(positional_indexes) - 1):
                current_content_position = positional_indexes[n].position_content
                current_description_position = positional_indexes[n].position_description
                current_source_position = positional_indexes[n].position_source
                current_title_position = positional_indexes[n].position_title

                current_content_position = [x + 1 for x in current_content_position]
                current_description_position = [x + 1 for x in current_description_position]
                current_source_position = [x + 1 for x in current_source_position]
                current_title_position = [x + 1 for x in current_title_position]

                content_b = any(x in current_content_position for x in positional_indexes[n + 1].position_content)
                description_b = any(
                    x in current_description_position for x in positional_indexes[n + 1].position_description)
                source_b = any(x in current_source_position for x in positional_indexes[n + 1].position_source)
                title_b = any(x in current_title_position for x in positional_indexes[n + 1].position_title)

                if not (content_b or description_b or source_b or title_b):
                    self.similar_document_in_order.append(False)
                    falsed = True
                    break
            if not falsed:
                self.similar_document_in_order.append(True)
