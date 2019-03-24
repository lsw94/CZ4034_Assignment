class PositionalIndexList:

    def __init__(self):
        self.positional_index_list = []

    def add_positional_index(self, positional_index):
        self.positional_index_list.append(positional_index)

    def term_frequency(self):
        total = 0
        for positionalIndex in self.positional_index_list:
            total = total + len(positionalIndex)
        return total

    def __len__(self):
        len(self.positional_index_list)

    def document_frequency(self):
        return len(self.positional_index_list)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.positional_index_list[item.start:item.stop]
        else:
            return self.positional_index_list[item]

    def get_positional_index_by_doc_id(self, id):
        for pi in self.positional_index_list:
            if id < pi.document:
                break
            if id == pi.document:
                return pi
        return None

    def sort_by_document_id(self):
        self.positional_index_list = self.sort(self.positional_index_list)

    def get_document_id_list(self):
        doc_id_list = []
        for pos_id in self.positional_index_list:
            doc_id_list.append(pos_id.document)
        return doc_id_list

    def is_in_list(self, doc):
        for pi in self.positional_index_list:
            if doc.document < pi.document:
                break
            if doc == pi:
                return True
        return False

    def sort(self, array):
        less = []
        equal = []
        greater = []

        if len(array) > 1:
            pivot = array[0]
            for x in array:
                if x.document < pivot.document:
                    less.append(x)
                elif x.document == pivot.document:
                    equal.append(x)
                else:
                    greater.append(x)
            return self.sort(less) + equal + self.sort(greater)
        else:
            return array


