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
        l = 0
        r = len(self.positional_index_list) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if self.positional_index_list[mid].document == id:
                return self.positional_index_list[mid]
            elif self.positional_index_list[mid].document < id:
                l = mid + 1
            else:
                r = mid - 1
        return None

    def sort_by_document_id(self):
        self.positional_index_list.sort(key=lambda x: x.document)
        # self.positional_index_list = self.sort(self.positional_index_list)

    def get_document_id_list(self):
        doc_id_list = []
        for pos_id in self.positional_index_list:
            doc_id_list.append(pos_id.document)
        return doc_id_list

    def is_doc_id_in_list(self, id):
        l = 0
        r = len(self.positional_index_list) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if self.positional_index_list[mid].document == id:
                return True
            elif self.positional_index_list[mid].document < id:
                l = mid + 1
            else:
                r = mid - 1
        return False
