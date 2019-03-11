from Backend.Objects.Term import Term


class TermList:

    def __init__(self):
        self.term_list = []

    def __len__(self):
        return len(self.term_list)

    def add_term(self, word, documents, frequency):
        self.term_list.append(Term(word, documents, len(self.term_list), frequency))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.term_list[item.start:item.stop]
        else:
            return self.term_list[item]
