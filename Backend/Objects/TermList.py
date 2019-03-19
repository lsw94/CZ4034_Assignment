from Backend.Objects.Term import Term


class TermList:

    def __init__(self):
        self.term_list = []

    def __len__(self):
        return len(self.term_list)

    def add_term(self, word, frequency):
        self.term_list.append(Term(word, len(self.term_list), frequency))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.term_list[item.start:item.stop]
        else:
            return self.term_list[item]

    def get_total_number_of_terms(self):
        total = 0
        for term in self.term_list:
            total = total + len(term.positional_index)
        return total
