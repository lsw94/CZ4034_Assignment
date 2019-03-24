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

    def get_term(self, search_term):
        for term in self.term_list:
            if len(term) > len(search_term):
                break
            if len(search_term) != len(term):
                continue
            if search_term == term.term:
                return term
        return None

    def sort_by_term_length(self):
        self.term_list = self.sort(self.term_list)

    def sort(self, array):
        less = []
        equal = []
        greater = []

        if len(array) > 1:
            pivot = array[0]
            for x in array:
                if len(x) < len(pivot):
                    less.append(x)
                elif len(x) == len(pivot):
                    equal.append(x)
                else:
                    greater.append(x)
            return self.sort(less) + equal + self.sort(greater)
        else:
            return array
