from SearchEngine.Backend.Objects.Term import Term


class TermList:

    def __init__(self):
        self.term_list = []
        self.term_length_stop_position_dict = {}

    def __len__(self):
        return len(self.term_list)

    def add_term(self, word):
        self.term_list.append(Term(word, len(self.term_list)))

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

    def get_term(self, term):
        if str(len(term)) not in self.term_length_stop_position_dict:
            return None
        l, r = self.term_length_stop_position_dict[str(len(term))]
        for n in range(l, r):
            if self.term_list[n].term == term:
                return self.term_list[n]
        return None

    def sort_by_term_length(self):
        self.term_list.sort(key=lambda x: len(x))

    def generate_term_length_stop_positions(self):
        current_length = 1
        start_position = 0
        for n, term in enumerate(self.term_list):
            if len(term) > current_length:
                end_position = n
                self.term_length_stop_position_dict[str(current_length)] = (start_position, end_position)
                start_position = n
                current_length = len(term)
        self.term_length_stop_position_dict[str(current_length)] = (start_position, len(self.term_list) - 1)