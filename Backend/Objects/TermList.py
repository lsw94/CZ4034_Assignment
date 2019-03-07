class TermList:

    def __init__(self):
        self.term_list = []

    def append(self, term):
        self.term_list.append(term)

    def __len__(self):
        return len(self.term_list)
