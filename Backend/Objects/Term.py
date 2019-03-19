from Backend.Objects.PositionalIndexList import PositionalIndexList


class Term:

    def __init__(self, term, id, frequency):
        self.id = id
        self.term = term
        self.frequency = frequency
        self.positional_index = PositionalIndexList()

    def add_positional_index(self, positional_index):
        self.positional_index = positional_index
