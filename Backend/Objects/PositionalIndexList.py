class PositionalIndexList:

    def __init__(self):
        self.positional_index_list = []

    def add_positional_index(self, positional_index):
        self.positional_index_list.append(positional_index)

    def __len__(self):
        total = 0
        for positionalIndex in self.positional_index_list:
            total = total + len(positionalIndex)
        return total

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.positional_index_list[item.start:item.stop]
        else:
            return self.positional_index_list[item]
