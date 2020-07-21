#2020/7/16

import json

'''
a table is represented by set of cells
'''
class AnnotatedTable:

    # construct the table with the given dataset.
    def __init__(self, cells):
        self.df = cells  # a set of cells
        #self.load_and_clean_dataframe(dataset)

    def to_dataframe(self):
        """ convert annotated table to a dataframe
        """
        pass


def check_function(actual, target):
    """ check if the set of values stored in target table is a subset of the set
    of values stored in this table, if the argument contained in the trace of
    tuple stored in target table is a subset of argument in the corresponding trace,
    and check if operators match.
    """
    for target_cell in target.df:
        exist = False
        for this_cell in actual.df:
            # check if there is at least one tuple in this table
            # that is a parent of the target tuple
            if target_cell.is_subset_of(this_cell):
                exist = True
        if not exist:
            return False
    return True


def load_and_clean_dataframe(df):
    pass

def get_df(self):
    pass

class TableCell:
    """
    this class represents a cell stored in the data frame with its trace
    """
    def __init__(self, value, argument, operator):
        self.value = value
        self.argument = argument
        self.operator = operator

    def get_value(self):
        return self.value

    def is_subset_of(self, other):
        # check this argument is a subset of target argument
        if not self.argument.issubset(other.argument):
            return False
        if self.operator != other.operator:
            return False
        if self.value != other.value:
            return False
        return True

    def modify_trace(self):
        pass

