#2020/7/16

import json
import pandas as pd

"""
a table is represented by set of cells
"""
class AnnotatedTable:

    """
    construct the table with the given dataset.
    """
    def __init__(self, cells):
        self.df = cells  # a set of cells
        #self.load_and_clean_dataframe(dataset)

    def extract_values(self):
        """ convert annotated table to a dataframe 
            (drop trace information and keep only values and store it as a dataframe)
        """
        pass

    def to_dict(self):
        """convert to a dictionary for easy import export
        something like this:
        [{"a": {"value": 3, "trace": {"operator": "sum", "argument": [1, 2]}} ,"b": 4,"c": 5},
         {"a": {"value": 6, "trace": {"operator": "sum", "argument": [4, 2]}} ,"b": 3,"c": 7}]
        """
        pass

    def from_dict(self):
        """load from a dictionary represented annotated table"""
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

