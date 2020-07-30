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
    def __init__(self, source):
        """load from a dictionary represented annotated table"""
        self.df = []
        for cell in source:
            self.df.append(TableCell(cell['value'], cell['argument'], cell['operator'], cell['attribute']))

    def extract_values(self):
        """ convert annotated table to a dataframe 
            (drop trace information and keep only values and store it as a dataframe)
        """
        data = []
        for cell in self.df:
            dict = cell.to_dict()
            attribute = cell.get_attribute()
            data.append({attribute: dict[attribute]['value']})
        return pd.DataFrame(data)

    def to_dict(self):
        """convert to a dictionary for easy import export
        something like this:
        [{"a": {"value": 3, "trace": {"operator": "sum", "argument": [1, 2]}} ,"b": 4,"c": 5},
         {"a": {"value": 6, "trace": {"operator": "sum", "argument": [4, 2]}} ,"b": 3,"c": 7}]
        """
        dicts = []
        for cell in self.df:
            dicts.append(cell.to_dict())
        return dicts


    def from_dict(self, source):
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
            if target_cell.contained_by(this_cell):
                exist = True
        if not exist:
            return False
    return True


class TableCell:
    """
    this class represents a cell stored in the data frame with its trace
    """
    def __init__(self, value, argument, operator, attribute):
        self.value = value
        self.argument = argument  # a list of [value, coordinate_x, coordinate_y]
        self.operator = operator
        self.attribute = attribute

    def get_value(self):
        return self.value

    def contained_by(self, other):
        # check this argument is a subset of target argument
        if self.argument is not None and \
                not self.is_sublist(self.argument, other.argument):
            return False
        if self.operator is not None and self.operator != other.operator:
            return False
        if self.value is None or self.value != other.value:
            return False
        return True

    def is_sublist(self, lst1, lst2):
        ls1 = [element for element in lst1 if element in lst2]
        ls2 = [element for element in lst2 if element in lst1]
        return ls1 == ls2

    def to_dict(self):
        return {self.attribute:
                    {'value': self.value,
                     'trace': {
                         'operator': self.operator,
                         'argument': self.argument.copy()
                        }
                     }
                }

    def get_attribute(self):
        return self.attribute
