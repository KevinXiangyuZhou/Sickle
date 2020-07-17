#2020/7/16

import json

'''
a table is represented by set of cells
'''
class Table:

    # construct the table with the given dataset.
    def __init__(self, dataset, tuples):
        self.df = tuples  # a set of tuples
        self.load_and_clean_dataframe(dataset)

    '''
    check if the set of values stored in target table is a subset of the set
    of values stored in this table, if the argument contained in the trace of
    tuple stored in target table is a subset of argument in the corresponding trace,
    and check if operators match.
    '''
    def check_function(self, target):
        for target_tuple in target.df:
            exist = False
            for this_tuple in self.df:
                # check if there is at least one tuple in this table
                # that is a parent of the target tuple
                if target_tuple.is_subset_of(this_tuple):
                    exist = True
            if not exist:
                return False
        return True


    def load_and_clean_dataframe(df):
        pass

    def get_df(self):
        pass

'''
this class represents a tuple stored in the data frame with its trace
'''
class Tuple:

    def __init__(self, values, argument, operator):
        self.values = values
        self.argument = argument
        self.operator = operator

    def get_value(self, index):
        return self.values[index]

    def is_subset_of(self, other):
        # check this argument is a subset of target argument
        if self.argument > other.argument:
            return False
        if self.operator != other.operator:
            return False
        if self.values != other.values:
            return False
        return True

    def modify_trace(self):
        pass

