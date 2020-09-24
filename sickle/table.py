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
        self.df = []  # stored as a two-level array with columns to be the inner level
        col_cells = {}
        for cell in source:
            if cell["attribute"] not in col_cells.keys():
                col_cells[cell["attribute"]] = []
            col_cells[cell["attribute"]].append(TableCell(cell["value"],
                                                          cell["argument"],
                                                          cell["operator"],
                                                          cell["attribute"]))
        for key in col_cells:
            self.df.append(col_cells[key])

    def get_cell(self, x, y):
        return self.df[x][y]

    def extract_values(self):
        """ convert annotated table to a dataframe 
            (drop trace information and keep only values and store it as a dataframe)
        """
        data = {}
        for i in range(len(self.df)):
            for j in range(len(self.df[i])):
                cell = self.df[i][j]
                attribute = cell.get_attribute()
                if attribute not in data.keys():
                    data[attribute] = []
                data[attribute].append(cell.get_value())
        return pd.DataFrame.from_dict(data)

    def to_dict(self):
        """convert to a dictionary for easy import export
        something like this:
        [{"a": {"value": 3, "trace": {"operator": "sum", "argument": [1, 2]}} ,"b": 4,"c": 5},
         {"a": {"value": 6, "trace": {"operator": "sum", "argument": [4, 2]}} ,"b": 3,"c": 7}]
        """
        dicts = []
        for i in range(len(self.df)):
            for j in range(len(self.df[i])):
                cell = self.df[i][j]
                dicts.append(cell.to_dict())
        return dicts

    def cells(self):
        return self.df.copy()

    def from_dict(self, source):
        pass

    def get_col_num(self):
        return len(self.df)

    def get_row_num(self):
        if self.df is []:
            return 0
        return len(self.df[0])


def check_function(actual, target):
    """ check if the set of values stored in target table is a subset of the set
    of values stored in this table, if the argument is contained by the trace of
    tuple stored in target table is a subset of argument in the corresponding trace,
    and check if operators match.
    """
    for target_cell in target.df:
        exist = False
        for actual_cell in actual.df:
            # check if there is at least one cell in actual table
            # that is a parent of the target tuple
            if target_cell.contained_by(actual_cell):
                exist = True
        if not exist:
            return False
    return True


def checker_function(actual, target):
    # if not check_function(actual, target):
    #   return False
    #     # map each cell with its coordinate in the table
    # actual_df = actual.extract_values()

    # find mappings from cells in target to actual for each cell
    # store for each cell with format: {(x, y): [(0,0), (1,2)]}
    # TODO: reduce time complexity
    mapping = {}
    for cid in range(target.get_col_num()):
        for rid in range(target.get_row_num()):
            #print(str(cid) + ", " + str(rid))
            mapping[(cid, rid)] = search_values(actual, target.get_cell(cid, rid))
            # let it fail here
            if not check_mappings(mapping):
                return None

    # use column and row to remove infeasible mappings
    target_df = target.extract_values()
    print("step1 mapping")
    print(mapping)
    # pruning each column
    x = 0
    for col in target_df.to_dict():
        list = [mapping[(x,y)] for y in target_df.to_dict()[col]]
        smallest = find_smallest_array(list)
        x_list = [a for (a, b) in smallest]
        # y_list = [b for (a, b) in smallest]
        for y in target_df.to_dict()[col]:
            mapping[(x,y)] = [t for t in mapping[(x,y)] if t[0] in x_list]
        x += 1
    print("prune by col")
    print(mapping)
    # pruning each row
    for y in target_df.index.tolist():
        list = [mapping[(x, y)] for x in range(len(target_df.columns))]
        smallest = find_smallest_array(list)
        y_list = [b for (a, b) in smallest]
        for x in range(len(target_df.columns)):
            mapping[(x, y)] = [t for t in mapping[(x, y)] if t[1] in y_list]
    print("prune by row")
    print(mapping)

    # search for possible mappings
    # stop whenever we find on feasible mapping, and return the mapping
    if not check_mappings(mapping):
        return None

    # extract mappings
    # use dfs to search for the valid mapping
    keys = [*mapping.keys()]
    open = [(keys[0], coord) for coord in mapping[keys[0]]]
    closed = []
    rlt = {}
    while open:
        curr = open.pop()
        rlt[curr[0]] = curr[1]
        #print(rlt)
        if check_valid_mapping(rlt, keys):
            print("result mappings:")
            return rlt
        # explore the state if have not
        if curr[1] not in closed:
            successor = get_successor(mapping, curr[0], keys)
            open += successor
            closed.append(curr[1])
        else:
            closed.remove(curr[1])
    return None


def get_successor(mapping, key, keys):
    index = keys.index(key)
    if index + 1 == len(keys):
        return []
    return [(keys[index + 1], coord) for coord in mapping[keys[index + 1]]]


def check_valid_mapping(mapping, keys):
    if [*mapping.keys()] != keys:
        return False
    found = []
    for key in mapping.keys():
        if mapping[key] in found:
            return False
        found.append(mapping[key])
    return True


# check if there is no empty list in mappings
def check_mappings(mapping):
    return all([len(mapping[k]) > 0 for k in mapping.keys()])


def search_values(table, cell):
    rlt = []
    for cid in range(table.get_col_num()):
        for rid in range(table.get_row_num()):
            if cell.contained_by(table.get_cell(cid, rid)):
                rlt.append((cid, rid))
    return rlt


def find_smallest_array(list):
    # choose the firstly found array if tie
    if len(list) == 0:
        return None
    rlt = list[0]
    for array in list:
        if len(array) < len(rlt):
            rlt = array
    return rlt


class TableCell:
    """
    this class represents a cell stored in the data frame with its trace
    """
    def __init__(self, value, argument, operator, attribute):
        self.value = value
        self.argument = argument  # a tuple of (value, coordinate_x, coordinate_y)
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
        ls = [element for element in lst1 if element not in lst2]
        # ls2 = [element for element in lst2 if element in lst1]
        return len(ls) == 0

    def to_dict(self):
        return {self.attribute:
                    {"value": self.value,
                     "trace": {
                         "operator": self.operator,
                         "argument": self.argument.copy()
                        }
                     }
                }

    def get_attribute(self):
        return self.attribute
