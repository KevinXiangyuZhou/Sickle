# 2020/7/16

import json
import pandas as pd
from table_cell import *

# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

"""
a table is represented by set of cells
"""


class AnnotatedTable:
    """
    construct the table with the given dataset.
    """

    def __init__(self, source, from_source=False):
        """load from a dictionary represented annotated table"""
        self.df = []  # stored as a two-level array with columns to be the inner level
        if not from_source:
            self.load_from_dict(source)
        else:
            self.load_from_source(source)

    def load_from_source(self, source):
        self.df = source.copy()

    # source should be a two-level array [[{val, exp}], [{val, exp}]]
    def load_from_dict(self, source):
        for col_id in range(len(source)):
            self.df.append([])
            for cell_id in range(len(source[col_id])):
                cell = source[col_id][cell_id]
                self.df[col_id].append(TableCell(cell["value"], cell["exp"]))

    def add_column(self, new_column):
        self.df.append(new_column.copy())

    def add_row(self, new_row):
        if self.is_empty():
            # initialize then append when the current space is empty
            for i in range(len(new_row)):
                self.df.append([new_row[i]])
        else:
            if len(new_row) != self.get_col_num():
                print("[error] new row with inconsistent column number added")
            for i in range(len(self.df)):
                self.df[i].append(new_row[i])

    def get_cell(self, x, y):
        return self.df[x][y]

    def get_column(self, col_index):
        return self.df[col_index].copy()

    def get_row(self, row_index):
        # get a list of cells in the same row index
        rlt = []
        for i in range(len(self.df)):
            rlt.append(self.df[i][row_index])
        return rlt

    def get_col_num(self):
        if self.df is []:
            return 0
        return len(self.df)

    def get_row_num(self):
        if self.df is []:
            return 0
        return len(self.df[0])

    def is_empty(self):
        return self.df == []

    def extract_values(self):
        """ convert annotated table to a dataframe 
            (drop trace information and keep only values and store it as a dataframe)
        """
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.get_value())
        return pd.DataFrame.from_dict(data)

    def extract_traces(self):
        """ version that only keeps trace info"""
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.get_exp())
        return pd.DataFrame.from_dict(data)

    def to_dataframe(self):
        """ convert annotated table to a dataframe
            cells in the dataframe are represented as <{self.value}, {self.operator}, {self.argument}>
        """
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.to_stmt())
        return pd.DataFrame.from_dict(data)

    def to_dict(self):
        """convert to a dictionary for easy import export"""
        dicts = []
        for i in range(len(self.df)):
            for j in range(len(self.df[i])):
                cell = self.df[i][j]
                dicts.append(cell.to_dict())
        return dicts

    def to_plain_dict(self):
        """for print use"""
        dicts = []
        for j in range(len(self.df[0])):
            d = {}
            for i in range(len(self.df)):
                cell = self.df[i][j]
                temp = cell.to_dict()
                for k in temp:
                    d[k] = temp[k]["value"]
            dicts.append(d)
        return dicts


"""
from format of eg.
[{"cust_country": "UK", "grade": 2, "outstanding_amt": 3600},
{"cust_country": "USA", "grade": 2, "outstanding_amt": 5400}]
"""

""" ----- annotated table util functions ----- """
def select_columns(att, cols):
    cell_list = []
    for col in cols:
        temp = []
        for i in range(att.get_row_num()):
            cell = att.get_cell(col, i)
            temp.append({"value": cell.get_value(), "exp": cell.get_exp()})
        cell_list.append(temp)
    return AnnotatedTable(cell_list)


"""checker function for pruning annotated outputs
actual: the table generated by synthesizer
target: the annotated table generated based on user inputs
"""
def checker_function(actual, target, print_result=False):
    if actual is None or target is None:
        return None

    # find mappings from cells in target to actual for each cell
    # store for each cell with format: {(x, y): [(0,0), (1,2)]}
    # TODO: reduce time complexity
    mapping = find_mapping(target, actual)
    # print(mapping)
    if mapping is None:
        return None

    # use column and row to remove infeasible mappings
    target_df = target.extract_values()
    prune_by_row_column(mapping, target_df, print_result)

    # search for possible mappings
    # stop whenever we find on feasible mapping, and return the mapping
    if not check_mappings(mapping):
        return None

    # extract mappings
    # use dfs to search for the valid mapping
    # print(mapping)
    keys = [*mapping.keys()]
    return extract_mappings(mapping, keys)


def check_cell_trace(prog, inputs, target_t):
    for x in range(target_t.get_col_num()):
        for y in range(target_t.get_row_num()):
            # compress the expression, no need for structured trace in this test
            exp = target_t.get_cell(x, y).get_flat_args()
            # Cost: x * y * n_exp * n_loc
            # print("checking cell at: " + str((x, y)))
            # print("arguments: " + str(exp))
            # print("-----")
            for t in exp:
                locs = infer_all_possible_loc(prog, inputs, t[1], t[2])
                exist = False
                # print("t: " + t + "locs: " + str(locs))
                # print("check t: " + str(t) + "with possible locs: " + str(locs))
                for loc in locs:
                    if loc[0] == x and (loc[1] == y or loc[1] == "?"):
                        exist = True
                if not exist:
                    # print out log on failure
                    print("=====Cell Check Result=====")
                    print("checking cell at: " + str((x, y)))
                    print("arguments: " + str(exp))
                    print("fail whenc check t: " + str(t) + "possible locs: " + str(locs))
                    print("==========")
                    return False
    return True

# compare two cells at a time
# if two cells are in the same row in the
def check_cell_trace_2(prog, inputs, target_t):
    for x in range(target_t.get_col_num()):
        for y in range(target_t.get_row_num()):
            # compress the expression, no need for structured trace in this test
            exp = target_t.get_cell(x, y).get_flat_args()
            # Cost: x * y * n_exp * n_loc
            # print("checking cell at: " + str((x, y)))
            # print("arguments: " + str(exp))
            # print("-----")
            for t in exp:
                locs = infer_all_possible_loc(prog, inputs, t[1], t[2])
                exist = False
                # print("t: " + t + "locs: " + str(locs))
                # print("check t: " + str(t) + "with possible locs: " + str(locs))
                for loc in locs:
                    if loc[0] == x and (loc[1] == y or loc[1] == "?"):
                        exist = True
                if not exist:
                    # print out log on failure
                    print("=====Cell Check Result=====")
                    print("checking cell at: " + str((x, y)))
                    print("arguments: " + str(exp))
                    print("fail whenc check t: " + str(t) + "possible locs: " + str(locs))
                    print("==========")
                    return False
    return True

def infer_all_possible_loc(prog, inputs, x, y):
    # call check cell trace to
    # get a map of each coord in input table with a list of all possible location it could be

    # check for output table, the coordinate of each cell should be contained in the intersection
    # of the lists of inferred location for all cells in the trace
    return prog.infer_cell(inputs, (x, y))


"""search for valid mapping for each cell in target table"""
def find_mapping(target, actual):
    mapping = {}
    for cid in range(target.get_col_num()):
        for rid in range(target.get_row_num()):
            # print(str(cid) + ", " + str(rid))
            mapping[(cid, rid)] = search_values(actual, target.get_cell(cid, rid))
            # let it fail here
            if not check_mappings(mapping):
                # print(mapping)
                return None
    return mapping


def search_values(table, cell):
    rlt = []
    for cid in range(table.get_col_num()):
        for rid in range(table.get_row_num()):
            if cell.matches(table.get_cell(cid, rid)):
                rlt.append((cid, rid))
    return rlt


"""prune mapping by relative column and row positions"""
def prune_by_row_column(mapping, target_df, print_result=False):
    if print_result:
        print(target_df)
        print("step1 mapping")
        print(mapping)
    # pruning each column
    # if two cells are in the same row in the output,
    # then their source (mappings in actual table) must be in the same row in actual
    x = 0
    for col in target_df.columns:
        l = [mapping[(x, y)] for y in range(len(target_df))]
        smallest = find_smallest_array(l)
        # get list of x value in the smallest mapping
        x_list = [a for (a, b) in smallest]
        # y_list = [b for (a, b) in smallest]
        for y in range(len(target_df)):
            mapping[(x, y)] = [t for t in mapping[(x, y)] if t[0] in x_list]
        x += 1
    if print_result:
        print("prune by col")
        print(mapping)
    # pruning each row
    # same pruning law as column
    for y in target_df.index.tolist():
        l = [mapping[(x, y)] for x in range(len(target_df.columns))]
        smallest = find_smallest_array(l)
        y_list = [b for (a, b) in smallest]
        for x in range(len(target_df.columns)):
            mapping[(x, y)] = [t for t in mapping[(x, y)] if t[1] in y_list]
    if print_result:
        print("prune by row")
        print(mapping)


"""use dfs to search for the valid mapping"""
def extract_mappings(mappings, keys):
    rlt = {}
    closed = []
    search_mappings(rlt, 0, mappings, keys, closed)
    # print(mappings)
    if check_valid_mapping(rlt, keys):
        return rlt
    else:
        return None

# helper function for search mappings
def search_mappings(rlt, index, mappings, keys, closed):
    if index == len(keys):
        # we are done searching for mappings
        # print(rlt)
        return False
    else:
        # iterate over maps
        coord = keys[index]
        # fail if there is no choice for current key
        # print([m for m in mappings[coord] if m not in closed])
        if not [m for m in mappings[coord] if m not in closed]:
            return True
        # iterate over mappings[coord]
        for i in range(len(mappings[coord])):
            if coord not in rlt:
                rlt[coord] = []
            if mappings[coord][i] not in closed:
                closed.append(mappings[coord][i])
                rlt[coord].append(mappings[coord][i])
                # print(rlt)
                found = search_mappings(rlt, index + 1, mappings, keys, closed)
                if found:
                    return True
                if check_mappings(rlt):
                    # if we found a valid mapping
                    return True
                rlt[coord].pop()
                closed.pop()


def check_valid_mapping(mapping, keys):
    if [*mapping.keys()] != keys:
        return False
    return check_mappings(mapping)


# check if there is no empty list in mappings
def check_mappings(mapping):
    if len(mapping) == 0:
        return False
    for k in mapping:
        if len(mapping[k]) == 0:
            return False
    return True


def find_smallest_array(list):
    # choose the firstly found array if tie
    if len(list) == 0:
        return []
    rlt = list[0]
    for array in list:
        if len(array) < len(rlt):
            rlt = array
    return rlt
