import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools
from table import *


class Node(ABC):
	def __init__(self):
		super(AbstractExpression, self).__init__()

	@abstractmethod
	def eval(self, inputs):
		"""the inputs are dataframes,
			it returns a pandas dataframe representation"""
		pass

	@abstractmethod
	def to_dict(self):
		pass

	@staticmethod
	def load_from_dict(ast):
		"""given a dictionary represented AST, load it in to a program form"""
		constructors = {
			"select": Select, "unite": Unite,
			"filter": Filter, "separate": Separate,
			"spread": Spread, "gather": Gather,
			"group_sum": GroupSummary,
			"cumsum": CumSum, "mutate": Mutate,
			"mutate_custom": MutateCustom,
		}
		if ast["op"] == "table_ref":
			return Table(ast["children"][0]["value"])
		else:
			node = constructors[ast["op"]](
						Node.load_from_dict(ast["children"][0]), 
						*[arg["value"] for arg in ast["children"][1:]])
			return node


class Table(Node):
	def __init__(self, data_id):
		self.data_id = data_id


	def eval(self, inputs):
		inp = inputs[self.data_id]
		if isinstance(inp, (list,)):
			df = pd.DataFrame.from_dict(inp)
		else:
			df = inp
		return df

	def to_dict(self):
		return {
			"type": "node",
			"op": "table_ref",
			"children": [
				value_to_dict(self.data_id, "table_id")
			]
		}


class Select(Node):
	def __init__(self, q, cols):
		self.q = q
		self.cols = cols

	def eval(self, inputs):
		df = self.q.eval(inputs)  # of pandas dataframe
		cell_list = []
		# for each column each cell in the selected column list
		# make it a dictionary of the cell looks like
		# {"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": None}
		cid = 0
		for colName, colData in df.iteritems():
			if colName not in self.cols:
				cid += 1
				continue
			rid = 0
			for data in colData:
				cell_list.append({"value": data,
								"argument": [(data, cid, rid)],
								"operator": "select",
								"attribute": colName})
				rid += 1
			cid += 1
		# return an annotated table
		return AnnotatedTable(cell_list)

	def to_dict(self):
		return {
			"type": "node",
			"op": "select",
			"children": [self.q.to_dict(), value_to_dict(self.cols, "col_index_list")]
		}


class Unite(Node):
	def __init__(self, q, col1, col2):
		""" col1, col2 are column indexes"""
		self.q = q
		self.col1 = col1
		self.col2 = col2

	def eval(self, inputs):
		# unite target columns to form a new df
		df = self.q.eval(inputs)
		ret = df.copy()
		new_col = get_fresh_col(list(ret.columns))[0]
		c1, c2 = ret.columns[self.col1], ret.columns[self.col2]
		ret[new_col] = ret[c1] + ret[c2]
		ret = ret.drop(columns=[c1, c2])
		# transform new df into annotated table
		print(ret)

		arguments = generate_direct_arguments(df)
		print(arguments)
		# map arguments for new column
		for index in arguments:
			arguments[index][new_col] = arguments[index][c1] + arguments[index][c2]

		return df_to_annotated_table_index_colname(ret, "unite", arguments)


	def to_dict(self):
		return {
			"type": "node",
			"op": "unite",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.col1, "col_index"), 
				value_to_dict(self.col2, "col_index")]}

class Filter(Node):
	def __init__(self, q, col_index, op, const):
		self.q = q
		self.col_index = col_index
		self.op = op
		self.const = const


	def eval(self, inputs):
		df = self.q.eval(inputs)
		col = df.columns[self.col_index]
		if self.op == "==":
			ret = df[df[col] == self.const]
		elif self.op == "!=":
			ret = df[df[col] != self.const]
		else:
			sys.exit(-1)
		arguments = generate_direct_arguments(df)
		return df_to_annotated_table_index_colname(ret, "filter", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "filter",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.col_index, "col_index"), 
				value_to_dict(self.op, "binop"), 
				value_to_dict(self.const, "constant")
			]}

class GroupSummary(Node):
	def __init__(self, q, group_cols, aggr_col, aggr_func):
		self.q = q
		self.group_cols = group_cols
		self.aggr_col = aggr_col
		self.aggr_func = aggr_func

	def eval(self, inputs):
		df = self.q.eval(inputs)
		res = df.copy()
		group_keys = [res.columns[idx] for idx in self.group_cols]
		# print(df.to_dict())
		target = res.columns[self.aggr_col]
		res = res.groupby(group_keys)

		# map argument for keys and groups
		arguments = {}  # {rid: {colname: argument}}
		gid = 0
		for (key, group) in res:
			# key can be a tuple if there are multiple group cols
			arguments[gid] = {}
			for col_name in group.to_dict().keys():
				arguments[gid][col_name] = []
				for row_index in group.to_dict()[col_name]:
					arguments[gid][col_name].append((group.to_dict()[col_name][row_index],
													 get_col_index_by_name(group, col_name),
													 row_index))
				"""
				if col in self.group_cols: # add the arguments for group_cols
					for rid in group.to_dict()[col]:  # {0: 3, 1: 3}
						key_arguments.append((group.to_dict()[col][rid], get_col_index_by_name(res, col), rid))
				else: # add arguments for other cols
					for rid in group.to_dict()[col]:  # {0: 4, 1: 5}
						group_arguments.append((group.to_dict()[col][rid], get_col_index_by_name(res, col), rid))
				"""
			gid += 1
		res = res.agg({target: self.aggr_func})
		#print(res.to_dict())
		if self.aggr_func == "mean":
			res[target] = res[target].round(2)
		res = res.rename(columns={target: f"{self.aggr_func}_{target}"})
		# print(res.to_dict())

		# change name of the target col name in arguments
		for row in arguments:
			arguments[row][self.aggr_func + "_" + target] = arguments[row][target]
		print(df)
		print(res)

		return df_to_annotated_table_rid_colname(res, "group + " + self.aggr_func, arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_sum",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_col, "col_index"), 
				value_to_dict(self.aggr_func, "aggr_func")
			]}

class GroupMutate(Node):
	def __init__(self, q, group_cols, target_col, aggr_func, sort_by, asc):
		self.q = q
		self.group_cols = group_cols
		self.target_col = target_col
		self.aggr_func = aggr_func
		self.sort_by = sort_by
		self.asc = asc

	def eval(self, inputs):
		df = self.q.eval(inputs)
		print(df)
		res = df.copy()
		target = df.columns[self.target_col]
		if self.sort_by is not None:
			res = res.sort_values(self.sort_by, ascending=self.asc)
		group_keys = [df.columns[idx] for idx in self.group_cols]

		temp = res.groupby(group_keys, sort=False)
		# {index: {colname: argument}}
		# iterate through df, map each cell in resulting table with its argument
		arguments = generate_direct_arguments(df)

		# map argument for keys and groups
		for (key, group) in temp:
			# key can be a tuple if there are multiple group cols
			# get the group argument for the target column
			temp_arg = []
			for row_index in group.to_dict()[target]:
				temp_arg.append((group.to_dict()[target][row_index],
												 get_col_index_by_name(group, target),
												 row_index))
			# map the group argument with the target col
			for row_index in group.to_dict()[target]:
				if row_index not in arguments:
					arguments[row_index] = {}
				arguments[row_index][target] = temp_arg

		res[target] = temp.transform(self.aggr_func)
		res = res.rename(columns={target: f"{self.aggr_func}_{target}"})

		# change name of the target col name in arguments
		for i in arguments:
			arguments[i][self.aggr_func + "_" + target] = arguments[i][target]

		print(res)
		return df_to_annotated_table_index_colname(res, "group + " + self.aggr_func, arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_sum",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.target_col, "col_index"),
				value_to_dict(self.func, "func")
			]}

class Mutate(Node):
	def __init__(self, q, target_col, func):
		self.q = q
		self.target_col = target_col
		self.func = func

	def eval(self, inputs):
		df = self.q.eval(inputs)
		target = df.columns[self.target_col]
		expression = eval("lambda x: x[\'" + target + "\']" + self.func)
		function = {target: expression}
		res = df.copy()
		arguments = generate_direct_arguments(res)
		res = res.assign(**function)
		res = res.rename(columns={target: f"{target}{self.func}"})
		print(res)

		for index in arguments:
			arguments[index][target + self.func] = arguments[index][target]
		return df_to_annotated_table_index_colname(res, "Mutate", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_mutate",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.target_col, "target_col"),
				value_to_dict(self.func, "func")]}

class CumSum(Node):
	def __init__(self, q, target):
		self.q = q
		self.target = target

	def eval(self, inputs):
		df = self.q.eval(inputs)
		ret = df.copy()
		# new_col = get_fresh_col(list(ret.columns))[0]
		target = ret.columns[self.target]
		ret["cumsum"] = ret[ret.columns[self.target]].cumsum()
		arguments = generate_direct_arguments(df)

		# map argument for col cumsum
		#print(ret.to_dict())
		temp_arg = []
		for index in ret.to_dict()["cumsum"]:
			temp_arg.append((df.to_dict()[target][index],
							 get_col_index_by_name(df, target),
							 index))

			if index not in arguments:
				arguments[index] = {}
			arguments[index]["cumsum"] = temp_arg
		#print(arguments)
		return df_to_annotated_table_index_colname(ret, "cumsum", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "cumsum",
			"children": [self.q.to_dict(), value_to_dict(self.target, "col_index")]
		}

#utility functions
def get_fresh_col(used_columns, n=1):
	"""get a fresh column name used in pandas evaluation"""
	names = []
	for i in range(0, 1000):
		if "COL_{}".format(i) not in used_columns:
			names.append("COL_{}".format(i))
		if len(names) >= n:
			break
	return names

def get_temp_var(used_vars):
	"""get a temp variable name """
	for i in range(0, 1000):
		var_name = "t{}".format(i)
		if var_name not in used_vars:
			return var_name

def value_to_dict(val, val_type):
	"""given the value and its type, dump it to a dict 
		the helper function to dump values into dict ast
	"""
	return {"type": val_type, "value": val}

def extract_table_schema(df):
	"""Given a dataframe, extract it's schema """
	def dtype_mapping(dtype):
		"""map pandas datatype to c """
		dtype = str(dtype)
		if dtype == "object" or dtype == "string":
			return "string"
		elif "int" in dtype or "float" in dtype:
			return "number"
		elif "bool" in dtype:
			return "bool"
		else:
			print(f"[unknown type] {dtype}")
			sys.exit(-1)

	schema = [dtype_mapping(s) for s in df.infer_objects().dtypes]
	return schema


""" return an annotated table for dataframe df with the given operator op
def df_to_annotated_table(df, op):
	cell_list = []
	cid = 0
	for colName in df.columns:
		rid = 0
		colData = df[colName]
		for data in colData:
			cell_list.append({"value": data,
							  "argument": [(data, cid, rid)],
							  "operator": op,
							  "attribute": colName})
			rid += 1
		cid += 1
	return AnnotatedTable(cell_list)
"""

def generate_direct_arguments(df):
	arguments = {}
	for index in df.index.tolist():
		arguments[index] = {}
		for col_name in df.to_dict().keys():
			arguments[index][col_name] = []
			arguments[index][col_name].append((df.to_dict()[col_name][index],
											   get_col_index_by_name(df, col_name),
											   index))
	return arguments


def df_to_annotated_table_rid_colname(df, op, arguments):
	cell_list = []
	cid = 0
	#print(df.to_dict())
	#print(df.columns.to_list())
	for colName in df.columns:
		rid = 0
		colData = df[colName]
		for data in colData:
			cell_list.append({"value": data,
							  "argument": arguments[rid][colName],
							  "operator": op,
							  "attribute": colName})
			rid += 1
		cid += 1
	return AnnotatedTable(cell_list)


def df_to_annotated_table_index_colname(df, op, arguments):
	cell_list = []
	#print(df.to_dict())
	#print(df.columns.to_list())
	for index in df.index.tolist():
		for colName in df.columns.tolist():
			cell_list.append({"value": df.to_dict()[colName][index],
							  "argument": arguments[index][colName],
							  "operator": op,
							  "attribute": colName})
	return AnnotatedTable(cell_list)

def get_col_index_by_name(df, colName):
	return df.columns.get_loc(colName)

# not used after revisions
def get_index_by_rownum(df, rid):
	return df.index.tolist()[rid]

def get_value_by_row_col(df, rid, cid):
	return df.iloc[rid][df.columns[cid]]


