import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools


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
		cellList = []
		# for each column each cell in the selected column list
		# make it a dictionary of the cell looks like
		# {'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg', 'attribute': None}
		cid = 0
		for colName, colData in df.iteritems():
			if colName not in self.cols:
				cid += 1
				continue
			rid = 0
			for data in colData:
				cellList.append({'value': data,
								'argument': [[data, cid, rid]],
								'operator': 'select',
								'attribute': colName})
				rid += 1
		# return an annotated table
		return AnnotatedTable(cellList)

	def to_dict(self):
		return {
			"type": "node",
			"op": "select",
			"children": [self.q.to_dict(), value_to_dict(self.cols, "col_index_list")]
		}


class Unite(Node):
	def __init__(self, q, col1, col2, sep="_"):
		""" col1, col2 are column indexes"""
		self.q = q
		self.col1 = col1
		self.col2 = col2
		self.sep = sep

	def eval(self, inputs):
		df = self.q.eval(inputs)
		ret = df.copy()
		new_col = get_fresh_col(list(ret.columns))[0]
		c1, c2 = ret.columns[self.col1], ret.columns[self.col2]
		ret[new_col] = ret[c1] + self.sep + ret[c2]
		ret = ret.drop(columns=[c1, c2])
		return ret

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
			return df[df[col] == self.const].reset_index()
		elif self.op == "!=":
			return df[df[col] != self.const].reset_index()
		else:
			sys.exit(-1)

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
		group_keys = [df.columns[idx] for idx in self.group_cols]
		target = df.columns[self.aggr_col]
		res = df.groupby(group_keys).agg({target: self.aggr_func})
		if self.aggr_func == "mean":
			res[target] = res[target].round(2)
		res = res.rename(columns={target: f'{self.aggr_func}_{target}'}).reset_index()
		return res

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

class CumSum(Node):
	def __init__(self, q, target):
		self.q = q
		self.target = target

	def eval(self, inputs):
		df = self.q.eval(inputs)
		ret = df.copy()
		#new_col = get_fresh_col(list(ret.columns))[0]
		ret["cumsum"] = ret[ret.columns[self.target]].cumsum()
		return ret

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