import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools
from table import *


# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

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
			"select": Select,
			"unite": Unite,
			"filter": Filter,
			"group_sum": GroupSummary,
			"group_mutate": GroupMutate,
			"cumsum": CumSum,
			"mutate": Mutate,
		}
		if ast["op"] == "table_ref":
			return Table(ast["children"][0]["value"])
		else:
			node = constructors[ast["op"]](
						Node.load_from_dict(ast["children"][0]), 
						*[arg["value"] for arg in ast["children"][1:]])
			return node

	def to_stmt_dict(self):
		"""translate the expression into a  """
		def _recursive_translate(ast, used_vars):
			if ast["op"] == "table_ref":
				# create a variable to capture the return variable
				stmt_dict = copy.copy(ast)
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return [stmt_dict], used_vars + [var]
			else:
				stmt_dict = copy.copy(ast)
				# iterate over all possible subtrees
				sub_tree_stmts = []
				for i, arg in enumerate(ast["children"]):
					# check if the argument is an ast
					if isinstance(arg, (dict,)) and arg["type"] == "node":
						stmts, used_vars = _recursive_translate(ast["children"][0], used_vars)
						sub_tree_stmts += stmts
						# the subtree is replaced by a reference to the variable
						retvar = stmts[-1]["return_as"]
						stmt_dict["children"][i] = {"value": retvar, "type": "variable"}
				# use a temp variable to wrap the current statement, and add it to the coolection
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return sub_tree_stmts + [stmt_dict], used_vars + [var]
		stmts, _ = _recursive_translate(self.to_dict(), [])
		return stmts

	def is_abstract(self):

		"""Check if the subtree is abstract (contains any holes)"""

		def contains_hole(node):

			for i, arg in enumerate(node["children"]):

				if arg["type"] == "node":

					if contains_hole(arg):
						return True

				elif arg["value"] == HOLE:

					# we find a variable to infer

					return True

			return False

		return contains_hole(self.to_dict())

	def stmt_string(self):

		"""generate a string from stmts, for the purpose of pretty printing"""

		stmts = self.to_stmt_dict()

		result = []

		for s in stmts:
			lhs = s['return_as']

			f = s['op']

			arg_str = ', '.join([str(x['value']) for x in s["children"]])

			result.append(f"{lhs} <- {f}({arg_str})")

		return "; ".join(result)

class Table(Node):
	def __init__(self, data_id):
		self.data_id = data_id

	def infer_domain(self, arg_id, inputs, config):
		assert False, "Table has no args to infer domain."

	def infer_output_info(self, inputs):
		"""infer output schema """
		inp = inputs[self.data_id]
		if isinstance(inp, (list,)):
			df = pd.DataFrame.from_dict(inp)
		else:
			df = inp
		schema = extract_table_schema(df)
		return schema

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

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			input_schema = self.q.infer_output_info(inputs)
			col_num = len(input_schema)
			col_list_candidates = []
			for size in range(1, col_num + 1):
				col_list_candidates += list(itertools.combinations(list(range(col_num)), size))
			return col_list_candidates
		else:
			assert False, "[Select] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		schema = self.q.infer_output_info(inputs)
		return [s for i, s in enumerate(schema) if i in self.cols]

	def eval(self, inputs):
		df = self.q.eval(inputs)  # of pandas dataframe
		print(df)
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
		print(df)
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

	def infer_domain(self, arg_id, inputs, config):
		input_schema = self.q.infer_output_info(inputs)
		str_cols = [i for i, s in enumerate(input_schema) if s == "string"]
		if arg_id == 1:
			return str_cols
		if arg_id == 2:
			# refine the domain according to the first argumnet
			return str_cols if self.col1 == HOLE else [i for i in str_cols if i > self.col1]
		else:
			assert False, "[Unite] No args to infer domain for id > 2."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		return [s for i, s in enumerate(input_schema) if i not in [self.col1, self.col2]] + ["string"]

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

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			col_num = len(self.q.infer_output_info(inputs))
			return list(range(col_num))
		elif arg_id == 2:
			return config["filer_op"]
		elif arg_id == 3:
			return config["constants"]
		else:
			assert False, "[Filter] No args to infer domain for id > 3."

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs)

	def eval(self, inputs):
		df = self.q.eval(inputs)
		col = df.columns[self.col_index]
		if self.op == "==":
			ret = df[df[col] == self.const]
		elif self.op == "!=":
			ret = df[df[col] != self.const]
		elif self.op == ">":
			ret = df[df[col] > self.const]
		elif self.op == "<":
			ret = df[df[col] < self.const]
		else:
			sys.exit(-1)
		arguments = generate_direct_arguments(df)
		print(df)
		print(ret)
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

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			# approximation: only get fields with more than one values
			# for the purpose of avoiding empty fields
			try:
				df = self.q.eval(inputs)
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to elimiate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num + 1 - 1):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumbed by a table key, so all fields will be distinct
						continue
					gb_cols = df[[df.columns[k] for k in gb_keys]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			return col_list_candidates
		elif arg_id == 2:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			# the special column -1 is used for the purpose of "count", no other real intent
			cols += [-1]
			return cols
		elif arg_id == 3:
			if self.aggr_col != HOLE:
				if self.aggr_col == -1:
					return ["count"] if "count" in config["aggr_func"] else []
				else:
					return [f for f in config["aggr_func"] if f != "count"]
			else:
				return config["aggr_func"]
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		aggr_type = input_schema[self.aggr_col] if self.aggr_func != "count" else "number"
		return [s for i, s in enumerate(input_schema) if i in self.group_cols] + [aggr_type]


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
	def __init__(self, q, group_cols, target_col, aggr_func, sort_by=None, asc=True):
		self.q = q
		self.group_cols = group_cols
		self.target_col = target_col
		self.aggr_func = aggr_func
		self.sort_by = sort_by
		self.asc = asc

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			# approximation: only get fields with more than one values
			# for the purpose of avoiding empty fields
			try:
				df = self.q.eval(inputs)
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to elimiate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num + 1 - 1):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumbed by a table key, so all fields will be distinct
						continue
					gb_cols = df[[df.columns[k] for k in gb_keys]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			return col_list_candidates
		elif arg_id == 2:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			# the special column -1 is used for the purpose of "count", no other real intent
			cols += [-1]
			return cols
		elif arg_id == 3:
			if self.target_col != HOLE:
				if self.target_col == -1:
					return ["count"] if "count" in config["aggr_func"] else []
				else:
					return [f for f in config["aggr_func"] if f != "count"]
			else:
				return config["aggr_func"]
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		aggr_type = input_schema[self.target_col] if self.aggr_func != "count" else "number"
		return [s for i, s in enumerate(input_schema) if i in self.group_cols] + [aggr_type]

	def eval(self, inputs):
		df = self.q.eval(inputs)
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
		res[target] = temp.transform(self.aggr_func)[target]
		res = res.rename(columns={target: f"{self.aggr_func}_{target}"})

		# change the target col name in arguments
		for i in arguments:
			arguments[i][self.aggr_func + "_" + target] = arguments[i][target]

		print(df)
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
				value_to_dict(self.aggr_func, "func")
			]}

class Mutate(Node):
	def __init__(self, q, target_col, func, agg):
		self.q = q
		self.target_col = target_col
		self.func = func
		self.agg = agg

	def eval(self, inputs):
		df = self.q.eval(inputs)
		target = df.columns[self.target_col]
		res = df.copy()
		new_col = get_fresh_col(list(res.columns))[0]
		expression = eval("lambda x: x[\'" + target + "\']" + self.func)
		function = {new_col: expression}
		arguments = generate_direct_arguments(res)
		res = res.assign(**function)
		res = res.rename(columns={new_col: f"{target}{self.func}"})
		# change the target col name in arguments
		for index in arguments:
			arguments[index][target + self.func] = arguments[index][target]
		if self.agg:
			# trace for the generated new column should be a list of cells in target column
			target_col = []
			for i in arguments:
				target_col.append(arguments[i][target + self.func])
			for index in arguments:
				arguments[index][target + self.func] = target_col.copy()
		print(res)
		return df_to_annotated_table_index_colname(res, "Mutate", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_mutate",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.target_col, "target_col"),
				value_to_dict(self.func, "func"),
				value_to_dict(self.agg, "bool")]}

class CumSum(Node):
	def __init__(self, q, target):
		self.q = q
		self.target = target

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			input_schema = self.q.infer_output_info(inputs)
			return [i for i, s in enumerate(input_schema) if s == "number"]
		else:
			assert False, "[CumSum] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		return input_schema + ["number"]

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

			arguments[index]["cumsum"] = temp_arg.copy()
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

	# returns a list
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


