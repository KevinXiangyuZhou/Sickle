import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools
from table import *
from tabulate import tabulate
from table_cell import *
from table_cell_structureless import *


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
			"filter": Filter,
			"group_sum": GroupSummary,
			"group_mutate": GroupMutate,
			"mutate_arithmetic": Mutate_Arithmetic,
			"join": Join
		}
		if ast["op"] == "table_ref":
			return Table(ast["children"][0]["value"])
		else:
			if ast["op"] == "join":
				node = constructors[ast["op"]](
					Node.load_from_dict(ast["children"][0]),
					Node.load_from_dict(ast["children"][1]))
			else:
				node = constructors[ast["op"]](
					Node.load_from_dict(ast["children"][0]),
					*[arg["value"] for arg in ast["children"][1:]])
			return node

	def to_stmt_dict(self):
		"""translate the expression into a  """
		def _recursive_translate(ast, used_vars):
			#print(ast)
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
						stmts, used_vars = _recursive_translate(ast["children"][i], used_vars)
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

	""" return an abstract annotated table"""
	def infer_computation(self, inputs):
		pass

	def infer_colnum(self, inputs):
		pass

	def infer_rownum(self, inputs):
		pass

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
		def val_to_str(x):
			if x["value"] == HOLE:
				return "?"
			if x["type"] == "bv_filter":
				return "".join(["|" if k else "." for k in x["value"][:min(5, len(x["value"]))]])
			# if x["type"] == "predicates":
			# 	return pred_to_str(x["value"], hide_type=True)
			else:
				return str(x["value"])

		stmts = self.to_stmt_dict()
		result = []
		for s in stmts:
			lhs = s['return_as']
			f = s['op']
			arg_str = ', '.join([val_to_str(x) for x in s["children"]])
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
		t = df_to_annotated_table_index_colname(df, None,
												generate_direct_arguments(df, data_id=self.data_id), None)
		return t

	def to_dict(self):
		return {
			"type": "node",
			"op": "table_ref",
			"children": [
				value_to_dict(self.data_id, "table_id")
			]
		}

	def get_id(self):
		return self.data_id

	def infer_computation(self, inputs):
		return self.eval(inputs)

	def infer_cell_2(self, inputs):
		return self.eval(inputs)

	def infer_cell(self, inputs, target):
		return [target]

	def infer_colnum(self, inputs):
		return len(inputs[self.data_id][0])

	def infer_rownum(self, inputs):
		return len(inputs[self.data_id])


class Join(Node):
	def __init__(self, q1, q2):
		self.q1 = q1
		self.q2 = q2

	def infer_domain(self, arg_id, inputs, config):
		return []

	def infer_output_info(self, inputs):
		# schema_1, table_boundaries_1 = self.q1.infer_output_info(inputs)
		# schema_2, table_boundaries_2 = self.q2.infer_output_info(inputs)
		schema_1 = self.q1.infer_output_info(inputs)
		schema_2 = self.q2.infer_output_info(inputs)
		return schema_1 + schema_2

	def eval(self, inputs):
		# make a copy of table for argument reference
		# for convenience of infer_computation
		table1 = self.q1.eval(inputs)
		table2 = self.q2.eval(inputs)

		# evaluate the dataframes from inputs
		df1 = table1.extract_values()
		df2 = table2.extract_values()

		# perform join
		res = (df1.assign(temp_join_key=1)
			   .merge(df2.assign(temp_join_key=1), on="temp_join_key")
			   .drop("temp_join_key", axis=1))

		arguments = {}
		# add trace info for df1
		for colname in df1.columns:
			x_coord = get_col_index_by_name(df1, colname)
			for i in range(len(df1)):
				for j in range(len(df2)):
					# position info in output table
					# the rid is the same as in the output table
					rid = i * len(df2) + j
					colname = res.columns[x_coord]
					if rid not in arguments:
						arguments[rid] = {}
					arguments[rid][colname] = [("table_1", x_coord, i)]

		# add trace info for df2
		for colname in df2.columns:
			x_coord = get_col_index_by_name(df2, colname)
			for i in range(len(df1)):
				for j in range(len(df2)):
					# position info in output table
					rid = i * len(df2) + j
					# the cid in the output table is shifted
					# right by the number of columns in df1
					colname = res.columns[x_coord + len(df1.columns)]
					if rid not in arguments:
						arguments[rid] = {}
					arguments[rid][colname] = [("table_2", x_coord, j)]

		#print(res)
		return df_to_annotated_table_join(res, None, arguments, table1, table2)

	def to_dict(self):
		return {
			"type": "node",
			"op": "join",
			"children": [
				self.q1.to_dict(),
				self.q2.to_dict(),
			]
		}

	# To make sure there is no duplicate column name
	# 1. could replace column names in two join tables first then join
	# 2. could add new name when needed
	def infer_computation(self, inputs):
		# a cross product of computed intermediate of the two joined programs
		table1 = self.q1.infer_computation(inputs)
		table2 = self.q2.infer_computation(inputs)

		# two empty table we will build and merge together
		empty_table1 = AnnotatedTable([])
		empty_table2 = AnnotatedTable([])
		# build the table by first replicating table2 by row(table1) times
		# (add the result as new rows; also added one by one due to implementation decision)
		for i in range(table1.get_row_num()):
			# repeat all rows
			for j in range(table2.get_row_num()):
				empty_table2.add_row(table2.get_row(j))

		# then replicate each row of table1 by row(table2) times
		for i in range(table1.get_row_num()):
			# each row is repeated by row(table2) times
			for j in range(table2.get_row_num()):
				empty_table1.add_row(table1.get_row(i))
		for i in range(empty_table2.get_col_num()):
			empty_table1.add_column(empty_table2.get_column(i))
		return empty_table1

	def infer_cell(self, inputs, target):
		# if loc[0] in [0, n1 - 1] (in table 1)
		# next loc is (loc[0], row(t2) * loc[1] + i) for i in range(row(t2))
		# if loc[0] in [n1, n2 - 1] (in table 2)
		# next loc is (loc[0], row(t2) * i + loc[1]) for i in range(row(t1))
		def infer_single_cell1(loc):
			if loc[1] == "?":
				return [(loc[0], loc[1])]
			else:
				return [(loc[0], r2 * loc[1] + i) for i in range(r2)]

		def infer_single_cell2(loc):
			if loc[1] == "?":
				return [(loc[0] + n1, loc[1])]
			else:
				return [(loc[0] + n1, r2 * i + loc[1]) for i in range(r1)]
		n1 = self.q1.infer_colnum(inputs)
		# n2 = self.q2.infer_colnum(inputs)
		r1 = self.q1.infer_rownum(inputs)
		r2 = self.q2.infer_rownum(inputs)
		curr = []
		for c in self.q1.infer_cell(inputs, target):
			curr += infer_single_cell1(c)

		for c in self.q2.infer_cell(inputs, target):
			curr += infer_single_cell2(c)
		return curr

	def infer_cell_2(self, inputs):
		pass


	def infer_colnum(self, inputs):
		return self.q1.infer_colnum(inputs) + self.q2.infer_column(inputs)

	def infer_rownum(self, inputs):
		return self.q1.infer_colnum(inputs) * self.q2.infer_column(inputs)


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
				col_list_candidates += list(itertools.permutations(list(range(col_num)), size))  # want permutation
			return col_list_candidates
		else:
			assert False, "[Select] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		schema = self.q.infer_output_info(inputs)
		return [s for i, s in enumerate(schema) if i in self.cols]

	def eval(self, inputs):
		df = self.q.eval(inputs).extract_values()  # of type pandas dataframe
		res = self.q.eval(inputs)
		# check if df has input cols
		for col in self.cols:
			if col >= len(df.columns):
				print("error in select")
				assert False

		df_prev = df.copy()
		select_cols = [df.columns[i] for i in self.cols]
		df = df[select_cols]
		arguments = {}
		for rid in range(len(df)):
			for colname in df.columns:
				if rid not in arguments:
					arguments[rid] = {}
				arguments[rid][colname] = [(get_col_index_by_name(df_prev, colname), rid)]
		return df_to_annotated_table_index_colname(df, None, arguments, res)

	def to_dict(self):
		return {
			"type": "node",
			"op": "select",
			"children": [self.q.to_dict(), value_to_dict(self.cols, "col_index_list")]
		}

	def infer_computation(self, inputs):
		# if concrete return evaluation of this level
		if self.cols != HOLE:
			return self.eval(inputs)
		else:
			return self.q.infer_computation(inputs)

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
		if df is None:
			return None
		if not isinstance(df, pd.DataFrame):
			df = df.extract_values()
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
		# print(df)
		# print(ret)
		return df_to_annotated_table_index_colname(ret, "filter", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "filter",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.col_index, "col_index"), 
				value_to_dict(self.op, "filter_op"),
				value_to_dict(self.const, "constants")
			]}

class GroupSummary(Node):
	def __init__(self, q, group_cols, aggr_func, aggr_col):
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
				df = df.extract_values()
				if df.empty:
					return []
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to eliminate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num + 1 - 1):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumed by a table key, so all fields will be distinct
						continue
					# print(df)
					gb_cols = df[[df.columns[k] for k in gb_keys if k < len(df.columns)]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			if not col_list_candidates:
				print("pruned by group keys")
			return col_list_candidates
		elif arg_id == 3:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			return cols
		elif arg_id == 2:
			if self.aggr_col != HOLE:
				return [f for f in config["aggr_func"]]
			else:
				return config["aggr_func"]
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		# aggr_type = input_schema[self.aggr_col] if self.aggr_func != "count" else "number"
		aggr_type = input_schema[self.aggr_col]
		# return [s for i, s in enumerate(input_schema) if i in self.group_cols] + [aggr_type]
		output_schema = [s for i, s in enumerate(input_schema) if i in self.group_cols]
		if aggr_type == "number":
			output_schema += ["number"]
		return output_schema

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = self.q.eval(inputs)
		df = self.q.eval(inputs).extract_values()
		res = df.copy()
		# print(res)
		group_keys = [res.columns[idx] for idx in self.group_cols]
		# print(df.to_dict())
		target = res.columns[self.aggr_col]
		res = res.groupby(group_keys)

		# map argument for keys and groups
		arguments = {}  # {rid: {colname: argument}}
		gid = 0
		for (key, group) in res:
			arguments[gid] = {}
			for colname in group.columns:
				val_arg = []
				key_arg = []
				# group.to_dict() in {col_name:{rid:}} format
				for row_index in group.to_dict()[colname]:
					# (value, cid, rid)
					if colname == target:
						val_arg.append((get_col_index_by_name(group, colname),
										 row_index))
					elif colname in [group.columns[i] for i in self.group_cols]:
						# check if it is an expnode obj
						temp_exp = table.get_cell(get_col_index_by_name(group, colname),
												  row_index).get_exp()
						if isinstance(temp_exp, ExpNode):
							temp_exp = [temp_exp]
						key_arg += temp_exp
				# map the group argument with the target col
				arguments[gid][colname] = val_arg
				if key_arg != []:
					arguments[gid][colname] += [ArgOr(key_arg)]
			gid += 1
		res = res.agg({target: self.aggr_func})
		res = res.reset_index()
		res = round_df(res)
		# change name of the target col name in arguments
		res = res.rename(columns={target: f"{self.aggr_func}_{target}"})
		for row in arguments:
			arguments[row][self.aggr_func + "_" + target] = arguments[row][target]
		# print(df)
		# print(res)
		return df_to_annotated_table_index_colname(res, self.aggr_func, arguments,
													table,  target_cols=[self.aggr_func + "_" + target])

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_sum",
			"children": [
				self.q.to_dict(), 
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_func, "aggr_func"),
				value_to_dict(self.aggr_col, "col_index")
			]}

	def infer_colnum(self, inputs):
		n = self.q.infer_colnum(inputs)
		if self.group_cols == HOLE:
			return n
		else:
			return len(self.group_cols) + 1

	# there is some inaccuracy in inferring the rownum of groupsum program
	def infer_rownum(self, inputs):
		return self.q.infer_rownum(inputs)

	def infer_computation(self, inputs):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.aggr_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		if self.group_cols == HOLE:
			table = self.q.infer_computation(inputs)
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, HOLE))
			table.add_column(new_col)
			return table
		table = select_columns(self.q.infer_computation(inputs), self.group_cols)
		if self.aggr_func == HOLE:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, HOLE))
			table.add_column(new_col)
		else:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, ExpNode(self.aggr_func, [HOLE])))
			table.add_column(new_col)
		return table

	def infer_cell(self, inputs, target):
		def infer_single_cell(loc, n):
			pre = [(i, "?") for i in range(n)]
			if self.group_cols == HOLE and self.aggr_col == HOLE:
				return pre
			if self.group_cols != HOLE:
				if loc[0] in self.group_cols:
					pre = [(self.group_cols.index(loc[0]), "?")]
				else:
					pre = [(len(self.group_cols), "?")]
			if self.aggr_col != HOLE and loc[0] == self.aggr_col:
				pre = [(len(self.group_cols), "?")]
			elif self.aggr_col != HOLE and self.group_cols != HOLE and \
					loc[0] not in self.group_cols and loc[0] != self.aggr_col:
				pre = []
			return pre
		n = self.q.infer_colnum(inputs)
		pre_list = self.q.infer_cell(inputs, target)
		curr = []
		for c in pre_list:
			curr += infer_single_cell(c, n)
		return curr

	def infer_cell_2(self, inputs):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.aggr_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)

		table = self.q.infer_cell_2(inputs)
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		if self.group_cols == HOLE:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:  # the new cell in new column can come from any cell
						new_source[cid].append(TableCell(HOLE, HOLE))
					else:  # other cells should remain in its previous pos
						new_source[cid].append(table.get_cell(cid, rid))
		elif self.aggr_func == HOLE:
			for cid in range(colnum):  # group_cols + one new col
				if cid not in self.group_cols and cid != colnum:
					continue
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not be placed in group cols
						trace = [(x, y) for x in range(colnum) for y in range(rownum) if x not in self.group_cols]
					else:  # cid in self.group_cols:
						# this column is group column
						# its trace should be ArgOr of all cells in the column
						trace = [(cid, y) for y in range(rownum)]
					args = []
					for c in trace:
						if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
							args += table.get_cell(c[0], c[1]).get_exp()
						else:
							args += [table.get_cell(c[0], c[1]).get_exp()]
					new_cell = TableCell(HOLE, args)
					new_source[-1].append(new_cell)
		else:
			for cid in range(colnum):  # group_cols + one new col
				if cid not in self.group_cols and cid != colnum:
					continue
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not be placed in group cols
						trace = [(x, y) for x in range(colnum) for y in range(rownum) if x not in self.group_cols]
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, ExpNode(self.aggr_func, args))
					else:  # cid in self.group_cols:
						# this column is group column
						# its trace should be ArgOr of all cells in the column
						trace = [(cid, y) for y in range(rownum)]
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, args)
					new_source[-1].append(new_cell)
		return AnnotatedTable(new_source, from_source=True)


class GroupMutate(Node):
	def __init__(self, q, group_cols, aggr_func, target_col, sort_by=None, asc=True):
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
				df = df.extract_values()
				if df.empty:
					return []
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to elimiate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumbed by a table key, so all fields will be distinct
						continue
					gb_cols = df[[df.columns[k] for k in gb_keys if k < len(df.columns)]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			col_list_candidates.append([])
			return col_list_candidates
		elif arg_id == 3:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			# the special column -1 is used for the purpose of "count", no other real intent
			# cols += [-1]
			return cols
		elif arg_id == 2:
			if self.target_col != HOLE:
				return [f for f in config["mutate_func"] if f != "count"]
			else:
				return config["mutate_func"]
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		# aggr_type = input_schema[self.aggr_col] if self.aggr_func != "count" else "number"
		aggr_type = input_schema[self.target_col]
		output_schema = [s for i, s in enumerate(input_schema)]
		if aggr_type == "number":
			output_schema += ["number"]
		return output_schema

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = copy.copy(self.q.eval(inputs))
		df = self.q.eval(inputs).extract_values()

		res = df.copy()
		target = df.columns[self.target_col]
		if self.sort_by is not None:
			res = res.sort_values(self.sort_by, ascending=self.asc)
		new_col = get_fresh_col(list(res.columns))[0]
		arguments = generate_direct_arguments(res)
		if self.group_cols:
			group_keys = [df.columns[idx] for idx in self.group_cols]
			# Format: {index: {colname: argument}}
			# iterate through df, map each cell in resulting table with its argument
			temp = res.groupby(group_keys, sort=False)

			# map argument for keys and groups
			for (key, group) in temp:
				# key can be a tuple if there are multiple group cols
				# get the group argument for the target column
				temp_arg = []
				# group.to_dict() in {col_name:{rid:}} format
				if self.aggr_func == "cumsum":
					# map argument for col cumsum
					for index in group.to_dict()[target]:
						temp_arg.append((self.target_col, index))
						arguments[index][new_col] = temp_arg.copy()
					continue
				# get a list of all coordinate in the same group in target column
				for row_index in group.to_dict()[target]:
					temp_arg.append((self.target_col, row_index))
				# map the group argument with the target col
				for row_index in group.to_dict()[target]:
					if row_index not in arguments:
						arguments[row_index] = {}
					arguments[row_index][new_col] = temp_arg
			# do aggregation work
			res[new_col] = temp.transform(self.aggr_func)[target]
		else:
			# if we have not group_key, we simply do a mutate
			if self.aggr_func == "cumsum":
				res[new_col] = res[target].cumsum()
				# map argument for col cumsum
				temp_arg = []
				for index in res.to_dict()[target]:
					temp_arg.append((self.target_col, index))
					arguments[index][new_col] = temp_arg.copy()
			else:
				res[new_col] = res.apply(self.aggr_func)[target]
				# add arguments for the new column
				temp_arg = []
				for index in arguments:
					temp_arg += (arguments[index][target].copy())
				for index in arguments:
					arguments[index][new_col] = temp_arg.copy()

		res = round_df(res)
		# print(df)
		# print(res)
		# print(arguments)
		return df_to_annotated_table_index_colname(res, self.aggr_func, arguments,
												   table, target_cols=[new_col])

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_mutate",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_func, "func"),
				value_to_dict(self.target_col, "col_index")
			]}

	def infer_colnum(self, inputs):
		return self.q.infer_colnum(inputs) + 1

	def infer_rownum(self, inputs):
		return self.q.infer_rownum(inputs)

	def infer_computation(self, inputs):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.target_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		table = self.q.infer_computation(inputs)
		df = table.extract_values()
		#target = get_fresh_col(df.columns)[0]
		if self.aggr_func == HOLE or self.group_cols == HOLE:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, HOLE))
			table.add_column(new_col)
		else:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, ExpNode(self.aggr_func, [HOLE])))
			table.add_column(new_col)
		return table

	def infer_cell(self, inputs, target):
		def infer_single_cell(loc, n):
			pre = [(loc[0], loc[1]), (n, "?")]
			if self.group_cols != HOLE:
				if self.group_cols is []:
					pre = [(loc[0], loc[1]), (n, loc[1])]
				elif loc[0] in self.group_cols:
					pre = [(loc[0], loc[1])]
			if self.target_col != HOLE and loc[0] != self.target_col:
				# should only be placed in its original position anyway
				pre = [(loc[0], loc[1])]
			return pre
		n = self.q.infer_colnum(inputs)
		# get a list of possible positions the target cell could be placed
		pre_list = self.q.infer_cell(inputs, target)
		curr = []
		for c in pre_list:
			curr += infer_single_cell(c, n)
		return curr

	# a with-trace version of infer computation
	# we infer the trace for each cell in the intermediate table
	# the column should be stored in some key-value format where keys are unique within tables (no "?")
	# 1. all holes. then
	# 2. group_col = hole.
	# for each cell, infer a list of cell it could come from
	# so traverse input cells infer positions for each cell, and accumulate the result into a list
	# (not quite reasonable)
	# whether the pruning described above effective (create intermediate table with cell trace)

	def infer_cell_2(self, inputs):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.target_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)

		table = self.q.infer_cell_2(inputs)
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		if self.group_cols == HOLE:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:  # the new cell in new column can come from any cell
						new_source[cid].append(TableCell(HOLE, HOLE))
					else:  # other cells should remain in its previous pos
						new_source[cid].append(table.get_cell(cid, rid))
		elif self.aggr_func == HOLE:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not come from cells in group cols
						trace = [(x, y) for x in range(colnum) for y in range(rownum) if x not in self.group_cols]
					elif cid in self.group_cols:
						# this column is group column
						# its trace should be ArgOr of all cells in the column
						# trace = [(cid, y) for y in range(rownum)]
						trace = [(cid, rid)]
					else:
						# other cells can only come from its previous pos
						trace = [(cid, rid)]
					args = []
					for c in trace:
						if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
							args += table.get_cell(c[0], c[1]).get_exp()
						else:
							args += [table.get_cell(c[0], c[1]).get_exp()]
					new_cell = TableCell(HOLE, args)
					new_source[cid].append(new_cell)
		else:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not be placed in group cols
						trace = [(x, y) for x in range(colnum) for y in range(rownum) if x not in self.group_cols]
					elif cid in self.group_cols:
						# this column is group column
						# its trace should be ArgOr of all cells in the column
						# trace = [(cid, y) for y in range(rownum)]
						trace = [(cid, rid)]
					else:
						# other cells can only come from its previous pos
						trace = [(cid, rid)]

					if cid == colnum:
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, ExpNode(self.aggr_func, args))
					else:
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, args)
					new_source[cid].append(new_cell)
		return AnnotatedTable(new_source, from_source=True)


class Mutate_Arithmetic(Node):
	def __init__(self, q, func, cols):
		self.q = q
		self.cols = cols
		self.func = func

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			return config["mutate_function"]
		elif arg_id == 2:
			if self.func != HOLE:
				expression = eval(self.func)
				argc = expression.__code__.co_argcount
				columns = [i for i, s in enumerate(schema) if s == "number"]
				combinations_objects = list(itertools.permutations(columns, argc))
				return combinations_objects
			else:
				return []
		else:
			assert False, "[Mutate] No args to infer domain for id > 3."

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs) + ["number"]

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = copy.copy(self.q.eval(inputs))
		df = self.q.eval(inputs).extract_values()

		res = df.copy()
		arguments = generate_direct_arguments(res)
		col_name = str(self.cols)

		expression = eval(self.func)  # lambda x, y: x - y - 0.1 * x
		argc = expression.__code__.co_argcount
		if argc == 0 or len(self.cols) != argc:
			print("invalid number of arguments")
			assert False
		x_func = "lambda f: lambda x: f(x[0]"
		for i in range(1, argc):
			x_func += ", x[" + str(i) + "]"
		x_func += ")"
		# get the result of the function and make a new column with the value
		new_col = eval(x_func)(expression)([res[res.columns[i]] for i in self.cols])
		# new_col = [new_value for i in range(0, len(res))]
		new_colname = get_fresh_col(res.columns)[0]
		function = {new_colname: new_col}
		res = res.assign(**function)

		# add arguments for the new column
		for index in arguments:
			arguments[index][new_colname] = []
			for i in self.cols:
				arguments[index][new_colname] += arguments[index][res.columns[i]]

		res = round_df(res)

		# print(res)
		return df_to_annotated_table_index_colname(res, self.func, arguments,
												   table, target_cols=new_colname)

	def to_dict(self):
		return {
			"type": "node",
			"op": "mutate_arithmetic",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.func, "func"),
				value_to_dict(self.cols, "cols")
			]}

	def infer_computation(self, inputs):
		if self.func != HOLE and self.cols != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		table = self.q.infer_computation(inputs)
		df = table.extract_values()
		#target = get_fresh_col(df.columns)[0]
		if self.func == HOLE:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, HOLE))
			table.add_column(new_col)
		else:
			new_col = []
			for i in range(table.get_row_num()):
				new_col.append(TableCell(HOLE, ExpNode(self.func, [HOLE])))
			table.add_column(new_col)
		return table

	def infer_cell(self, inputs, target):
		def infer_single_cell(loc, n):
			pre = [(loc[0], loc[1]), (n, loc[1])]
			if self.cols != HOLE and loc[0] not in self.cols:
					pre = [pre[0]]
			return pre
		pre_list = self.q.infer_cell(inputs, target)
		curr = []
		n = self.q.infer_colnum(inputs)
		for c in pre_list:
			curr += infer_single_cell(c, n)
		return curr

	def infer_cell_2(self, inputs):
		if self.func != HOLE and self.cols != HOLE:
			# the program has all parameters
			return self.eval(inputs)

		table = self.q.infer_cell_2(inputs)
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		if self.func == HOLE:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not be placed in group cols
						trace = [(x, rid) for x in range(colnum)]
					else:
						# other cells can only come from its previous pos
						trace = [(cid, rid)]
					args = []
					for c in trace:
						if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
							args += table.get_cell(c[0], c[1]).get_exp()
						else:
							args += [table.get_cell(c[0], c[1]).get_exp()]
					new_cell = TableCell(HOLE, args)
					new_source[-1].append(new_cell)
		else:
			for cid in range(colnum + 1):  # include new column
				new_source.append([])
				for rid in range(rownum):
					if cid == colnum:
						# the new cell in new column can come from any cell
						# but it should not be placed in group cols
						trace = [(x, rid) for x in range(colnum)]
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, ExpNode(self.func, args))
					else:
						# other cells can only come from its previous pos
						trace = [(cid, rid)]
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						new_cell = TableCell(HOLE, args)
					new_source[-1].append(new_cell)
		return AnnotatedTable(new_source, from_source=True)

	def infer_colnum(self, inputs):
		return self.q.infer_colnum(inputs) + 1

	def infer_rownum(self, inputs):
		return self.q.infer_rownum(inputs)

""" ----- utility functions -----"""
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
	# list of str types
	schema = [dtype_mapping(s) for s in df.infer_objects().dtypes]
	return schema


"""round data in the given df to degree two"""
def round_df(res):
	for col_name in res.columns:
		if res.dtypes[col_name] == 'double':
			res[col_name] = res[col_name].round(2)
	return res


def generate_direct_arguments(df, data_id=None):
	"""generate direct argument mapping for the given df
	returned argument format eg. {0: {'COL1':[(0,1,0), (0,1,1)]}}"""
	arguments = {}
	for index in df.index.tolist():
		arguments[index] = {}
		for col_name in df.to_dict().keys():
			arguments[index][col_name] = []
			if data_id is not None:
				arguments[index][col_name].append((data_id,
												   get_col_index_by_name(df, col_name),
												   index))
			else:
				arguments[index][col_name].append((get_col_index_by_name(df, col_name), index))  # only coordinate
	return arguments


def df_to_annotated_table_index_colname(df, op, arguments, table, target_cols=None):
	"""convert df to annotated table with given op and arguments (trace info)"""
	cell_list = []
	# print(df)
	# print(arguments)
	# print(df.index.tolist())
	for colName in df.columns.tolist():
		cid = get_col_index_by_name(df, colName)
		cell_list.append([])
		for index in df.index.tolist():
			# get full arguments for this level
			this_arguments = arguments[index][colName]
			# print(this_arguments)
			cell_arg = []
			if table is None:
				exp = this_arguments.copy()
			else:
				for arg in this_arguments:
					if isinstance(arg, ArgOr):
						cell_arg += [arg]
					else:
						temp_exp = table.get_cell(arg[0], arg[1]).get_exp()
						if not isinstance(temp_exp, list):
							temp_exp = [temp_exp]
						cell_arg += temp_exp
				if colName in target_cols:
					exp = ExpNode(op, cell_arg)
				else:
					exp = cell_arg
			cell_list[cid].append({"value": df.to_dict()[colName][index], "exp": exp})
	return AnnotatedTable(cell_list)


""" special handler for join"""
def df_to_annotated_table_join(df, op, arguments, table1, table2):
	"""special handler for join operation which include two tables"""
	cell_list = []
	for colName in df.columns.tolist():
		cid = get_col_index_by_name(df, colName)
		cell_list.append([])
		for index in df.index.tolist():
			# get full arguments for this level
			this_arguments = arguments[index][colName]
			args = []
			for arg in this_arguments:
				temp_exp = []
				if arg[0] == "table_1":
					temp_exp = table1.get_cell(arg[1], arg[2]).get_exp()
				elif arg[0] == "table_2":
					temp_exp = table2.get_cell(arg[1], arg[2]).get_exp()
				if not isinstance(temp_exp, list):
					temp_exp = [temp_exp]
				args += temp_exp
			cell_list[cid].append({"value": df.to_dict()[colName][index], "exp": args})
	#print(cell_list)
	return AnnotatedTable(cell_list)


def get_col_index_by_name(df, colName):
	return df.columns.get_loc(colName)

# not used after revisions
def get_index_by_rownum(df, rid):
	return df.index.tolist()[rid]


def get_value_by_row_col(df, rid, cid):
	return df.iloc[rid][df.columns[cid]]


def dict_to_program(l):
	def to_program(q, dict):
		operators = ["select", "filter", "group_sum", "group_mutate", "mutate_arithmetic", "join"]
		if dict["op"] not in operators:
			return None
		op = dict["op"]
		if op == "select":
			return Select(q, dict["0"])
		if op == "group_sum":
			return GroupSummary(q, dict["0"], dict["1"], dict["2"])
		if op == "group_mutate":
			return GroupMutate(q, dict["0"], dict["1"], dict["2"])
		if op == "mutate_arithmetic":
			return Mutate_Arithmetic(q, dict["0"], dict["1"])
		if op == "join":
			return Join(q, Table(dict["0"]))
	q = Table(0)
	for i in range(1, len(l)):
		q = to_program(q, l[i])
	return q
