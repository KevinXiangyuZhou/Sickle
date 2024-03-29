import sys
import traceback
import copy
from pprint import pprint
import pandas as pd
from enum_strategies import *
from table import *
import time
import logging
# from table_cell import *

from table_ast import (HOLE, Node, Table, Select, Filter, GroupSummary,
                       GroupMutate, Mutate_Arithmetic, Join,
                       n_program_search, run_time, n_program_search_analysis, run_time_analysis)
# from eval_main import cheap_check_sw

abstract_combinators = {
    "join": lambda q1, q2, predicate, is_outer: Join(q1, q2, predicate=predicate, is_left_outer=is_outer),
    "select": lambda q: Select(q, cols=HOLE),
    "filter": lambda q: Filter(q, col_index=HOLE, op=HOLE, const=HOLE),
    "group_sum": lambda q: GroupSummary(q, group_cols=HOLE, aggr_func=HOLE, aggr_col=HOLE),
    "group_mutate": lambda q: GroupMutate(q, group_cols=HOLE, aggr_func=HOLE, target_col=HOLE),
    "mutate_arithmetic": lambda q: Mutate_Arithmetic(q, cols=HOLE, func=HOLE)
}

pd.set_option('display.max_colwidth', None)

def update_tree_value(node, path, new_val):
    """from a given ast node, locate the reference to the arg,
       and update the value"""
    for k in path:
        node = node["children"][k]
    node["value"] = new_val


def get_node(node, path):
    for k in path:
        node = node["children"][k]
    return node


class Synthesizer(object):
    def __init__(self, config=None):
        if config is None:
            self.config = {
                "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
                "aggr_func": ["mean", "sum", "count", "max", "min"],
                "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
                "join_predicates": [],
                "mutate_function": ["lambda x, y: x - y",
                                    "lambda x, y: x + y",
                                    "lambda x, y: x * y",
                                    "lambda x, y: x / y"
                                    ],
            }
        else:
            self.config = config
        self.program_memo = {}

    def enum_sketches(self, inputs, output, size):
        """enumerate program sketches up to the given size"""
        input_size = len(inputs)
        candidates = {}
        predicates = self.config["parameter_config"]["join_predicates"]
        for level in range(0, size + 3):
            candidates[level] = []

        if len(predicates) == 0:  # no join needed
            candidates[0] = [Table(data_id=i) for i in range(len(inputs))]
        else:
            if "join_outer" not in self.config["parameter_config"].keys():
                self.config["parameter_config"]["join_outer"] = [True, False]
            join_candidates = {}
            for join_level in range(0, len(predicates)):
                join_candidates[join_level] = []
                for is_outer in self.config["parameter_config"]["join_outer"]:
                    if join_level == 0:
                        q = abstract_combinators["join"](Table(data_id=0), Table(data_id=1), predicates[0], is_outer)
                        join_candidates[0].append(q)
                    else:
                        for joined in join_candidates[join_level - 1]:
                            q = abstract_combinators["join"](joined, Table(data_id=1 + join_level), predicates[join_level], is_outer)
                            join_candidates[join_level].append(q)
            # for k in join_candidates:
            candidates[0] += join_candidates[len(predicates) - 1]

        # check the "join" in the first level against user demonstration
        # only bring the ones that passed the check to the next level

        for level in range(1, size + 1):
            for op in abstract_combinators:
                # ignore operators that are not set
                if op not in self.config["operators"]:
                    continue
                if op == "join":
                    continue
                """
                if op == "join":
                    if level < input_size:  # enumerate all combinations of inputs
                        # q = abstract_combinators[op]
                        # candidates[level].append(q)
                        # continue
                        for q0 in candidates[level - 1]:
                            if not isinstance(q0, Join) and not isinstance(q0, Table):
                                continue
                            for q1 in candidates[0]:
                                for predicate in self.config["parameter_config"]["join_predicates"]:
                                    for outer_join in self.config["parameter_config"]["join_outer"]:
                                        q = abstract_combinators[op](copy.copy(q0), copy.copy(q1), predicate, outer_join)
                                        candidates[level].append(q)
                            for q1 in candidates[level - 1]:
                                for q2 in candidates[0]:
                                    q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))   #TODO
                                    candidates[level + 1].append(q)
                                for q2 in candidates[1]:
                                    q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))
                                    candidates[level + 2].append(q)
                        else:
                            continue
                """
                for q0 in candidates[level - 1]:
                    q = abstract_combinators[op](copy.copy(q0))
                    candidates[level].append(q)

        # for level in range(0, size + 1):
        #    candidates[level] = [q for q in candidates[level]]  # if not disable_sketch_limited(q)

        return candidates

    def pick_vars(self, ast, inputs):
        """list paths to all holes in the given ast"""
        def get_paths_to_all_holes(node):
            results = []
            for i, child in enumerate(node["children"]):
                if child["type"] == "node":
                    # try to find a variable to infer
                    paths = get_paths_to_all_holes(child)
                    for path in paths:
                        results.append([i] + path)
                elif child["value"] == HOLE:
                    # we find a variable to infer
                    results.append([i])
            return results
        return get_paths_to_all_holes(ast)

    def infer_domain(self, ast, var_path, inputs):
        # infer the set of parameters for nodes
        node = Node.load_from_dict(get_node(ast, var_path[:-1]))
        return node.infer_domain(arg_id=var_path[-1], inputs=inputs, config=self.config["parameter_config"])

    def instantiate(self, ast, var_path, inputs):
        """instantiate one hole in the program sketch"""
        domain = self.infer_domain(ast, var_path, inputs)
        candidates = []
        for val in domain:
            new_ast = copy.deepcopy(ast)
            update_tree_value(new_ast, var_path, val)
            candidates.append(new_ast)
        #print("~~~")
        #print(candidates)
        return candidates

    def instantiate_one_level(self, ast, inputs):
        """generate program instantitated from the most recent level
            i.e., given an abstract program, it will enumerate all possible abstract programs that concretize
        """
        var_paths = self.pick_vars(ast, inputs)
        # there is no variables to instantiate
        if var_paths == []:
            return [], []
        # find all variables at the innermost level
        innermost_level = max([len(p) for p in var_paths])
        target_vars = [p for p in var_paths if len(p) == innermost_level]
        recent_candidates = [ast]
        for var_path in target_vars:
            temp_candidates = []
            for partial_prog in recent_candidates:
                temp_candidates += self.instantiate(partial_prog, var_path, inputs)
            recent_candidates = temp_candidates

        concrete_program_level = innermost_level - 1
        return recent_candidates, concrete_program_level

    def iteratively_instantiate_and_print(self, p, inputs, level, print_programs=False):
        """iteratively instantiate a program (for the purpose of debugging)"""
        if print_programs:
            print(f"{'  '.join(['' for _ in range(level)])}{p.stmt_string()}")
        results = []
        if p.is_abstract():
            ast = p.to_dict()
            var_path = self.pick_vars(ast, inputs)[0]
            # domain = self.infer_domain(ast, path, inputs)
            candidates = self.instantiate(ast, var_path, inputs)
            for c in candidates:
                nd = Node.load_from_dict(c)
                results += self.iteratively_instantiate_and_print(nd, inputs, level + 1, print_programs)
            return results
        else:
            return [p]

    """ basic enumeration"""
    def enumerative_all_programs(self, inputs, output, max_prog_size):
        """Given inputs and output, enumerate all programs in the search space until
            find a solution p such that output ⊆ subseteq p(inputs)  """

        all_sketches = self.enum_sketches(inputs, output, size=max_prog_size)
        concrete_programs = []
        for level, sketches in all_sketches.items():
            for s in sketches:
                concrete_programs += self.iteratively_instantiate_and_print(s, inputs, 1, True)
        for p in concrete_programs:
            try:
                t = p.eval(inputs)
                print(p.stmt_string())
                print(t)
            except Exception as e:
                print(f"[error] {sys.exc_info()[0]} {e}")
                tb = sys.exc_info()[2]
                tb_info = ''.join(traceback.format_tb(tb))
                print(tb_info)
        print("----")
        print(f"number of programs: {len(concrete_programs)}")

    def enumerative_search(self, inputs, output, max_prog_size):
        """Given inputs and output, enumerate all programs in the search space until
            find a solution p such that output ⊆ subseteq p(inputs)  """
        all_sketches = self.enum_sketches(inputs, output, size=max_prog_size)
        candidates = []
        for level, sketches in all_sketches.items():
            for s in sketches:
                concrete_programs = self.iteratively_instantiate_and_print(s, inputs, 1)
                for p in concrete_programs:
                    try:
                        print(p.stmt_string())
                        t = p.eval(inputs)
                        # if align_table_schema(output, t.to_dict(orient="records")) != None:
                        if checker_function(t, output) is not None:
                            # print the result of annotated table
                            candidates.append(p)
                    except Exception as e:
                        print(f"[error] {sys.exc_info()[0]} {e}")
                        tb = sys.exc_info()[2]
                        tb_info = ''.join(traceback.format_tb(tb))
                        print(tb_info)
        print("----")
        print(f"number of programs: {len(candidates)}")
        return candidates

    def run_synthesis(self, inputs, output, level_limit, logger, correct_out, with_analysis=True, use_val=False,
                      time_limit_sec=None, solution_limit=None, print_trace=False):
        """Given inputs and output, enumerate all programs with premise check until
            find a solution p such that output ⊆ subseteq p(inputs) """
        all_sketches = self.enum_sketches(inputs, output, size=level_limit)
        candidates = []
        start_time = time.time()
        searched = []
        # flat_out = get_flat_table(output)
        flat_out = None
        log_s = logging.getLogger("summary")
        for level, sketches in all_sketches.items():
            for s in sketches:
                # logger.info(s.stmt_string() + f"   (program searched: {len(searched)})")
                # programs = []
                try:
                    stop = self.iteratively_instantiate(s, inputs, output, flat_out, candidates, searched, start_time,
                                                        solution_limit, time_limit_sec, "", print_trace, correct_out,
                                                        with_analysis, use_val, print_stmts=False)
                    if stop:
                        finish_time = time.time() - start_time
                        print("[stop]")
                        self.log_info(logger, log_s, len(searched), finish_time, len(candidates), with_analysis)
                        return candidates, finish_time, len(searched)
                except Exception as e:
                    logger.info(f"[error] {sys.exc_info()[0]} {e}")
                    tb = sys.exc_info()[2]
                    tb_info = ''.join(traceback.format_tb(tb))
                    logger.info(tb_info)
        finish_time = time.time() - start_time
        if finish_time >= time_limit_sec or len(candidates) > 0:
            self.log_info(logger, log_s, len(searched), finish_time, len(candidates), with_analysis)
            if finish_time >= time_limit_sec:
                log_s.info("[TIME OUT]")
        else:
            log_s.info(f"no results found after {finish_time}s")
        return candidates, finish_time, len(searched)

    def log_info(self, logger, log_s, searched_len, finish_time, candidates_len, with_analysis):
        logger.info("----")
        logger.info(f"number of programs searched: {searched_len}")
        logger.info("time cost: " + str(finish_time))
        logger.info(f"number of solutions: {candidates_len}")
        # log the statistics into summary file
        if with_analysis:
            n_program_search_analysis.append(searched_len)
            run_time_analysis.append(finish_time)
        else:
            n_program_search.append(searched_len)
            run_time.append(finish_time)
        log_s.info(f"number of programs searched: {searched_len}")
        log_s.info("time cost: " + str(finish_time))
        log_s.info(f"number of solutions: {candidates_len}")

    def enumerative_synthesis(self, inputs, output, correct_out, max_prog_size,
                              time_limit_sec=None, solution_limit=None, print_trace=False, use_val=False):
        """Given inputs and output, enumerate all programs with premise check until
            find a solution p such that output ⊆ subseteq p(inputs) """
        all_sketches = self.enum_sketches(inputs, output, size=max_prog_size)
        candidates = []
        start_time = time.time()
        searched = []
        # flat_out = get_flat_table(output)
        flat_out = None
        # turn on cheap analysis
        with_analysis = True
        for level, sketches in all_sketches.items():
            for s in sketches:
                try:
                    stop = self.iteratively_instantiate(s, inputs, output, flat_out, candidates, searched, start_time,
                                                        solution_limit, time_limit_sec, "", print_trace, correct_out,
                                                        with_analysis, use_val, print_stmts=True)
                    if stop:
                        finsh_time = time.time() - start_time
                        print("----")
                        print(f"number of programs searched: {len(searched)}")
                        print(f"number of solutions: {len(candidates)}")
                        print("time cost: " + str(finsh_time))
                        print()
                        return candidates
                except Exception as e:
                    print(f"[error] {sys.exc_info()[0]} {e}")
                    tb = sys.exc_info()[2]
                    tb_info = ''.join(traceback.format_tb(tb))
                    print(tb_info)
        finsh_time = time.time() - start_time
        print("----")
        print(f"number of programs: {len(candidates)}")
        print("time cost: " + str(finsh_time))
        print()
        return candidates

    def iteratively_instantiate(self, p, inputs, output, flat_out, results, searched, start_time,
                                solution_limit, time_limit_sec, indent, print_trace, correct_out,
                                with_analysis, use_val=False, print_stmts=False, print_time=False):
        """iteratively instantiate abstract programs w/ promise check"""
        def instantiate(p, inputs, output, flat_out, indent):
            """instantiate programs and then check each one of them against the premise """
            curr_indent = indent
            # get a flat version of output table for cheap check
            results = []
            if p.is_abstract():
                # switches for checks
                cheap_check = False
                value_check = use_val
                expensive_check = not value_check
                print(curr_indent + p.stmt_string())
                # if print_stmts:
                #     print(curr_indent + p.stmt_string())
                if time.time() - start_time > time_limit_sec:
                    return [], curr_indent

                ast = p.to_dict()
                if isinstance(p, Join):  # if it is join then skip
                    return [], curr_indent
                # join cannot appear on the last level
                """generate program instantitated from the most recent level
                    i.e., given an abstract program, it will enumerate all possible abstract programs that concretize
                """
                # value check
                if with_analysis and value_check:
                    searched.append(p)
                    # print(curr_indent + p.stmt_string())
                    if print_stmts:
                        print(curr_indent + p.stmt_string())
                    infer_rlt = p.infer_cell(inputs, self.config["parameter_config"])
                    if checker_function(infer_rlt, output, print_result=print_trace,
                                        print_time=print_time, cmp_val=True) is None:
                        if print_trace:
                            print(curr_indent + "falx check failed!")
                            print("=====Falx Check Result=====")
                            print(infer_rlt.to_dataframe())
                            print("\n\n")
                        return [], curr_indent
                    else:
                        if print_trace:
                            print(curr_indent + "value check passed!")
                var_paths = self.pick_vars(ast, inputs)
                # there is no variables to instantiate
                if var_paths == []:
                    return [], curr_indent
                # find all variables at the innermost level
                innermost_level = max([len(p) for p in var_paths])
                # if this is the last level to be instantiate
                if innermost_level == 1 and expensive_check and with_analysis:
                    concrete_traces = extract_last_level(output)
                    infer_rlt = p.infer_cell_2(inputs, self.config["parameter_config"])
                    infer_rlt_compressed = infer_rlt.compress_sum()
                    # if not check_concrete(infer_rlt, concrete_traces) \
                    if not check_concrete(infer_rlt_compressed, concrete_traces):
                        if print_trace:
                            print(curr_indent + p.stmt_string())
                            print(curr_indent + "X (pruned by backward reasoning)")
                        if print_trace:
                            print(infer_rlt.to_dataframe())
                        return [], curr_indent
                target_vars = [p for p in var_paths if len(p) == innermost_level]
                recent_candidates = [ast]
                for var_path in target_vars:
                    temp_candidates = []
                    for partial_prog in recent_candidates:
                        # print(curr_indent + Node.load_from_dict(partial_prog).stmt_string())
                        instantiated_progs = self.instantiate(partial_prog, var_path, inputs)
                        # pruning with running checker function on abstract program
                        valid_progs = []
                        for partial_p in instantiated_progs:
                            if not use_val:
                                searched.append(p)
                            if time.time() - start_time > time_limit_sec:
                                break
                            pp = Node.load_from_dict(partial_p)
                            # print(pp.program_list())
                            if pp.is_abstract() and with_analysis:
                                check_start = time.time()
                                if not value_check:
                                    if print_stmts:
                                        print(curr_indent + pp.stmt_string())
                                    infer_start = time.time()
                                    if pp.stmt_string() not in self.program_memo:
                                        infer_rlt = pp.infer_cell_2(inputs, self.config["parameter_config"])
                                        self.program_memo[pp.stmt_string()] = infer_rlt
                                    else:
                                        infer_rlt = self.program_memo[pp.stmt_string()]
                                    infer_rlt_compressed = infer_rlt.compress_sum()
                                    if print_time:
                                        print(f"infer cost: {time.time() - infer_start}")
                                    # print(indent + "cell trace check 2")
                                    # if checker_function(infer_rlt, output, print_result=print_trace,
                                    #                     print_time=print_time) is None \
                                    if checker_function(infer_rlt_compressed, output, print_result=print_trace,
                                                        print_time=print_time) is None:
                                        if print_trace:
                                            print(curr_indent + "X")
                                        if print_trace:
                                            print(curr_indent + "cell trace check 2 failed!")
                                            print("=====Cell Trace 2 Check Result=====")
                                            print(infer_rlt.to_dataframe())
                                            print("\n\n")
                                        if print_time:
                                            print(f"check cost: {time.time() - check_start}")
                                        continue
                                    else:
                                        print(curr_indent + "analysis passed!")
                                        if print_time:
                                            print(f"check cost: {time.time() - check_start}")
                            valid_progs += [partial_p]
                        temp_candidates += valid_progs
                        if time.time() - start_time > time_limit_sec:
                            break
                        # number of candidate restriction
                    recent_candidates = temp_candidates
                    if recent_candidates:
                        curr_indent += "     "
                    if time.time() - start_time > time_limit_sec:
                        break
                # level = innermost_level - 1
                next_level_programs = recent_candidates
                for _ast in next_level_programs:
                    results.append(Node.load_from_dict(_ast))
                return results, curr_indent
            else:
                return [], curr_indent
        if time_limit_sec is not None and time.time() - start_time > time_limit_sec:
            return True
        if p.is_abstract():
            candidates, curr_indent = instantiate(p, inputs, output, flat_out, indent)
            for _p in candidates:
                stop = self.iteratively_instantiate(_p, inputs, output, flat_out, results, searched, start_time,
                                                    solution_limit, time_limit_sec, curr_indent, print_trace,
                                                    correct_out, with_analysis, use_val, print_stmts)
                if stop:
                    return True
            return False
        else:
            """
            stmt = p.stmt_string()
            
            print(stmt)
            if stmt == "t0 <- table_ref(0); t1 <- group_sum(t0, (0, 2, 4), sum, 1); 
            t2 <- group_mutate(t1, (0,), cumsum, 3); t3 <- mutate_arithmetic(t2, lambda x, y: x / y * 100, (4, 2))":
                return True
            return False
            """
            if not with_analysis or use_val:
                searched.append(p)
            # handling concrete programs
            curr_time = time.time()
            # if print_stmts:
            print(indent + p.stmt_string())
            # if print_trace:
            # print(indent + "run checker_function!")
            eval_start = time.time()
            curr_out = p.eval(inputs).compress_sum()
            # curr_out = p.eval(inputs)
            if print_time:
                print(f"eval cost: {time.time() - eval_start}")

            if not use_val and with_analysis:
                concrete_traces = extract_last_level(output)
                if check_concrete_list(extract_last_level(curr_out), concrete_traces)\
                        and checker_function(curr_out, output, print_result=print_trace, print_time=print_time,
                                             check_value=True) is not None:
                    results.append(p)
                    print(indent + "checker_function Passed!")
                else:
                    if print_trace:
                        print(indent + "X")
            else:
                if checker_function(curr_out, output, print_result=print_trace, print_time=print_time, check_value=True) is not None:
                    results.append(p)
                    print(indent + "checker_function Passed!")
                else:
                    if print_trace:
                        print(indent + "X")

            if correct_out is None and len(results) > solution_limit:
                return True

            if (correct_out is not None and (curr_out.equals(correct_out))) \
                             or (time_limit_sec is not None and curr_time - start_time > time_limit_sec):
                # stop when we find the program that generate correct output
                # print(curr_out.to_dataframe())
                print("process stopped")
                return True
            return False
