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
    "select": lambda q: Select(q, cols=HOLE),
    "filter": lambda q: Filter(q, col_index=HOLE, op=HOLE, const=HOLE),
    "group_sum": lambda q: GroupSummary(q, group_cols=HOLE, aggr_func=HOLE, aggr_col=HOLE),
    "group_mutate": lambda q: GroupMutate(q, group_cols=HOLE, aggr_func=HOLE, target_col=HOLE),
    "mutate_arithmetic": lambda q: Mutate_Arithmetic(q, cols=HOLE, func=HOLE),
    "join": lambda q1, q2: Join(q1, q2, predicate=HOLE, is_left_outer=HOLE)
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
                "operators": ["select", "filter", "group_sum", "group_mutate", "mutate_arithmetic"],
                "filer_op": [">", "<", "=="],
                "constants": [],
                "aggr_func": ["mean", "sum", "count"],
                "mutate_op": ["+", "-"]
            }
        else:
            self.config = config

    def enum_sketches(self, inputs, output, size):
        """enumerate program sketches up to the given size"""
        """
        inp_val_set = set([v for t in inputs for r in t for k, v in r.items()] + [k for t in inputs for k in t[0]])
        out_val_set = set([v for r in output for k, v in r.items()])
        new_vals = out_val_set - inp_val_set
        sep_in_col_names = [key for t in inputs for key in t[0] if ('-' in key or '_' in key or '/' in key)]
        sep_in_content = [v for t in inputs for r in t for k, v in r.items() if
                          (isinstance(v, str) and ('-' in v or '_' in v or '/' in v))]
        has_sep = (len(sep_in_col_names) > 0) or (len(sep_in_content) > 0)
        """
        candidates = {}
        for level in range(0, size + 3):
            candidates[level] = []
        for level in range(0, size + 1):
            if level == 0:
                candidates[level] += [Table(data_id=i) for i in range(len(inputs))]
            else:
                for op in abstract_combinators:
                    # ignore operators that are not set
                    if op not in self.config["operators"]:
                        continue
                    if op == "join":
                        if level == 1:
                            # q = abstract_combinators[op]
                            # candidates[level].append(q)
                            # continue
                            for q0 in candidates[0]:
                                for q1 in candidates[0]:
                                    q = abstract_combinators[op](copy.copy(q0), copy.copy(q1))
                                    candidates[level].append(q)
                            """
                            for q1 in candidates[level - 1]:
                                for q2 in candidates[0]:
                                    q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))   #TODO
                                    candidates[level + 1].append(q)
                                for q2 in candidates[1]:
                                    q = abstract_combinators[op](copy.copy(q1), copy.copy(q2))
                                    candidates[level + 2].append(q)
                            """
                        else:
                            continue
                    else:
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
        return node.infer_domain(arg_id=var_path[-1], inputs=inputs, config=self.config)

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

    def run_synthesis(self, inputs, output, level_limit, logger, correct_out, with_analysis=True,
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
                logger.info(s.stmt_string() + f"   (program searched: {len(searched)})")
                # programs = []
                try:
                    stop = self.iteratively_instantiate(s, inputs, output, flat_out, candidates, searched, start_time,
                                                        solution_limit, time_limit_sec, "", print_trace, correct_out,
                                                        with_analysis, print_stmts=False)
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
                              time_limit_sec=None, solution_limit=None, print_trace=False):
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
                                                        with_analysis, print_stmts=True)
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
                                with_analysis, print_stmts=False, print_time=False):
        """iteratively instantiate abstract programs w/ promise check"""
        def instantiate(p, inputs, output, flat_out, indent):
            """instantiate programs and then check each one of them against the premise """
            results = []
            # get a flat version of output table for cheap check
            if p.is_abstract():
                if print_stmts:
                    print(indent + p.stmt_string())
                if time.time() - start_time > time_limit_sec:
                    return []

                ast = p.to_dict()
                """generate program instantitated from the most recent level
                    i.e., given an abstract program, it will enumerate all possible abstract programs that concretize
                """
                var_paths = self.pick_vars(ast, inputs)
                # there is no variables to instantiate
                if var_paths == []:
                    return []
                # find all variables at the innermost level
                innermost_level = max([len(p) for p in var_paths])
                target_vars = [p for p in var_paths if len(p) == innermost_level]
                recent_candidates = [ast]
                for var_path in target_vars:
                    temp_candidates = []
                    for partial_prog in recent_candidates:
                        # print(indent + Node.load_from_dict(partial_prog).stmt_string())
                        instantiated_progs = self.instantiate(partial_prog, var_path, inputs)
                        # pruning with running checker function on abstract program
                        valid_progs = []
                        for partial_p in instantiated_progs:
                            if time.time() - start_time > time_limit_sec:
                                break
                            pp = Node.load_from_dict(partial_p)
                            searched.append(pp)
                            if pp.is_abstract() and with_analysis:
                                if print_stmts:
                                    print(indent + pp.stmt_string())
                                infer_start = time.time()
                                infer_rlt = pp.infer_cell_2(inputs)
                                if print_time:
                                    print(f"infer cost: {time.time() - infer_start}")
                                # infer_rlt = pp.infer_cell_2(inputs)
                                # print(with_analysis)
                                cheap_check = False
                                expensive_check = with_analysis
                                # cheap_check
                                if cheap_check:
                                    # print(indent + "cheap check")
                                    flat_rlt = get_flat_table(infer_rlt)
                                    if checker_function(flat_rlt,
                                                        flat_out, check_relations=False,
                                                        print_result=print_trace) is None:
                                        if print_trace:
                                            print(indent + "cheap check failed!")
                                            print("=====Cheap Check Result=====")
                                            print(flat_rlt.to_dataframe())
                                            print("\n\n")
                                        continue
                                # expensive_check
                                check_start = time.time()
                                if expensive_check:
                                    # print(indent + "cell trace check 2")
                                    if checker_function(infer_rlt, output, print_result=print_trace,
                                                        print_time=print_time) is None:
                                        if print_trace:
                                            print(indent + "cell trace check 2 failed!")
                                            print("=====Cell Trace 2 Check Result=====")
                                            print(infer_rlt.to_dataframe())
                                            print("\n\n")
                                        if print_time:
                                            print(f"check cost: {time.time() - check_start}")
                                        continue
                                    else:
                                        if print_time:
                                            print(f"check cost: {time.time() - check_start}")
                                        if print_trace:
                                            print("***************passed******************************************")
                            valid_progs += [partial_p]
                        temp_candidates += valid_progs
                        if time.time() - start_time > time_limit_sec:
                            break
                        # number of candidate restriction
                    recent_candidates = temp_candidates
                    if time.time() - start_time > time_limit_sec:
                        break
                # level = innermost_level - 1
                next_level_programs = recent_candidates
                for _ast in next_level_programs:
                    results.append(Node.load_from_dict(_ast))
                return results
            else:
                return []
        if time_limit_sec is not None and time.time() - start_time > time_limit_sec:
            return True
        if p.is_abstract():
            candidates = instantiate(p, inputs, output, flat_out, indent)
            for _p in candidates:
                stop = self.iteratively_instantiate(_p, inputs, output, flat_out, results, searched, start_time,
                                                    solution_limit, time_limit_sec, indent + "     ", print_trace,
                                                    correct_out, with_analysis, print_stmts)
                if stop:
                    return True
            return False
        else:
            # handling concrete programs
            searched.append(p)
            curr_time = time.time()
            if print_stmts:
                print(indent + p.stmt_string())
            if print_trace:
                print(indent + "run checker_function!")
            eval_start = time.time()

            curr_out = p.eval(inputs)
            if print_time:
                print(f"eval cost: {time.time() - eval_start}")
            # compare against both user examples and correct label
            # if checker_function(curr_out, output, print_result=print_trace) is not None:
            #    results.append(p)
            if checker_function(curr_out, correct_out, print_result=print_trace, print_time=print_time) is not None:
                results.append(p)
            if (solution_limit is not None and len(results) >= solution_limit) \
                    or (time_limit_sec is not None and curr_time - start_time > time_limit_sec):
                # stop when enough progs being found
                return True
            return False

    # ---reference---
    def xiteratively_instantiate_with_premises_check(self, p, inputs, premise_chains, time_limit_sec=None):
        """iteratively instantiate abstract programs w/ promise check """
        def instantiate_with_premises_check(p, inputs, premise_chains):
            """instantiate programs and then check each one of them against the premise """
            results = []
            if p.is_abstract():
                print(p.stmt_string())
                ast = p.to_dict()
                next_level_programs, level = self.instantiate_one_level(ast, inputs)
                for _ast in next_level_programs:
                    # force terminate if the remaining time is running out
                    if time_limit_sec is not None and time.time() - start_time > time_limit_sec:
                        return results
                    premises_at_level = [[pm for pm in premise_chain if len(pm[1]) == level][0] for premise_chain in
                                         premise_chains]
                    subquery_res = None
                    for premise, subquery_path in premises_at_level:
                        if subquery_res is None:
                            # check if the subquery result contains the premise
                            subquery_node = get_node(_ast, subquery_path)
                            print("  {}".format(Node.load_from_dict(subquery_node).stmt_string()))
                            subquery_res = Node.load_from_dict(subquery_node).eval(inputs)
                        # print(subquery_res)
                        # print(subquery_res)
                        # if check_table_inclusion(premise.to_dict(orient="records"), subquery_res.to_dict(orient="records")):
                        if table.checker_function(premise,
                                                  subquery_res):
                            # debug
                            # p = Node.load_from_dict(_ast)
                            # if not p.is_abstract():
                            # 	print(f"{' - '}{p.stmt_string()}")
                            # 	print(subquery_res)
                            # 	print(premise)
                            # 	print( check_table_inclusion(premise.to_dict(orient="records"), subquery_res.to_dict(orient="records")))
                            results.append(Node.load_from_dict(_ast))
                            break
                return results
            else:
                return []

        print("time limit: {}".format(time_limit_sec))
        results = []
        if p.is_abstract():
            if time_limit_sec < 0:
                return []
            start_time = time.time()
            candidates = instantiate_with_premises_check(p, inputs, premise_chains)
            for _p in candidates:
                # if time_limit_sec is not None and time.time() - start_time > time_limit_sec:
                # 	return results
                remaining_time_limit = time_limit_sec - (
                            time.time() - start_time) if time_limit_sec is not None else None
                results += self.iteratively_instantiate_with_premises_check(_p, inputs, premise_chains,
                                                                            remaining_time_limit)
            return results
        else:
            # handling concrete programs won't take long, allow them to proceed
            return [p]

    def xenumerative_synthesis(self, inputs, output, max_prog_size, time_limit_sec=None, solution_limit=None):
        """Given inputs and output, enumerate all programs with premise check until
            find a solution p such that output ⊆ subseteq p(inputs) """
        #start_time = time.time()
        all_sketches = self.enum_sketches(inputs, output, size=max_prog_size)
        candidates = []
        for level, sketches in all_sketches.items():
            for s in sketches:
                print(s.stmt_string())
                ast = s.to_dict()
                # out_df = pd.DataFrame.from_dict(output)
                # out_df = remove_duplicate_columns(out_df)
                # all premise chains for the given ast
                # premise_chains = abstract_eval.backward_eval(ast, out_df)
                # remaining_time_limit = time_limit_sec - (
                #            time.time() - start_time) if time_limit_sec is not None else None
                programs = self.iteratively_instantiate_with_premises_check(s, inputs, premise_chains,
                                                                            remaining_time_limit)
                for p in programs:
                    try:
                        print(p.stmt_string())
                        t = p.eval(inputs)
                        # print(t.to_dict())
                        # print(t.extract_values())
                        # if align_table_schema(output, t.to_dict(orient="records")) != None:
                        if checker_function(t, output) is not None:
                            # print the result of annotated table
                            candidates.append(p)
                    except Exception as e:
                        print(f"[error] {sys.exc_info()[0]} {e}")
                        tb = sys.exc_info()[2]
                        tb_info = ''.join(traceback.format_tb(tb))
                        print(tb_info)
                # early return if the termination condition is met
                # time_limit may be exceeded if the synthesizer is stuck on iteratively instantiation
                # if time_limit_sec is not None and time.time() - start_time > time_limit_sec:
                #    return candidates
        return candidates



