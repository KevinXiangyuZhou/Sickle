from table import *
import unittest
from table_ast import *
from synthesizer import *
from tabulate import tabulate
import json
from table_cell import *
import random
import math

pd.set_option('expand_frame_repr', False)

HOLE = "_?_"

# small parameter config for tests
test_config = {
                "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "count", "max"],
                "mutate_func": ["mean", "sum", "max", "cumsum", "count", "rank"],
                "join_predicates": ["[(0, 1), (0, 0)]",
                                    "[(0, 1), (1, 0)]",
                                    "[(0, 0), (2, 3)]",
                                    "[(0, 1), (0, 1)]"],
                "mutate_function": ["lambda x, y: x - y",
                                    "lambda x, y: x + y",
                                    "lambda x, y: x * y",
                                    "lambda x, y: x / y",
                                    "lambda x: x - (x * 0.1)",
                                    "lambda x, y: y / (x - y)",
                                    "lambda x: 1",
                                    "lambda x, y, z: x - y / z",
                                    "lambda x: x * 1000"]
            }

test_config_10 = {
                "operators": ["select", "join", "mutate_2"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "count", "max"],
                "mutate_op": ["+2", "-2"],
                "mutate_function": ["lambda x, y: x - y"]
            }

test_config_11 = {
                "operators": ["group_sum"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum", "mean"],
                "mutate_op": ["+2", "-2"],
                "mutate_function": ["lambda x, y: x - y"]
            }

test_config_12 = {
                "operators": ["mutate_2"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "count", "max"],
                "mutate_op": ["+2", "-2"],
                "mutate_function": ["lambda x, y: x - y"]
            }

test_config_join = {
                "operators": ["join", "select"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum"],
                "mutate_op": ["+2", "-2"],
                "mutate_function": ["lambda x, y: x + y"]
            }

test_config_008 = {
                "operators": ["mutate_2"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "count", "max"],
                "mutate_op": ["sum"],
                "mutate_function": ["lambda x: x - (x * 0.1)"]
            }

test_config_009 = {
                "operators": ["mutate_2", "group_sum", "group_mutate"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "max"],
                "mutate_op": ["sum"],
                "mutate_function": ["lambda x, y: x / y - x "]
            }

test_config_010 = {
                "operators": ["mutate_2", "mutate"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "max"],
                "mutate_op": ["sum"],
                "mutate_function": ["lambda x, y: (x / y) * 100 "]
            }

test_config_list = {"008": test_config_008,
                    "009": test_config_009}

permutation_test = False  # edit this for permute user outputs
partial_table = False  # select random region of the table as demonstration
partial_trace = False  # trace info could be incomplete
level_limit = 5
time_limit = 900
solution_limit = 1
random_test = False

random.seed(7)

class SynthesizerTest(unittest.TestCase):
    @unittest.skip
    def test_run_single(self):
        with open('test_prog.json', 'r') as filehandler:
            data = json.load(filehandler)
            # join with arithmetic
            # description:
            inputs = data["input_data"]
            p = dict_to_program(data["exp_out1"])
            print(p.eval(inputs).to_dataframe())
            annotated_output = p.eval(inputs)
            # rlt = checker_function(computed_out, annotated_output, print_result=True)
            # print(rlt)


    # @unittest.skip
    def test_run(self):
        with open('testbenches/009.json', 'r') as filehandler:
            data = json.load(filehandler)
            # description:
            inputs = data["input_data"]
            correct_out = None
            if "exp_out" in data.keys():
                p = dict_to_program(data["exp_out"])
                annotated_output = p.eval(inputs)
                print(annotated_output.extract_values())
                print(annotated_output.to_dataframe())
                correct_out = copy.copy(annotated_output)
            else:
                print("load error")
            curr_config = test_config
            if "parameter_config" in data.keys():
                curr_config = data["parameter_config"]

            if permutation_test:
                columns = [i for i in range(annotated_output.get_col_num())]
                permutation_list = list(itertools.permutations(columns, annotated_output.get_col_num()))
                if len(permutation_list) > 100:
                    permutation_list = permutation_list[:100]
                # print(permutation_list)  # verify permutations of column ids
                output_candidates = [select_columns(annotated_output, selected)
                                     for selected in permutation_list]
                for i in range(len(output_candidates)):
                    att_out = output_candidates[i]
                    print("=======output candidates " + str(i) + "==========")
                    print(att_out.to_dataframe())
                print("\n\n\n")
                if random_test:
                    sample_id = random.randrange(len(output_candidates))
                else:
                    sample_id = 4 if 4 < len(output_candidates) else len(output_candidates) - 1
                annotated_output = output_candidates[sample_id]
                print("=======output candidates " + str(sample_id) + "==========")
                print(annotated_output.to_dataframe())
                print("===============================")
            if partial_table:
                if random_test:
                    x_start = random.randrange(annotated_output.get_col_num() / 2)
                    y_start = random.randrange(annotated_output.get_row_num() / 2)
                    x_end = random.randrange(annotated_output.get_col_num() / 2, annotated_output.get_col_num())
                    y_end = random.randrange(annotated_output.get_row_num() / 2, annotated_output.get_row_num())
                else:
                    x_end = annotated_output.get_col_num()
                    x_start = int(annotated_output.get_col_num() / 2)
                    y_end = annotated_output.get_row_num()
                    y_start = int(annotated_output.get_row_num() / 2)
                annotated_output = annotated_output.select_region((x_start, x_end), (y_start, y_end))

                print(annotated_output.to_dataframe())
            if partial_trace:
                annotated_output = annotated_output.randomize()
                print("=======with randomized trace==========")
                print(annotated_output.to_dataframe())
            # only include the first and last column
            # annotated_output = annotated_output.select_region((annotated_output.get_col_num() - 2, annotated_output.get_col_num()),
            #                                                   (0, 3))
            print("=======user sample==========")
            print(annotated_output.to_dataframe())
            candidates = []
            for i in range(4, level_limit + 1):
                candidates += Synthesizer(curr_config)\
                    .enumerative_synthesis(inputs, annotated_output, correct_out, i,
                                           solution_limit=solution_limit, time_limit_sec=time_limit, print_trace=False)
                if len(candidates) > 0:
                    break
            print("=======user sample==========")
            print(annotated_output.to_dataframe())
            print("=======correct output==========")
            print(correct_out.to_dataframe())
            for p in candidates:
                # print(alignment_result)
                print(p.stmt_string())
                print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
                print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
                print()
            print(f"number of programs: {len(candidates)}")
            print("\n\n\n\n\n\n")
            print("------------------------------------------------------------------------------------------")

    @unittest.skip
    def test_computation(self):
        with open('testbenches/034.json', 'r') as filehandler:
            # with open('testbenches/005.json', 'r') as filehandler:
            data = json.load(filehandler)
            # join with arithmetic
            # description:
            inputs = data["input_data"]
            # output = data["output_data"]
            compute1 = [{"0": 0},
                {"op": "group_mutate", "0": [1, 2], "1": HOLE, "2": HOLE}
              ]

            compute2 = [{"0": 0},
                        {"op": "group_sum", "0": [1], "1": HOLE, "2": HOLE},
                        {"op": "group_mutate", "0": HOLE, "1": HOLE, "2": HOLE }
                       ]

            compute3 = [{"0": 0},
                        {"op": "group_mutate", "0": [0], "1": HOLE, "2": HOLE},
                        {"op": "mutate_arithmetic", "0": HOLE, "1": HOLE}
                        ]

            compute09 = [{"0": 0},
                         {"op": "group_mutate", "0": [1, 2], "1": HOLE, "2": HOLE},
                         {"op": "group_mutate", "0": HOLE, "1": HOLE, "2": HOLE},
                         {"op": "mutate_arithmetic", "0": HOLE, "1": HOLE},
                         {"op": "group_sum", "0": HOLE, "1": HOLE, "2": HOLE}
                        ]
            compute26 = [{"0": 0},
                        {"op": "join", "0": 1, "1": "(0, 0)"},
                        {"op": "group_sum", "0": [1], "1": "count", "2": 0},
                        {"op": "group_mutate", "0": [], "1": "cumsum", "2": 1}]

            p = dict_to_program(compute26)  # select computation
            print(p.stmt_string())
            compute_rlt = p.infer_cell_2(inputs)
            # print(p.infer_cell_2(inputs).to_dataframe())

            # labeled expected output
            expp = dict_to_program(data["exp_out"])
            print(expp.eval(inputs).to_dataframe())
            print(expp.eval(inputs).extract_values())
            # annotated_output = expp.eval(inputs)
            # annotated_output = p.eval(inputs)

            # print(checker_function(compute_rlt, annotated_output))



    @unittest.skip
    def test_enum_sketches(self):
        c = Synthesizer(test_config).enum_sketches(test_data_emp_input, td, 2)  # currently no output to check
        print(c)
        # candidates = Synthesizer(test_config).enumerative_all_programs(test_data_emp_input, [], 2)
        candidates = Synthesizer(test_config).enumerative_search(test_data_emp_input, td, 2)
        print()
        for p in candidates:
            # print(alig
            # nment_result)
            print(p.stmt_string())
            print(p.eval(test_data_emp_input))

    @unittest.skip
    def test_1(self):
        inputs = [[
                {"empno": 7369, "depno": 20, "sal": 800},
                {"empno": 7499, "depno": 30, "sal": 1600},
                {"empno": 7521, "depno": 30, "sal": 1250},
                {"empno": 7566, "depno": 20, "sal": 2975},
                {"empno": 7654, "depno": 30, "sal": 1250},
                {"empno": 7698, "depno": 30, "sal": 2850},
                {"empno": 7782, "depno": 10, "sal": 2450},
                {"empno": 7788, "depno": 20, "sal": 3000},
                {"empno": 7839, "depno": 10, "sal": 5000},
                {"empno": 7844, "depno": 30, "sal": 1500},
                {"empno": 7876, "depno": 20, "sal": 1100},
                {"empno": 7900, "depno": 30, "sal": 950},
                {"empno": 7902, "depno": 20, "sal": 3000},
                {"empno": 7934, "depno": 10, "sal": 1300}
              ]]

        output = [
                {"depno": 10, "mean_sal": 2916.67},
                {"depno": 20, "mean_sal": 2175.00},
                {"depno": 30, "mean_sal": 1566.67}
              ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_synthesis(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_2(self):
        inputs = [[
                {"empno": 7369, "depno": 20, "sal": 800},
                {"empno": 7499, "depno": 30, "sal": 1600},
                {"empno": 7521, "depno": 30, "sal": 1250},
                {"empno": 7566, "depno": 20, "sal": 2975},
                {"empno": 7654, "depno": 30, "sal": 1250},
                {"empno": 7698, "depno": 30, "sal": 2850},
                {"empno": 7782, "depno": 10, "sal": 2450},
                {"empno": 7788, "depno": 20, "sal": 3000},
                {"empno": 7839, "depno": 10, "sal": 5000},
                {"empno": 7844, "depno": 30, "sal": 1500},
                {"empno": 7876, "depno": 20, "sal": 1100},
                {"empno": 7900, "depno": 30, "sal": 950},
                {"empno": 7902, "depno": 20, "sal": 3000},
                {"empno": 7934, "depno": 10, "sal": 1300}
              ]]
        output = [
                {"empno": 7369, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7499, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7521, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7566, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7654, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7698, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7782, "depno": 10, "mean_sal": 2916.67},
                {"empno": 7788, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7839, "depno": 10, "mean_sal": 2916.67},
                {"empno": 7844, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7876, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7900, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7902, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7934, "depno": 10, "mean_sal": 2916.67}
              ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_3(self):
        inputs = [[
                {"empno": 7369, "depno": 20, "sal": 800},
                {"empno": 7499, "depno": 30, "sal": 1600},
                {"empno": 7521, "depno": 30, "sal": 1250},
                {"empno": 7566, "depno": 20, "sal": 2975},
                {"empno": 7654, "depno": 30, "sal": 1250},
                {"empno": 7698, "depno": 30, "sal": 2850},
                {"empno": 7782, "depno": 10, "sal": 2450},
                {"empno": 7788, "depno": 20, "sal": 3000},
                {"empno": 7839, "depno": 10, "sal": 5000},
                {"empno": 7844, "depno": 30, "sal": 1500},
                {"empno": 7876, "depno": 20, "sal": 1100},
                {"empno": 7900, "depno": 30, "sal": 950},
                {"empno": 7902, "depno": 20, "sal": 3000},
                {"empno": 7934, "depno": 10, "sal": 1300}
              ]]
        output = [
                {"empno": 7369, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7499, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7521, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7566, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7654, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7698, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7782, "depno": 10, "mean_sal": 2916.67},
                {"empno": 7788, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7839, "depno": 10, "mean_sal": 2916.67},
                {"empno": 7844, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7876, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7900, "depno": 30, "mean_sal": 1566.67},
                {"empno": 7902, "depno": 20, "mean_sal": 2175.00},
                {"empno": 7934, "depno": 10, "mean_sal": 2916.67}
              ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_search(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_4(self):
        inputs = [[
                {"id": 1, "total": 24.78},
                {"id": 2, "total": 28.54},
                {"id": 3, "total": 48.69},
                {"id": 4, "total": -16.39},
                {"id": 5, "total": 29.92},
                {"id": 6, "total": 12.5},
                {"id": 7, "total": 10.2},
                {"id": 8, "total": 5.22}
              ]]
        output = [
            {"id": 1, "percentage": 0.82},
            {"id": 2, "percentage": 0.73},
            {"id": 3, "percentage": 1.50},
            {"id": 4, "percentage": -0.50},
            {"id": 5, "percentage": 0.70},
            {"id": 6, "percentage": 0.29},
            {"id": 7, "percentage": 0.26},
            {"id": 8, "percentage": 0.17}
        ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_5(self):
        inputs = [[
            {"trip_id": 944732, "date": "2015-09-24"},
            {"trip_id": 984595, "date": "2015-09-24"},
            {"trip_id": 984596, "date": "2015-09-24"},
            {"trip_id": 1129385, "date": "2015-09-24"},
            {"trip_id": 1030383, "date": "2015-09-30"},
            {"trip_id": 969490, "date": "2015-09-30"},
            {"trip_id": 947105, "date": "2015-09-30"},
            {"trip_id": 1011650, "date": "2015-11-16"}
        ]]
        output = [
            {"date": "2015-09-24", "count_trip_id": 4, "cum_sum_date": 4},
            {"date": "2015-09-30", "count_trip_id": 3, "cum_sum_date": 7},
            {"date": "2015-11-16", "count_trip_id": 1, "cum_sum_date": 8}
        ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_6(self):
        inputs = [[
                {"name": "Smith", "job": "Clerk", "salary": 800},
                {"name": "Allen", "job": "Salesman", "salary": 1600},
                {"name": "Ward", "job": "Salesman", "salary": 1250},
                {"name": "Jones", "job": "Manager", "salary": 800},
                {"name": "Blake", "job": "Manager", "salary": 2975},
                {"name": "Clark", "job": "Manager", "salary": 2850},
                {"name": "Turner", "job": "Salesman", "salary": 2450}
              ]]
        output = [
            {"name": "Allen", "job": "Salesman"},
            {"name": "Ward", "job": "Salesman"},
            {"name": "Blake", "job": "Manager"},
            {"name": "Clark", "job": "Manager"},
            {"name": "Turner", "job": "Salesman"}
        ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_search(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_7(self):
        # only join
        inputs = [[
            {"trip_id": 944732, "date": "2015-09-24"},
            {"trip_id": 984595, "date": "2015-09-24"},
            {"trip_id": 984596, "date": "2015-09-24"},
            {"trip_id": 1129385, "date": "2015-09-24"}
        ]
        ]
        output = [
            {"trip_id": 944732, "date1": "2015-09-24", "date2": "2015-09-24"},
            {"trip_id": 984595, "date1": "2015-09-24", "date2": "2015-09-24"},
            {"trip_id": 984596, "date1": "2015-09-24", "date2": "2015-09-24"},
            {"trip_id": 1129385, "date1": "2015-09-24", "date2": "2015-09-24"}
        ]
        # annotated_output = load_from_dict(output)
        annotated_output = AnnotatedTable(
            [{"value": None, "argument": [(0, 0, 1)], "operator": [], "attribute": ""},
             {"value": None, "argument": [(0, 0, 0)], "operator": [], "attribute": ""}]
        )
        candidates = Synthesizer(test_config_join).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe().to_csv())

    @unittest.skip
    def test_8(self):
        # join with arithmetic
        # description:
        inputs = [
        [
            {"empno": 7369, "depno": 20, "sal": 800},
            {"empno": 7499, "depno": 30, "sal": 1600},
            {"empno": 7521, "depno": 30, "sal": 1250},
            {"empno": 7566, "depno": 20, "sal": 2975}
        ],
        [
            {"empno": 7369, "depno": 20, "extra_sal": 3000},
            {"empno": 7521, "depno": 30, "extra_sal": 5000}
        ]
        ]
        output = [
            {"empno": 7369, "depno": 20, "sal": 3800},
            {"empno": 7521, "depno": 30, "sal": 6250}
        ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config_join).enumerative_search(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_10(self):
        # join with arithmetic
        # description:
        inputs = [
            [
                {"A": 1, "B": 1},
                {"A": 1, "B": 2}
            ]
        ]
        output = [
            {"B": 1, "A": 1, "C": 0},
            {"B": 2, "A": 1, "C": 1}
        ]
        annotated_output = AnnotatedTable(
            [{"value": 1, "argument": [(0, 1, 0)], "operator": [], "attribute": "B"},
             {"value": 2, "argument": [(0, 1, 1)], "operator": [], "attribute": "B"},
             {"value": 1, "argument": [(0, 0, 0)], "operator": [], "attribute": "A"},
             {"value": 1, "argument": [(0, 0, 1)], "operator": [], "attribute": "A"},
             {"value": 0, "argument": [(0, 1, 0), (0, 0, 0)], "operator": ["lambda x, y: x - y"], "attribute": "C"},
             {"value": 1, "argument": [(0, 1, 1), (0, 0, 1)], "operator": ["lambda x, y: x - y"], "attribute": "C"}]
        )
        candidates = Synthesizer(test_config_10).enumerative_synthesis(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
            print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
        print(f"number of programs: {len(candidates)}")

    @unittest.skip
    def test_11(self):
        # join with arithmetic
        # description:
        inputs = [
            [
                {"A": 1, "B": 1},
                {"A": 1, "B": 2}
            ]
        ]
        output = [
            {"B": 3, "A": 1},
        ]
        annotated_output = AnnotatedTable(
            [{"value": None, "argument": [(0, 1, 0), (0, 1, 1)], "operator": ["group_sum"], "attribute": "B"},
             {"value": None, "argument": [(0, 0, 0)], "operator": [], "attribute": "A"}]
        )
        candidates = Synthesizer(test_config_11).enumerative_synthesis(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
            print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
        print(f"number of programs: {len(candidates)}")

    @unittest.skip
    def test_12(self):
        # join with arithmetic
        # description:
        inputs = [
            [
                {"A": 1, "B": 1},
                {"A": 1, "B": 2}
            ]
        ]
        output = [
            {"B": 5, "A": 1, "C": 3},
            {"B": 6, "A": 1, "C": 4}
        ]
        annotated_output = AnnotatedTable(
            [{"value": 5, "argument": None, "operator": None, "attribute": "B"},
             {"value": 6, "argument": None, "operator": None, "attribute": "B"},
             {"value": 1, "argument": None, "operator": None, "attribute": "A"},
             {"value": 1, "argument": None, "operator": None, "attribute": "A"},
             {"value": 3, "argument": None, "operator": None, "attribute": "C"},
             {"value": 4, "argument": None, "operator": None, "attribute": "C"}]
        )
        candidates = Synthesizer(test_config_12).enumerative_synthesis(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
            print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
        print(f"number of programs: {len(candidates)}")

    @unittest.skip
    def test_13(self):
        # join with arithmetic
        # description:
        inputs = [
            [
                {"A": 1, "B": 1},
                {"A": 1, "B": 2}
            ]
        ]
        output = [
            {"B": 3, "A": 1},
        ]
        annotated_output = AnnotatedTable(
            [{"value": 1.50, "exp": ExpNode("mean", [(0, 1, 0)]), "attribute": "B"},
             {"value": 1, "exp": ExpNode("", []), "attribute": "A"}]
        )
        p = dict_to_program([{"0": 0}, {"op": "group_sum", "0": [0], "1": "mean", "2": 1}])
        print(p.eval(inputs).to_dataframe())
        annotated_output = p.eval(inputs)
        #print(p.eval(inputs).to_dict())
        print()
        candidates = Synthesizer(test_config_11).enumerative_synthesis(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
            print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
        print(f"number of programs: {len(candidates)}")



if __name__ == '__main__':
    unittest.main()