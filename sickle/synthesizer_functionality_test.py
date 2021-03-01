from table import *
import unittest
from table_ast import *
from synthesizer import *
from tabulate import tabulate
import json
from table_cell import *

pd.set_option('expand_frame_repr', False)

# small parameter config for tests
test_config = {
                "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean", "sum", "count", "max"],
                "mutate_func": ["mean", "sum", "max", "cumsum"],
                "mutate_op": ["sum"],
                "mutate_function": ["lambda x, y: x - y",
                                    "lambda x, y: x + y",
                                    "lambda x, y: x * y",
                                    "lambda x, y: x / y",
                                    "lambda x: x - (x * 0.1)",
                                    "lambda x, y: y / (x - y)",
                                    "lambda x: 1"]
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
"""
annotated_output_009 = AnnotatedTable([
    {"value": "E.Els", "exp": ExpNode("", [ArgOr([(0, 0, 0), (0, 0, 1)])]), "attribute": "player"},
    {"value": "E.Els", "exp": ExpNode("", [ArgOr([(0, 0, 0), (0, 0, 1)])]), "attribute": "player"},
    {"value": 1, "exp": ExpNode("", [(0, 1, 1)]), "attribute": "rd"},
    {"value": 2, "exp": ExpNode("", [(0, 1, 1)]), "attribute": "rd"},
    {"value": None, "exp": ExpNode("", [(0, 3, 0), (0, 4, 0)]), "operator": ["group_mean", "sum", "max", "lambda x, y: x / y - x "], "attribute": "efficiency"},
    {"value": None, "argument": [(0, 3, 0), (0, 4, 0)], "operator": ["group_mean", "sum", "max", "lambda x, y: x / y - x "], "attribute": "efficiency"}
    ])

annotated_output_010 = AnnotatedTable([
    {"value": "Alabama", "exp": ExpNode("", [(0, 0, 0)]), "attribute": "state"},
    {"value": "Alaska", "exp": ExpNode("", [(0, 0, 1)]), "attribute": "state"},
    {"value": 1667, "exp": ExpNode("", [(0, 1, 0)]), "attribute": "count"},
    {"value": 507, "exp": ExpNode("", [(0, 1, 1)]), "attribute": "count"},
    {"value": None, "exp": ExpNode("lambda x, y: (x / y) * 100", [(0, 1, 0)]), "attribute": "percentage"},
    {"value": None, "exp": ExpNode("lambda x, y: (x / y) * 100", [(0, 1, 1)]), "attribute": "percentage"}
    ])


td = AnnotatedTable([
    {"value": 2916.67, "argument": [(2450, 2, 6)], "operator": "", "attribute": "mean_sal"},
    {"value": 2175.00, "argument": [(800, 2, 0)], "operator": "", "attribute": "mean_sal"},
    {"value": 1566.67, "argument": [(1600, 2, 1)], "operator": "", "attribute": "mean_sal"}
    ])
"""

class SynthesizerTest(unittest.TestCase):
    @unittest.skip
    def test_run_005(self):
        with open('test_prog.json', 'r') as filehandler:
            data = json.load(filehandler)
            # join with arithmetic
            # description:
            inputs = data["input_data"]
            p = dict_to_program(data["exp_out1"])
            print(p.eval(inputs).to_dataframe())
            annotated_output = p.eval(inputs)

            #rlt = checker_function(computed_out, annotated_output, print_result=True)
            #print(rlt)


    #@unittest.skip
    def test_run(self):
        with open('../benchmark/tests/005.json', 'r') as filehandler:
        #with open('testbenches/005.json', 'r') as filehandler:
            data = json.load(filehandler)
            # join with arithmetic
            # description:
            inputs = data["input_data"]
            #output = data["output_data"]

            if "exp_out" in data.keys():
                p = dict_to_program(data["exp_out"])
                print(p.eval(inputs).to_dataframe())
                annotated_output = p.eval(inputs)
            else:
                # annotated_output = load_from_dict(output)
                print("load error")
            candidates = []
            for i in range(1, 6):
                candidates = Synthesizer(test_config).enumerative_synthesis(inputs, annotated_output, i)
                if len(candidates) > 0:
                    break
            for p in candidates:
                # print(alignment_result)
                print(p.stmt_string())
                print(tabulate(p.eval(inputs).extract_values(), headers='keys', tablefmt='psql'))
                print(tabulate(p.eval(inputs).extract_traces(), headers='keys', tablefmt='psql'))
                print()
            print(f"number of programs: {len(candidates)}")


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