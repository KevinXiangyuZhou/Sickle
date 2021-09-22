from table import *
from table_ast import *
from synthesizer import *
from tabulate import tabulate
import json
from table_cell import *

test_config = {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": [],
            "constants": [],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": [],
            "join_outer": [],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                ]
        },
        "permutation_test": False,
        "random_test": False,
        "partial_table": False,
        "partial_trace": False,
        "level_limit": 6,
        "time_limit": 300,
        "solution_limit": 1
    }


level_limit = 7
time_limit = 600
solution_limit = 1


def run_example(file_path):
    with open(file_path, 'r') as filehandler:
        data = json.load(filehandler)
        # description:
        inputs = data["input_data"]

        if "example" in data.keys():
            trace = data["example"]
            annotated_output = read_traces(trace)
        else:
            print("error while loading!")
            return
        curr_config = test_config
        if "parameter_config" in data.keys():
            curr_config["parameter_config"] = data["parameter_config"]
        else:
            curr_config = test_config

        """
        source = [[TableCell('A', ['0_a0']), TableCell('A', ['0_a7'])],
                  [TableCell(32.86, ExpNode('lambda x, y: x / y * 100',
                                           [ExpNode('sum', ['0_b0', '0_b1']), '0_e0'])),
                   TableCell(61.39, ExpNode('lambda x, y: x / y * 100',
                                           [ExpNode('sum', ['0_b0', '0_b1', "_UNK_", '0_b7']), '0_e7']))
                   ]]
        annotated_output = AnnotatedTable(source, from_source=True)
        """

        print("=======user sample==========")
        print(annotated_output.to_dataframe())

        candidates = []
        for i in range(6, level_limit):
            candidates += Synthesizer(curr_config) \
                .enumerative_synthesis(inputs, annotated_output, None, i,
                                       solution_limit=solution_limit, time_limit_sec=time_limit)
            if len(candidates) > 0:
                break

        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(tabulate(p.eval(inputs).compress_sum().extract_values(), headers='keys', tablefmt='psql'))
            print(tabulate(p.eval(inputs).compress_sum().extract_traces(), headers='keys', tablefmt='psql'))
            print()
        print(f"number of programs: {len(candidates)}")
        print("\n\n\n\n\n\n")
        print("------------------------------------------------------------------------------------------")


# [{'op': 'lambda x, y: x / y * 100', ’children': [{}, {}]},
# [ExpNode('sum', ['0_b0', '0_b1']), '0_e0'])),
#                    TableCell(61.39, ExpNode('lambda x, y: x / y * 100',
#                                            [ExpNode('sum', ['0_b0', '0_b1', "_UNK_", '0_b7']), '0_e7']
def read_traces(trace):
    def build(exp_trace):
        if not isinstance(exp_trace, dict):
            return exp_trace
        return ExpNode(exp_trace["op"], [build(child) for child in exp_trace["children"]])
    source = [[]]
    for rid in range(len(trace)):
        for cid in range(len(trace[0])):
            if cid >= len(source):
                source.append([])
            source[cid].append(TableCell(HOLE, build(trace[rid][cid])))
    return AnnotatedTable(source, from_source=True)


if __name__ == '__main__':
    run_example(file_path='./test_example.json')
