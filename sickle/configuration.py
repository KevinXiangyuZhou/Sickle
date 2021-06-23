# small parameter config for tests
HOLE = "_?_"
config = {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "permutation_test": False,
        "random_test": False,
        "partial_table": False,
        "partial_trace": False,
        "level_limit": 5,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 200
    }

target_configs = [
    {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": False,
        "permutation_test": True,
        "random_test": True,
        "partial_table": False,
        "partial_trace": True,
        "level_limit": 5,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 200
    },
    {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": True,
        "permutation_test": True,
        "random_test": True,
        "partial_table": False,
        "partial_trace": True,
        "level_limit": 5,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 200
    },
{
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": False,
        "permutation_test": True,
        "random_test": True,
        "partial_table": True,
        "partial_trace": True,
        "level_limit": 5,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 200
    },
    {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": True,
        "permutation_test": True,
        "random_test": True,
        "partial_table": True,
        "partial_trace": True,
        "level_limit": 5,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 200
    },
{
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": True,
        "permutation_test": True,
        "random_test": True,
        "partial_table": True,
        "partial_trace": False,
        "level_limit": 6,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 300
    },
{
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": False,
        "permutation_test": True,
        "random_test": True,
        "partial_table": True,
        "partial_trace": False,
        "level_limit": 6,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 300
    },
    {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000"
                                ]
        },
        "with_analysis": True,
        "permutation_test": True,
        "random_test": True,
        "partial_table": True,
        "partial_trace": False,
        "level_limit": 6,
        "time_limit": 600,
        "solution_limit": 1,
        "row_limit": 50
    }

]
