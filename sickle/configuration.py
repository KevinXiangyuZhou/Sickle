# small parameter config for tests
config = {
            "operators": ["group_sum", "mutate_arithmetic", "group_mutate"],
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