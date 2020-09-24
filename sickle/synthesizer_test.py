from table import *
import unittest
from table_ast import *
from synthesizer import *

# small parameter config for tests
test_config = {
                "operators": ["group_sum"],
                "filer_op": [">", "<", "=="],
                "constants": [0, 7500],
                "aggr_func": ["mean", "sum", "count"],
                "mutate_op": ["+", "-"]
            }

test_data_emp = [[{"empno": 7369, "depno": 20, "sal": 800},
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
                 {"empno": 7934, "depno": 10, "sal": 1300}]]

class SynthesizerTest(unittest.TestCase):
    def test_enum_sketches(self):
        c = Synthesizer(test_config).enum_sketches(test_data_emp, [], 2)  # currently no output to check
        print(c)
        candidates = Synthesizer(test_config).enumerative_all_programs(test_data_emp, [], 2)


if __name__ == '__main__':
    unittest.main()