from table import *
import unittest
import pandas as pd
from table_cell import *

HOLE = "_?_"

"""
a = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"},
     {"value": 5, "argument": [(5, 0, 1)], "operator": "sum", "attribute": "a"},
     {"value": 6, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])

b = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"}])
c = AnnotatedTable([{"value": 3, "argument": [(5, 0, 2)], "operator": "avg", "attribute": "a"}])
d = AnnotatedTable([{"value": 6, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])
e = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])
f = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": None, "attribute": "a"},
                   {"value": 6, "argument": None, "operator": "sum", "attribute": "a"}])
g = AnnotatedTable([{"value": None, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"}])

h = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"},
                    {"value": 5, "argument": [(5, 0, 1)], "operator": "sum", "attribute": "a"},
                    {"value": 6, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"},
                    {"value": 4, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "b"},
                    {"value": 4, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "b"},
                    {"value": 4, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "b"},
                    ])
"""

class TableTest(unittest.TestCase):
    # @unittest.skip
    def test_table_cell_argor(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=[ArgOr([(0, 1, 0), (0, 1, 2), (0, 1, 3)])],
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=[ArgOr([(0, 1, 0), (0, 1, 2), (0, 1, 3)])],
            attribute="a")
        print("---cell test 7---")
        print(output_cell.to_stmt())
        print(target_cell.to_stmt())
        print(target_cell.matches(output_cell))

    #@unittest.skip
    def test_table_cell_different_level(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=[ExpNode(
                op="lambda x, y: x + y",
                children=[
                    (0, 1, 0), (0, 1, 2), (0, 1, 3)
                ])],
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                    op="lambda x, y: x + y",
                    children=[
                        (0, 1, 0), (0, 1, 2), (0, 1, 3)]
                    ),
            attribute="a")
        print("---cell test 6---")
        print(output_cell.get_exp().to_dict())
        print(output_cell.to_stmt())
        print(target_cell.to_stmt())
        print(target_cell.matches(output_cell))

    @unittest.skip
    def test_table_cell_equiv_children_containment(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="lambda x, y: x + y",
                children=[
                    ExpNode(op="sum", children=[(0, 1, 1), (0, 1, 2)]),
                    ExpNode(op="sum", children=[(0, 2, 1), (0, 2, 2)])
                ]),
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="lambda x, y: x + y",
                children=[
                    ExpNode(op="sum", children=[(0, 3, 1), (0, 1, 1), (0, 1, 2)]),
                    ExpNode(op="sum", children=[(0, 2, 1), (0, 2, 2)])
                ]),
            attribute="a")
        print("---cell test 1---")
        print(output_cell.to_stmt())
        print(target_cell.to_stmt())
        print(target_cell.matches(output_cell))

    @unittest.skip
    def test_table_cell_equiv_mixed_argument_type(self):
        # f1(sum(1, 3), 2))
        target_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="lambda x, y: x / y",
                children=[
                    ExpNode(op="sum", children=[(0, 1, 1), (0, 1, 2)]),
                    (0, 2, 2)]),
            attribute="a")

        # f1(sum(1, 3, 4), 1, 2)
        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="lambda x, y: x / y",
                children=[
                    ExpNode(op="sum", children=[(0, 1, 1), (0, 1, 2), (0, 1, 3)]),
                    (0, 2, 1),
                    (0, 2, 2)]),
            attribute="a")

        print("---cell test 2---")
        print(target_cell.matches(output_cell))
        print(output_cell.get_exp().to_dict())
        print(output_cell.get_exp().to_flat_list())

    #@unittest.skip
    def test_table_cell_equiv_inconsistent_level(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[
                    ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 2, 1), (0, 2, 2)]),
                    (0, 3, 0)
                ]),
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[
                    ExpNode(op="lambda x, y: x + y", children=[(0, 2, 1), (0, 2, 2)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2),
                                                               ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2)])]),
                    (0, 3, 0)
                ]),
            attribute="a")
        print("---cell test 3---")
        print(target_cell.matches(output_cell))
        print(output_cell.get_exp().to_flat_list())

    @unittest.skip
    def test_table_cell_equiv_more_arguments(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[
                    ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 2, 1), (0, 2, 2)]),
                    (0, 3, 0)
                ]),
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[
                    (0, 4, 0),
                    (0, 5, 0),
                    ExpNode(op="sum", children=[(0, 2, 4), (0, 2, 5)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 2, 1), (0, 2, 2)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2)]),
                    (0, 3, 0)
                ]),
            attribute="a")
        print("---cell test 4---")
        print(target_cell.matches(output_cell))

    @unittest.skip
    def test_table_cell_equiv_empty_lists(self):
        # sum(f1(1, 3), f1(2, 3))
        # f1(sum(1, 3), sum(2, 3))
        target_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[
                    ExpNode(op="lambda x, y: x + y", children=[(0, 1, 1), (0, 1, 2)]),
                    ExpNode(op="lambda x, y: x + y", children=[(0, 2, 1), (0, 2, 2)]),
                    (0, 3, 0)
                ]),
            attribute="a")

        output_cell = TableCell(
            value=9,
            exp=ExpNode(
                op="sum",
                children=[]),
            attribute="a")
        print("---cell test 5---")
        print(target_cell.matches(output_cell))



    @unittest.skip
    def test_checker_function_find_mapping(self):
        d1 = [{"cust_country": "UK", "grade": 2, "outstanding_amt": 4000},
              {"cust_country": "USA", "grade": 2, "outstanding_amt": 6000},
              {"cust_country": "USA", "grade": 3, "outstanding_amt": 6000},
              {"cust_country": "India", "grade": 2, "outstanding_amt": 8000},
              {"cust_country": "UK", "grade": 2, "outstanding_amt": 6000},
              {"cust_country": "UK", "grade": 1, "outstanding_amt": 11000},
              {"cust_country": "USA", "grade": 3, "outstanding_amt": 3000}]
        d2 = [{"outstanding_amt": 6000},
              {"outstanding_amt": 11000},
              {"outstanding_amt": 3}]
        actual = load_from_dict(d1)
        target = load_from_dict(d2)
        print(target.to_dataframe())
        print(actual.to_dataframe())
        print(find_mapping(target, actual))

    @unittest.skip
    def test_checker_function_find_mapping_2(self):
        actual = AnnotatedTable(
            [{"value": HOLE, "argument": HOLE, "operator": HOLE, "attribute": "B"},
             {"value": 6, "argument": [(0, 0, 1)], "operator": "select", "attribute": "B"},
             {"value": 1, "argument": [(0, 0, 2)], "operator": "select", "attribute": "B"}]
        )
        target = AnnotatedTable(
            [{"value": None, "argument": None, "operator": None, "attribute": "B"},
             {"value": 6, "argument": None, "operator": None, "attribute": "B"},
             {"value": None, "argument": [(0, 0, 2)], "operator": "select", "attribute": "B"}]
        )
        print(target.to_dataframe())
        print(actual.to_dataframe())
        print(find_mapping(target, actual))

    # test search mapping
    @unittest.skip
    def test_checker_function1(self):
        """
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, ["a", "b", "c"])
        rlt = q.eval(inputs)
        """
        rlt = AnnotatedTable([{"value": 3, "argument": [(0, 0), (0, 1), (2, 0)], "operator": [], "attribute": "a"},
                              {"value": 3, "argument": [(0, 0), (0, 1), (2, 0)], "operator": [], "attribute": "a"},
                              {"value": 4, "argument": [(1, 0)], "operator": [], "attribute": "b"},
                              {"value": 5, "argument": [(1, 1)], "operator": [], "attribute": "b"},
                              {"value": 3, "argument": [(0, 0), (0, 1), (2, 0)], "operator": [], "attribute": "c"},
                              {"value": 6, "argument": [(2, 1)], "operator": [], "attribute": "c"}])
        print("---CheckerFunction1---")
        print(checker_function(rlt, a))
        print()

    @unittest.skip
    def test_checker_function2(self):
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0, 1, 2])
        rlt = q.eval(inputs)
        print("---CheckerFunction2---")
        print(checker_function(rlt, b))
        print()

    @unittest.skip
    def test_checker_function3(self):
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0, 1, 2])
        rlt = q.eval(inputs)
        print("---CheckerFunction3---")
        print(checker_function(rlt, c))
        x = a.get_cell(0, 0)
        y = a.get_cell(1, 0)
        print()

    @unittest.skip
    def test_checker_function4(self):
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0, 1, 2])
        rlt = q.eval(inputs)
        print(rlt.to_dataframe().to_csv())
        print()
        print(load_from_dict(test_data_2).to_dict())
        print("---CheckerFunction4---")
        print(checker_function(rlt, load_from_dict(test_data_2)))
        x = a.get_cell(0, 0)
        y = a.get_cell(1, 0)
        print()

    @unittest.skip
    def testExtractValue(self):
        df = pd.DataFrame([{"a": 3}, {"a": 5}, {"a": 6}])
        print("---ExtractValue---")
        print(h.extract_values())
        print()

    @unittest.skip
    def testAnnotatedTableToDict(self):
        expected_dict_list_b = [{"a": {"value": 3, "trace": {"operator": "avg", "argument": [(1, 0, 0), (5, 0, 1)]}}}]
        self.assertEqual(expected_dict_list_b, b.to_dict())

    @unittest.skip
    def testCheckerFunctionSubTable(self):
        pass

if __name__ == '__main__':
    unittest.main()