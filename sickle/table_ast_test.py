from table import *
import unittest
from table_ast import *


test_data_0 = [{"Totals":7,"Value":"A","variable":"alpha","value":2,"cumsum":2},
             {"Totals":8,"Value":"B","variable":"alpha","value":2,"cumsum":2},
             {"Totals":9,"Value":"C","variable":"alpha","value":3,"cumsum":3},
             {"Totals":9,"Value":"D","variable":"alpha","value":3,"cumsum":3},
             {"Totals":9,"Value":"E","variable":"alpha","value":4,"cumsum":4},
             {"Totals":7,"Value":"A","variable":"beta","value":2,"cumsum":4},
             {"Totals":8,"Value":"B","variable":"beta","value":3,"cumsum":5},
             {"Totals":9,"Value":"C","variable":"beta","value":3,"cumsum":6},
             {"Totals":9,"Value":"D","variable":"beta","value":4,"cumsum":7},
             {"Totals":9,"Value":"E","variable":"beta","value":3,"cumsum":7},
             {"Totals":7,"Value":"A","variable":"gamma","value":3,"cumsum":7},
             {"Totals":8,"Value":"B","variable":"gamma","value":3,"cumsum":8},
             {"Totals":9,"Value":"C","variable":"gamma","value":3,"cumsum":9},
             {"Totals":9,"Value":"D","variable":"gamma","value":2,"cumsum":9},
             {"Totals":9,"Value":"E","variable":"gamma","value":2,"cumsum":9}]
test_data_1 = [{"a": 3, "b": 4},
               {"a": 5},
               {"a": 6, "b": 7, "c": 8}]
inputs = {0: pd.DataFrame.from_dict(test_data_0),
          1: pd.DataFrame.from_dict(test_data_1)}

class AstTest(unittest.TestCase):

    def testSelect(self):
        q = Table(data_id=1)
        # print(q.eval(inputs).to_dict())
        q = Select(q, ["a"])
        rlt = q.eval(inputs)
        annotated = [{"a" : {"value": 3, "trace": {"operator": "select", "argument": [(3, 0, 0)]}}},
             {"a" : {"value": 5, "trace": {"operator": "select", "argument": [(5, 0, 1)]}}},
             {"a" : {"value": 6, "trace": {"operator": "select", "argument": [(6, 0, 2)]}}}]
        print("---Select---")
        print(rlt.to_dict())
        print()
        # self.assertEqual(rlt.to_dict(), annotated)

    def testFilter(self):
        q = Table(data_id=1)
        q = Filter(q, 0, "==", 3)
        rlt = q.eval(inputs)

        # currently casted to float
        annotated = [{"a": {"value": 3, "trace": {"operator": "filter", "argument": [(3, 0, 0)]}}},
                     {"b": {"value": 4.0, "trace": {"operator": "filter", "argument": [(4.0, 1, 0)]}}},
                     {"c": {"value": float('nan'),
                            "trace": {"operator": "filter", "argument": [(float('nan'), 2, 0)]}}}]

        print("---Filter---")
        print(rlt.to_dict())
        print()
        # self.assertEqual(rlt.to_dict(), annotated)

    def testUnite(self):
        t = Table(data_id=1)
        q = Unite(t, 0, 1)
        rlt = q.eval(inputs)
        print("---Unite---")
        print(rlt.to_dict())
        print()

    def testCumSum(self):
        q = Table(data_id=1)
        q = CumSum(q, 0)
        rlt = q.eval(inputs)
        print("---CunSum---")
        print(rlt.to_dict())


if __name__ == '__main__':
    unittest.main()