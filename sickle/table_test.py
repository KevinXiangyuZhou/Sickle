from table import *
import unittest
import pandas as pd



a = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"},
     {"value": 5, "argument": [(5, 0, 1)], "operator": "sum", "attribute": "a"},
     {"value": 6, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])

b = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"}])
c = AnnotatedTable([{"value": 3, "argument": [(5, 0, 1)], "operator": "avg", "attribute": "a"}])
d = AnnotatedTable([{"value": 6, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])
e = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "sum", "attribute": "a"}])
f = AnnotatedTable([{"value": 3, "argument": [(1, 0, 0), (5, 0, 1)], "operator": None, "attribute": "a"},
                   {"value": 6, "argument": None, "operator": "sum", "attribute": "a"}])
g = AnnotatedTable([{"value": None, "argument": [(1, 0, 0), (5, 0, 1)], "operator": "avg", "attribute": "a"}])

class TableTest(unittest.TestCase):

    def testCheckFunctionSubTable(self):
        self.assertTrue(check_function(a, b))

    def testCheckFunctionSubTableSubArgument(self):
        self.assertTrue(check_function(a, c))

    def testCheckFunctionReflexive(self):
        self.assertTrue(check_function(a, a))
        self.assertTrue(check_function(b, b))

    def testCheckFunctionInconsistentOperator(self):
        self.assertFalse(check_function(b, e))

    def testCheckFunctionInconsistentValue(self):
        self.assertFalse(check_function(d, e))

    def testCheckFunctionNoneArgument(self):
        self.assertTrue(check_function(a,f))
        self.assertFalse(check_function(a,g))

    def testExtractValue(self):
        df = pd.DataFrame([{"a": 3}, {"a": 5}, {"a": 6}])
        self.assertTrue(df.equals(a.extract_values()))

    def testAnnotatedTableToDict(self):
        expected_dict_list_b = [{"a": {"value": 3, "trace": {"operator": "avg", "argument": [(1, 0, 0), (5, 0, 1)]}}}]
        self.assertEqual(expected_dict_list_b, b.to_dict())


if __name__ == '__main__':
    unittest.main()