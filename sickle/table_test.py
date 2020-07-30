from table import *
import unittest



a = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg', 'attribute': None},
     {'value': 5, 'argument': [[5, 0, 1]], 'operator': 'sum', 'attribute': None},
     {'value': 6, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum', 'attribute': None}])

b = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg', 'attribute': None}])
c = AnnotatedTable([{'value': 3, 'argument': [[5, 0, 1]], 'operator': 'avg', 'attribute': None}])
d = AnnotatedTable([{'value': 6, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum', 'attribute': None}])
e = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum', 'attribute': None}])
f = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': None, 'attribute': None},
                   {'value': 6, 'argument': None, 'operator': 'sum', 'attribute': None}])
g = AnnotatedTable([{'value': None, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg', 'attribute': None}])

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
        pass

if __name__ == '__main__':
    unittest.main()