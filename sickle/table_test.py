from table import *
import unittest


a = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg'},
     {'value': 5, 'argument': [[5, 0, 1]], 'operator': 'sum'},
     {'value': 6, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum'}])

b = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg'}])
c = AnnotatedTable([{'value': 3, 'argument': [[5, 0, 1]], 'operator': 'avg'}])
d = AnnotatedTable([{'value': 6, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum'}])
e = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'sum'}])
f = AnnotatedTable([{'value': 3, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': None},
                   {'value': 6, 'argument': None, 'operator': 'sum'}])
g = AnnotatedTable([{'value': None, 'argument': [[1, 0, 0], [5, 0, 1]], 'operator': 'avg'}])

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


if __name__ == '__main__':
    unittest.main()