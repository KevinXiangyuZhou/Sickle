from table import *
import unittest

class Table_test(unittest.TestCase):

    def test_check_function(self):
        tuples1 = set()
        tuples1.add(Tuple([1, 2], set([1, 2]), 'sum'))
        tuples1.add(Tuple([1, 2], set([1, 2]), 'sum'))
        tuples1.add(Tuple([2, 2, 3], set([2, 3]), 'sum'))
        tuples1.add(Tuple([1, 2, 3], set([3, 3]), 'sum'))
        tuples1.add(Tuple([2, 2, 3], set([2, 3]), 'avg'))
        a = Table(None, tuples1)

        tuples2 = set()
        tuples2.add(Tuple([2, 2, 3], set([2, 3]), 'sum'))

        tuples3 = set()
        tuples3.add(Tuple([2, 2, 3], set([2]), 'sum'))
        tuples3.add(Tuple([1, 2], set([1]), 'sum'))

        tuples4 = set()
        tuples4.add(Tuple([2, 2, 3], set([2, 3, 4]), 'sum'))
        tuples4.add(Tuple([1, 2], set([1]), 'sum'))

        tuples5 = set()
        tuples5.add(Tuple([2, 2, 3], set([2, 3]), 'avg'))

        b = Table(None, tuples2)
        c = Table(None, tuples3)
        d = Table(None, tuples4)
        e = Table(None, tuples5)

        self.assertTrue(a.check_function(b))
        self.assertTrue(a.check_function(c))
        self.assertFalse(a.check_function(d))
        self.assertFalse(a.check_function(e))


if __name__ == '__main__':
    unittest.main()