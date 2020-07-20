from table import *
import unittest


class TableTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TableTest, self).__init__(*args, **kwargs)
        tuples1 = set()
        tuples1.add(Table_cell(3, set([1, 2]), 'sum'))
        tuples1.add(Table_cell(1, set([1]), 'sum'))
        tuples1.add(Table_cell(5, set([2, 3]), 'sum'))
        tuples1.add(Table_cell(3, set([3, 3]), 'avg'))
        tuples1.add(Table_cell(3, set([2, 4]), 'avg'))
        self.a = Annotated_table(tuples1)

    def test_check_function_general(self):
        tuples2 = set()
        tuples2.add(Table_cell(3, set([2]), 'sum'))
        b = Annotated_table(tuples2)
        self.assertTrue(check_function(self.a, b))

    def test2(self):
        tuples3 = set()
        tuples3.add(Table_cell(5, set([2]), 'sum'))
        tuples3.add(Table_cell(5, set([3]), 'sum'))
        c = Annotated_table(tuples3)
        self.assertTrue(check_function(self.a, c))

    def test3(self):
        tuples4 = set()
        tuples4.add(Table_cell(1, set([]), 'sum'))
        tuples4.add(Table_cell(5, set([]), 'sum'))
        d = Annotated_table(tuples4)
        self.assertTrue(check_function(self.a, d))

    def test4(self):
        tuples5 = set()
        tuples5.add(Table_cell(3, set([1, 2, 4]), 'avg'))
        e = Annotated_table(tuples5)
        print(check_function(self.a,e))
        self.assertFalse(check_function(self.a,e))


if __name__ == '__main__':
    unittest.main()