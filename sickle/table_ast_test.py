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

test_data_biketrip = [
                        {"trip_id": 944732, "date": "2015-09-24"},
                        {"trip_id": 984595, "date": "2015-09-24"},
                        {"trip_id": 984596, "date": "2015-09-24"},
                        {"trip_id": 1129385, "date": "2015-09-24"},
                        {"trip_id": 1030383, "date": "2015-09-30"},
                        {"trip_id": 969490, "date": "2015-09-30"},
                        {"trip_id": 947105, "date": "2015-09-30"},
                        {"trip_id": 1011650, "date": "2015-11-16"}
                    ]
intermediate_test_data_biketrip = [{"date": "2015-09-24", "count_date": 4},
                              {"date": "2015-09-30", "count_date": 3},
                              {"date": "2015-11-16", "count_date": 1}]

test_data_salepercentage = [{"id": 1, "total": 24.78},
                            {"id": 2, "total": 28.54},
                            {"id": 3, "total": 48.69},
                            {"id": 4, "total": -16.39},
                            {"id": 5, "total": 29.92},
                            {"id": 6, "total": 12.5},
                            {"id": 7, "total": 10.2},
                            {"id": 8, "total": 5.22}]

test_data_int_salepercent = [{"id": 1, "total": 24.78, "total.sum()": 143.46},
                            {"id": 2, "total": 28.54, "total.sum()": 143.46},
                            {"id": 3, "total": 48.69, "total.sum()": 143.46},
                            {"id": 4, "total": -16.39, "total.sum()": 143.46},
                            {"id": 5, "total": 29.92, "total.sum()": 143.46},
                            {"id": 6, "total": 12.5, "total.sum()": 143.46},
                            {"id": 7, "total": 10.2, "total.sum()": 143.46},
                            {"id": 8, "total": 5.22, "total.sum()": 143.46}]

test_data_emp = [{"empno": 7369, "depno": 20, "sal": 800},
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
                 {"empno": 7934, "depno": 10, "sal": 1300}]


test_data_1 = [{"a": 3, "b": 4},
               {"a": 5},
               {"a": 6, "b": 7, "c": 8}]

test_data_2 = [{"a": 3, "b": 4, "c": 3},
               {"a": 3, "b": 5, "c": 6},
               {"a": 6, "b": 7, "c": 8}]

inputs = {0: pd.DataFrame.from_dict(test_data_0),
          1: pd.DataFrame.from_dict(test_data_1),
          2: pd.DataFrame.from_dict(test_data_2),
          3: pd.DataFrame.from_dict(test_data_biketrip),
          4: pd.DataFrame.from_dict(intermediate_test_data_biketrip),
          5: pd.DataFrame.from_dict(test_data_salepercentage),
          6: pd.DataFrame.from_dict(test_data_int_salepercent),
          7: pd.DataFrame.from_dict(test_data_emp)}

a = AnnotatedTable([{"value": 3, "argument": [(0, 0, 0)], "operator": [], "attribute": "a"},
                    {"value": 4, "argument": [(0, 1, 0)], "operator": [], "attribute": "b"}])

b = AnnotatedTable([{"value": 5, "argument": [(2, 1, 1)], "operator": [], "attribute": "b"},
                    {"value": 4, "argument": [(2, 1, 0)], "operator": [], "attribute": "b"},
                    {"value": 6, "argument": [(2, 2, 1)], "operator": [], "attribute": "c"},
                    {"value": 8, "argument": [(2, 2, 2)], "operator": [], "attribute": "c"}])

c = AnnotatedTable([{"value": 3, "argument": [(2, 0, 0)], "operator": [], "attribute": "a"},
                    {"value": 3, "argument": [(2, 2, 0)], "operator": [], "attribute": "c"},
                    {"value": 6, "argument": [(2, 0, 2)], "operator": [], "attribute": "a"},
                    {"value": 8, "argument": [(2, 2, 2)], "operator": [], "attribute": "c"}])


class AstTest(unittest.TestCase):
    def test_join(self):
        q = Table(data_id=2)
        q = Join(q, q)
        rlt = q.eval(inputs)
        print("---Join---")
        print(rlt.extract_values())
        print(rlt.to_dataframe().to_csv())
        print()

    def test_join_by(self):
        q = Table(data_id=2)
        q = JoinBy(q, q, 1, 1)
        rlt = q.eval(inputs)
        print("---JoinBy---")
        print(rlt.extract_values())
        print(rlt.to_dataframe().to_csv())
        print()

    def test_select(self):
        q = Table(data_id=1)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0])
        rlt = q.eval(inputs)
        print("---Select---")
        print(rlt.to_dataframe().to_csv())
        print()
        # self.assertEqual(rlt.to_dict(), annotated)

    def test_filter(self):
        q = Table(data_id=1)
        q = Filter(q, 0, "==", 5)
        rlt = q.eval(inputs)
        print("---Filter---")
        print(rlt.to_dataframe().to_csv())
        print()
        # self.assertEqual(rlt.to_dict(), annotated)

    def test_unite(self):
        t = Table(data_id=1)
        q = Unite(t, 0, 1)
        rlt = q.eval(inputs)
        print("---Unite---")
        print(rlt.to_dataframe().to_csv())
        print()

    def test_cumsum(self):
        q = Table(data_id=1)
        q = CumSum(q, 0)
        rlt = q.eval(inputs)
        print("---CumSum---")
        print(rlt.to_dataframe().to_csv())
        print()

    def test_zgroupsummary(self):
        q = Table(data_id=2)
        q = GroupSummary(q, [0], 1, "sum")
        rlt = q.eval(inputs)
        print("---GroupSummary---")
        print(rlt.to_dataframe().to_csv())
        print()

    def test_groupmutate(self):
        q = Table(data_id=2)
        q = GroupMutate(q, [0], "cumsum", 2)
        rlt = q.eval(inputs)
        print("---GroupMutate---")
        print(rlt.to_dataframe().to_csv())
        print()

    @unittest.skip
    def test_mutate(self):
        q = Table(data_id=2)
        q = Mutate(q, 1, "sum")
        rlt = q.eval(inputs)
        print("---Mutate---")
        print(rlt.to_dataframe().to_csv())
        print(rlt.extract_values())
        print()

    @unittest.skip
    def test_mutate_arith(self):
        q = Table(data_id=2)
        q = MutateArithmetic(q, 0, "*", 1)
        rlt = q.eval(inputs)
        print("---Mutate_Arithmetic---")
        print(rlt.to_dataframe().to_csv())
        print(rlt.to_plain_dict())
        print(rlt.extract_values())
        print()

    #@unittest.skip
    def test_mutate_arith2(self):
        q = Table(data_id=2)
        q = Mutate_2(q, "lambda x, y: x * y - 0.2 * x", (1, 2))
        rlt = q.eval(inputs)
        print("---Mutate_Arithmetic---")
        print(rlt.to_dataframe().to_csv())
        print(rlt.to_plain_dict())
        print(rlt.extract_values())
        print()

    #@unittest.skip
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

    #@unittest.skip
    def test_checker_function2(self):
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0, 1, 2])
        rlt = q.eval(inputs)
        print("---CheckerFunction2---")
        print(checker_function(rlt, b))
        print()

    #@unittest.skip
    def test_checker_function3(self):
        q = Table(data_id=2)
        # print(q.eval(inputs).to_dict())
        q = Select(q, [0, 1, 2])
        rlt = q.eval(inputs)
        print("---CheckerFunction3---")
        print(checker_function(rlt, c))
        x = a.get_cell(0,0)
        y = a.get_cell(1,0)
        print()

    #@unittest.skip
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
    def test_zbike_trips(self):
        q = Table(data_id=3)
        q = GroupSummary(q, [1], 0, "count")
        rlt = q.eval(inputs)
        print("---BikeTrip---")
        print(rlt.extract_values())
        print()
        #q = Table(data_id=4)
        q = CumSum(q, 1)
        rlt = q.eval(inputs)
        print(rlt.extract_values())

    @unittest.skip
    def test_zsale_percentage(self):
        q = Table(data_id=5)
        q = Mutate(q, 1, ".sum()", True)
        rlt = q.eval(inputs)
        print("---SalePercent---")
        print(rlt.extract_values())
        print()
        q = Table(data_id=6)
        q = Mutate(q, 1, "/ x[\'total.sum()\']", False)  # mutate function is currently limited
        rlt = q.eval(inputs)
        print(rlt.extract_values())
        print()

    @unittest.skip
    def test_zempsal1(self):
        q = Table(data_id=7)
        q = GroupSummary(q, [1], 2, "mean")
        rlt = q.eval(inputs)
        print("---EmpSal1---")
        print(rlt.to_dataframe().to_csv())
        print()

    @unittest.skip
    def test_zempsal2(self):
        q = Table(data_id=7)
        q = GroupMutate(q, [1], 2, "mean")
        rlt = q.eval(inputs)
        print("---EmpSal2---")
        print(rlt.to_dataframe().to_csv())
        print()



if __name__ == '__main__':
    unittest.main()