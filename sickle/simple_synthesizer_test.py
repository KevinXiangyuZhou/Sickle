from table import *
import unittest
from table_ast import *
from synthesizer import *

# small parameter config for tests
test_config = {
                "operators": ["group_sum"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["mean"],
                "mutate_function": ["*0.1", "mean"]
            }

test_config_22 = {
                "operators": ["mutate_2"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum", "max", "mean"],
                "arithmetic_op": ["+", "-", "*", "/"],
                "mutate_function": ["lambda x: x - (x * 0.1)", "lambda x,y,z: x * y * z"]
            }

test_config_2 = {
                "operators": ["mutate_arithmetic", "mutate"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum", "max", "mean"],
                "arithmetic_op": ["+", "-", "*", "/"],
                "mutate_function": ["*0.1", "*0.9"]
            }

test_config_3 = {
                "operators": ["mutate_arithmetic"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum", "max", "mean"],
                "arithmetic_op": ["+", "-", "*", "/"]
            }

test_config_5 = {
                "operators": ["mutate"],
                "filer_op": ["=="],
                "constants": [3000],
                "aggr_func": ["sum", "max", "mean"],
                "mutate_function": ["sum", "*2", "+ 1 * 12"]
            }

#pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class SimpleSynthesizerTest(unittest.TestCase):
    #@unittest.skip
    def test_1(self):
        # sample test case including neither arithmetic or aggregation
        # source link: https://stackoverflow.com/questions/62082860/
        # cumsum-on-one-column-conditional-on-number-of-occurence-of-values-from-another-c
        # expected program:
        inputs = [[
            {"empno": 7369, "depno": 20, "sal": 800},
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
            {"empno": 7934, "depno": 10, "sal": 1300}
        ]]

        output = [
            {"depno": 10, "mean_sal": 2916.67},
            {"depno": 20, "mean_sal": 2175.00},
            {"depno": 30, "mean_sal": 1566.67}
        ]

        #group sum
        annotated_output = AnnotatedTable(
            [{"value": None, "argument": [(0, 1, 0)], "operator": ["group_mean"], "attribute": "depno"},
             {"value": None, "argument": [(0, 1, 2)], "operator": ["group_mean"], "attribute": "depno"},
             {"value": None, "argument": [(0, 2, 0)], "operator": ["group_mean"], "attribute": "mean_sal"},
             {"value": None, "argument": [(0, 2, 1)], "operator": ["group_mean"], "attribute": "mean_sal"}]
        )
        # annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config).enumerative_synthesis(inputs, annotated_output, 3)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe().to_csv())

    @unittest.skip
    def test_2(self):
        # sample test case including only arithmetic
        # source link: https://www.w3resource.com/sql/update-statement/update-columns-using-arithmetical-expression.php
        # expected program:
        # UPDATE customer1
        # SET outstanding_amt=outstanding_amt-(outstanding_amt*.10)
        # WHERE cust_country='India' AND grade=1;
        inputs = [[
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 4000},
            {"cust_country": "USA", "grade": 2, "outstanding_amt": 6000},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 6000},
            {"cust_country": "India", "grade": 2, "outstanding_amt": 8000},
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 6000},
            {"cust_country": "UK", "grade": 1, "outstanding_amt": 11000},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 3000}
        ]]

        output = [
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 3600},
            {"cust_country": "USA", "grade": 2, "outstanding_amt": 5400},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 5400},
            {"cust_country": "India", "grade": 2, "outstanding_amt": 7200},
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 5400},
            {"cust_country": "UK", "grade": 1, "outstanding_amt": 9900},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 2700}
        ]
        # 'outstanding_amt'-('outstanding_amt'*.10)
        annotated_output = load_from_dict(output)
        print(annotated_output.to_dict())
        candidates = Synthesizer(test_config_22).enumerative_synthesis(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_3(self):
        # sample test case including only arithmetic
        # source link: https://stackoverflow.com/questions/64236076/
        # product-of-multiple-selected-columns-in-a-data-frame-in-r
        # expected program:
        inputs = [[
            {"X": 1, "Y": 2, "Z": 3},
            {"X": 2, "Y": 3, "Z": 4},
            {"X": 3, "Y": 4, "Z": 5}
        ]]

        output = [
            {"X": 1, "Y": 2, "Z": 3, "product": 6},
            {"X": 2, "Y": 3, "Z": 4, "product": 24},
            {"X": 3, "Y": 4, "Z": 5, "product": 60}
        ]
        annotated_output = load_from_dict(output)
        candidates = Synthesizer(test_config_22).enumerative_synthesis(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_4(self):
        # sample test case including neither arithmetic or aggregation
        # source link: https://stackoverflow.com/questions/31549958/
        # using-dplyrsummarize-function-for-multi-step-arithmetical-process?r=SearchResults
        # expected program:
        inputs = [[
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 525, "distShot": 367.6},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 157.4, "distShot": 130.8},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 27.5, "distShot": 27.4},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 1.2, "distShot": 1.2},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 222, "distShot": 216.6},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 6.8, "distShot": 6.6},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 0.3, "distShot": 0.3},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 378, "distShot": 244.4},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 135.9, "distShot": 141.6},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 6.7, "distShot": 6.0},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 0.1, "distShot": 0.1},
        ]]

        # intermediate output for debugging
        output_e1 = [
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 525, "distShot": 367.6, "sum_distShot": 527},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 157.4, "distShot": 130.8, "sum_distShot": 527},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 27.5, "distShot": 27.4, "sum_distShot": 527},
            {"player": "E.Els", "rd": 1, "hole": 1, "distToPin": 1.2, "distShot": 1.2, "sum_distShot": 527},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 222, "distShot": 216.6, "sum_distShot": 223.5},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 6.8, "distShot": 6.6, "sum_distShot": 223.5},
            {"player": "E.Els", "rd": 1, "hole": 2, "distToPin": 0.3, "distShot": 0.3, "sum_distShot": 223.5},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 378, "distShot": 244.4, "sum_distShot": 392.1},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 135.9, "distShot": 141.6, "sum_distShot": 392.1},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 6.7, "distShot": 6.0, "sum_distShot": 392.1},
            {"player": "E.Els", "rd": 2, "hole": 1, "distToPin": 0.1, "distShot": 0.1, "sum_distShot": 392.1},
        ]

        output_e2 = [
            {"player": "E.Els", "rd": 1, "hole": 1, "sum_distToPin": 525, "sum_distShot": 527},
            {"player": "E.Els", "rd": 1, "hole": 2, "sum_distToPin": 222, "sum_distShot": 223.5},
            {"player": "E.Els", "rd": 2, "hole": 1, "sum_distToPin": 378, "sum_distShot": 392.1},
        ]
        # by synthesizing input->output_e2, we get
        # p':
        # t0 <- table_ref(0);
        # t1 <- group_mutate(t0, (1, 2), 3, max);
        # t2 <- group_sum(t1, (0, 1, 2, 5), 4, sum)

        # intermediate output for debugging
        output_e3a = [
            {"player": "E.Els", "rd": 1, "hole": 1, "sum_distToPin": 525, "sum_distShot": 527, "difference": 2},
            {"player": "E.Els", "rd": 1, "hole": 2, "sum_distToPin": 222, "sum_distShot": 223.5, "difference": 1.5},
            {"player": "E.Els", "rd": 2, "hole": 1, "sum_distToPin": 378, "sum_distShot": 392.1, "difference": 14.1}
        ]
        output_e3 = [
            {"player": "E.Els", "rd": 1, "hole": 1, "sum_distToPin": 525, "sum_distShot": 527, "efficiency": 263.5},
            {"player": "E.Els", "rd": 1, "hole": 2, "sum_distToPin": 222, "sum_distShot": 223.5, "efficiency": 149},
            {"player": "E.Els", "rd": 2, "hole": 1, "sum_distToPin": 378, "sum_distShot": 392.1, "efficiency": 27.81}
        ]

        output = [
            {"player": "E.Els", "rd": 1, "efficiency": 206.25},
            {"player": "E.Els", "rd": 2, "efficiency": 27.81}
        ]

        # function: group_mean(group_max(disttopin) / group_sum(dist) - group_max(disttopin))
        annotated_output = load_from_dict(output_e2)
        candidates_1 = Synthesizer(test_config).enumerative_synthesis(inputs, annotated_output, 2)
        print()
        for p in candidates_1:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

        annotated_output = load_from_dict(output_e3)
        # print(expected_p.eval(inputs).to_plain_dict())
        candidates_2 = Synthesizer(test_config_3).enumerative_synthesis([output_e2],
                                                                     annotated_output, 2)
        print()
        for p in candidates_2:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval([output_e2]).to_dataframe())

        annotated_output = load_from_dict(output)
        # print(expected_p.eval(inputs).to_plain_dict())
        candidates_3 = Synthesizer(test_config).enumerative_synthesis([output_e3],
                                                                     annotated_output, 1)
        print()
        for p in candidates_3:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval([output_e3]).to_dataframe())

    @unittest.skip
    def test_5(self):
        # sample test case including neither arithmetic or aggregation
        # source link: https://stackoverflow.com/questions/60517090/
        # is-there-a-way-to-divide-answers-of-group-by-summary-statistics-in-r
        # expected program:
        inputs = [[
            {"state": "Alabama", "count": 1667},
            {"state": "Alaska", "count": 507},
            {"state": "Alabama", "count": 930},
            {"state": "Arizona", "count": 1352},
            {"state": "California", "count": 1817},
            {"state": "Colorado", "count": 2302},
            {"state": "Connecticut", "count": 1488}
        ]]

        output_e1 = [
            {"state": "Alabama", "count": 1667, "sum": 10063},
            {"state": "Alaska", "count": 507, "sum": 10063},
            {"state": "Alabama", "count": 930, "sum": 10063},
            {"state": "Arizona", "count": 1352, "sum": 10063},
            {"state": "California", "count": 1817, "sum": 10063},
            {"state": "Colorado", "count": 2302, "sum": 10063},
            {"state": "Connecticut", "count": 1488, "sum": 10063}
        ]

        output_e2 = [
            {"state": "Alabama", "count": 1667, "percentage": 0.5},
            {"state": "Alaska", "count": 507, "percentage": 0.5},
            {"state": "Alabama", "count": 930, "percentage": 0.5},
            {"state": "Arizona", "count": 1352, "percentage": 0.5},
            {"state": "California", "count": 1817, "percentage": 0.5},
            {"state": "Colorado", "count": 2302, "percentage": 0.5},
            {"state": "Connecticut", "count": 1488, "percentage": 0.5}
        ]

        # (count / sum(count)) * 100
        annotated_output = load_from_dict(output_e1)
        candidates = Synthesizer(test_config_5).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())


    @unittest.skip
    def test_6(self):
        # a trace version of ex2 (user provide expression and pointers)
        # sample test case including only arithmetic
        # source link: https://www.w3resource.com/sql/update-statement/update-columns-using-arithmetical-expression.php
        # expected program:
        # UPDATE customer1
        # SET outstanding_amt=outstanding_amt-(outstanding_amt*.10)
        # WHERE cust_country='India' AND grade=1;
        inputs = [[
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 4000},
            {"cust_country": "USA", "grade": 2, "outstanding_amt": 6000},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 6000},
            {"cust_country": "India", "grade": 2, "outstanding_amt": 8000},
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 6000},
            {"cust_country": "UK", "grade": 1, "outstanding_amt": 11000},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 3000}
        ]]

        output = [
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 3600},
            {"cust_country": "USA", "grade": 2, "outstanding_amt": 5400},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 5400},
            {"cust_country": "India", "grade": 2, "outstanding_amt": 7200},
            {"cust_country": "UK", "grade": 2, "outstanding_amt": 5400},
            {"cust_country": "UK", "grade": 1, "outstanding_amt": 9900},
            {"cust_country": "USA", "grade": 3, "outstanding_amt": 2700}
        ]
        # 'outstanding_amt'-('outstanding_amt'*.10)
        # get values through expression and trace
        annotated_output = AnnotatedTable(
            [{"value": None, "argument": [(2, 0)], "operator": "lambda x: x - (x * 0.1)", "attribute": ""},
             {"value": None, "argument": [(2, 1)], "operator": "lambda x: x - (x * 0.1)", "attribute": ""},
             {"value": None, "argument": [(2, 5)], "operator": "lambda x: x - (x * 0.1)", "attribute": ""},
             {"value": None, "argument": [(2, 6)], "operator": "lambda x: x - (x * 0.1)", "attribute": ""}]
        )
        print(load_from_dict(output).to_dict())
        candidates = Synthesizer(test_config_22).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe())

    @unittest.skip
    def test_7(self):
        # a trace version of ex2 (user provide expression and pointers)
        # sample test case including only arithmetic
        # source link: https://stackoverflow.com/questions/64236076/
        # product-of-multiple-selected-columns-in-a-data-frame-in-r
        # expected program:
        inputs = [[
            {"X": 1, "Y": 2, "Z": 3},
            {"X": 2, "Y": 3, "Z": 4},
            {"X": 3, "Y": 4, "Z": 5}
        ]]

        output = [
            {"X": 1, "Y": 2, "Z": 3, "product": 6},
            {"X": 2, "Y": 3, "Z": 4, "product": 24},
            {"X": 3, "Y": 4, "Z": 5, "product": 60}
        ]
        print(load_from_dict(output).to_dict())
        annotated_output = AnnotatedTable(
            [{"value": None, "argument": [(0, 0), (1, 0), (2, 0)], "operator": "lambda x,y,z: x * y * z", "attribute": ""},
             {"value": None, "argument": [(0, 1), (1, 1), (2, 1)], "operator": "lambda x,y,z: x * y * z", "attribute": ""},
             {"value": None, "argument": [(0, 2), (1, 2), (2, 2)], "operator": "lambda x,y,z: x * y * z", "attribute": ""}]
        )
        candidates = Synthesizer(test_config_22).enumerative_search(inputs, annotated_output, 2)
        print()
        for p in candidates:
            # print(alignment_result)
            print(p.stmt_string())
            print(p.eval(inputs).to_dataframe().to_csv())

if __name__ == '__main__':
    unittest.main()
