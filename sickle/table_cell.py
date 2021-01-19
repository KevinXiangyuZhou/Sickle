
###TODO

"""
this class represents a cell stored in the data frame with its trace
"""
class TableCell(object):
    def __init__(self, value, exp):
        self.value = value
        self.exp = exp

    def to_dict(self):
        return {
            "type": "table_cell",
            "value": self.value,
            "exp": self.exp.to_dict()
        }

    # other fields are same as old Table cell


EXP_OPS = ["sum", "average", "cumsum", "or"]

class ExpNode(object):

    @staticmethod
    def exact_equiv(exp1, exp2):
        if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
            return (exp1.op == exp2.op and len(exp1.children) == len(exp1.children) 
                and all([exact_equal(exp1.children[i], exp2.children[i]) for i in range(len(exp1.children))]))
        else:
            # then it's a leaf note, both exp are coordinates
            return exp1 == exp2

    @staticmethod
    def semantically_equiv(exp1, exp2):
        if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
            # TODO
            pass
        else:
            # then it's a leaf note, both exp are coordinates
            return exp1 == exp2

    def __init__(self, op, children):
        self.op = op
        self.children = children # children can be a list of ExpNode, or a list of CellCoordinates

    def to_dict(self):
        return {
            "type": "table_cell_exp_node",
            "op": self.op,
            "children": [c.to_dict() if isinstance(c, ExpNode) else c for c in self.children]
        }


"""
User demonstrate:
C   D
3   f1(sum(1, 3), sum(2, 3))
"""

if __name__ == '__main__':
    #f1(sum(1, 3), sum(2, 3))
    cell = TableCell(
            value=9, 
            exp=ExpNode(
                    op=lambda x,y: x + y, 
                    children=[
                        ExpNode(op="sum", children=[(0, 1, 1), (0, 1, 2)]),
                        ExpNode(op="sum", children=[(0, 2, 1), (0, 2, 2)])
                    ]))
    print(cell.to_dict())

