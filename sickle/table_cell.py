"""
this class represents a cell stored in the data frame with its trace
"""
import copy

# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

class TableCell(object):
    def __init__(self, value, exp, attribute):
        self.value = value
        self.exp = exp
        self.attribute = attribute

    def to_dict(self):
        return {
            "value": self.value,
            "exp": self.exp.to_dict(),
            "attribute": self.attribute
        }

    def matches(self, other):
        # firstly, check this argument is a subset of other's argument
        # we assume that if argument is not None then operator should not be None
        if self.exp == HOLE:
            return True
        if self.exp is not None and other.exp is not None:
            if semantically_equiv(other.exp, self.exp):
                return True
        elif self.value is not None and self.value == other.value:
            return True
        return False

    def get_value(self):
        return self.value

    def get_attribute(self):
        return self.attribute

    def get_exp(self):
        return copy.copy(self.exp)

    def to_stmt(self):
        #if not isinstance(self.exp, ExpNode):
        #    return f"<{self.value}, {None}>"
        return f"<{self.value}, {self.exp}>"

    def get_flat_args(self):
        res = []
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode) or isinstance(e, ArgOr):
                    res += e.to_flat_list()
                else:
                    res += [e]
        else:
            # print(self.exp)
            res = self.exp.to_flat_list()
        return set(res)


EXP_OPS = ["sum", "average", "cumsum", "or"]


def semantically_equiv(exp1, exp2):
    # check for containment
    # exp2 is our target expression and exp1 is the output expression
    # looser check if the target cell contain some unsubstantiated parts
    if exp1 == HOLE:
        return True
    if isinstance(exp1, ArgOr) and isinstance(exp2, tuple):
        return exp1.contains(exp2)
    elif isinstance(exp2, ArgOr) and isinstance(exp1, tuple):
        return exp2.contains(exp1)
    elif isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
        if exp1.op != exp2.op:
            return False
        if HOLE in exp1.children:
            return True
        if len(exp1.children) < len(exp2.children):
            return False
        if exp2.children is []:
            return True
        for child2 in exp2.children:
            # search through the children of exp1 and try to find some ExpNode
            # that has children that contains the child of exp2
            # TODO: handle duplicate cases
            exist = False
            for child1 in exp1.children:
                if semantically_equiv(child1, child2):
                    exist = True
            if not exist:
                return False
        return True
    elif isinstance(exp1, list) and isinstance(exp2, list):
        for c2 in exp2:
            exist = False
            for c1 in exp1:
                if semantically_equiv(c1, c2):
                    exist = True
            if not exist:
                return False
        return True
    elif isinstance(exp1, list) and isinstance(exp2, ExpNode):
        return semantically_equiv(exp1, [exp2])
    elif isinstance(exp1, ExpNode) and isinstance(exp2, list):
        return semantically_equiv([exp1], exp2)
    else:
        # we cannot do straight comparison if one of them is coord and the other is expnode
        #print([exp1,exp2])
        return exp1 == exp2


class ExpNode(object):
    def __init__(self, op, children):
        self.op = op
        self.children = children  # children can be a list of ExpNode and CellCoordinates

    def __eq__(self, other):
        def exact_equiv(exp1, exp2):
            if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
                return (exp1.op == exp2.op and len(exp1.children) == len(exp2.children)
                        and all([exact_equiv(exp1.children[i], exp2.children[i]) for i in range(len(exp1.children))]))
            else:
                # then it's a leaf note, both exp are coordinates
                return exp1 == exp2
        if not isinstance(other, ExpNode):
            return False
        return exact_equiv(self, other)

    def to_dict(self):
        return {
            "op": self.op,
            "children": [c.to_dict() if isinstance(c, ExpNode) else c for c in self.children]
        }

    def get_op(self):
        return self.op

    def get_children(self):
        return copy.copy(self.children)

    def __repr__(self):
        return str((self.op, self.children))

    def to_flat_list(self):
        # decompose to get a list of coordinates
        def add_leaves(children, out):
            for e in children:
                if isinstance(e, ExpNode):
                    add_leaves(e.children, out)
                elif isinstance(e, ArgOr):
                    out += e.to_flat_list()
                else:
                    out += [e]
        res = self.children
        output = []
        add_leaves(res, output)
        return output


def dict_to_exp(source):
    new_children = []
    for k in source["children"]:
        if isinstance(k, dict):
            new_children.append(dict_to_exp(k))
        else:
            new_children.append(k)
    return ExpNode(source["op"], new_children)



"""
User demonstrate:
C   D
3   f1(sum(1, 3), sum(2, 3))
"""

if __name__ == '__main__':
    # f1(sum(1, 3), sum(2, 3))
    cell = TableCell(
            value=9, 
            exp=ExpNode(
                    op=lambda x,y: x + y, 
                    children=[
                        ExpNode(op="sum", children=[(0, 1, 1), (0, 1, 2)]),
                        ExpNode(op="sum", children=[(0, 2, 1), (0, 2, 2)])
                    ]),
            attribute="a")
    print(cell.to_dict())


class ArgOr:
    """this class represent some arguments that are alternatives to each other
        used for comparing traces"""

    def __init__(self, arguments):
        self.arguments = arguments  # a list of (note, coordinate_x, coordinate_y)

    def __eq__(self, other):
        if not isinstance(other, ArgOr):
            return False
        return self.arguments == other.arguments

    def __repr__(self):
        return "ArgOr" + str(self.arguments)

    def contains(self, exp):
        for val in self.arguments:
            if semantically_equiv(val, exp):
                return True
        return False

    def to_stmt(self):
        return "ArgOr" + str(self.arguments)

    def to_flat_list(self):
        return self.arguments.copy()


