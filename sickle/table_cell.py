"""
this class represents a cell stored in the data frame with its trace
"""
import copy
import random
from configuration import config

# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

class TableCell(object):
    def __init__(self, value, exp):
        self.value = value
        self.exp = exp

    def to_dict(self):
        return {
            "value": self.value,
            "exp": self.exp.to_dict()
        }

    def matches(self, other):
        # firstly, check this argument is a subset of other's argument
        # we assume that if argument is not None then operator should not be None
        if self.exp is not None and other.exp is not None:
            if semantically_equiv(other.exp, self.exp):
                return True
        elif self.value is not None and self.value == other.value:
            return True
        return False

    def get_value(self):
        return self.value

    def get_exp(self):
        return copy.copy(self.exp)

    def to_stmt(self):
        # if not isinstance(self.exp, ExpNode):
        #    return f"<{self.value}, {None}>"
        return f"{self.exp}"

    def get_flat_args(self):
        res = set()
        #print("!!!")
        #print(self.exp)
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode) or isinstance(e, ArgOr):
                    res.update(e.to_flat_list())
                    #print(res)
                else:
                    res.add(e)
        else:
            # print(self.exp)
            res.update(self.exp.to_flat_list())
        return res

    def get_flat_ops(self):
        res = set()
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode):
                    res |= e.get_flat_ops()
        elif isinstance(self.exp, ExpNode):
            res |= self.exp.get_flat_ops()
        return res

    def randomize(self):
        if isinstance(self.exp, list):
            rand_exp = []
            for e in self.exp:
                if isinstance(e, ExpNode):
                    rand_exp += [e.randomize()]
                else:
                    rand_exp += [e]
        else:
            rand_exp = self.exp.randomize()
        return TableCell(self.value, rand_exp)



EXP_OPS = ["sum", "average", "cumsum", "or"]


def semantically_equiv(exp1, exp2):
    # check for containment
    # exp2 is our target expression and exp1 is the output expression
    # looser check if the target cell contain some unknown parts
    # if exp1 == HOLE:
    #     return True
    if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
        if exp1.op != HOLE and exp2.op != HOLE and exp1.op != exp2.op:
            return False
        # if HOLE in exp1.children:
        #    return True
        # if exp2.children is []:
        #     return True
        used = []
        for child2 in exp2.children:
            # search through the children of exp1 and try to find some ExpNode
            # that has children that contains the child of exp2
            exist = False
            for i in range(len(exp1.children)):
                if semantically_equiv(exp1.children[i], child2) and i not in used:
                    exist = True
                    used.append(i)
                    break
            if not exist:
                return False
        return True
    elif isinstance(exp1, list) and isinstance(exp2, list):
        used = []
        for c2 in exp2:
            exist = False
            for i in range(len(exp1)):
                if semantically_equiv(exp1[i], c2) and i not in used:
                    exist = True
                    used.append(i)
            if not exist:
                return False
        return True
    elif isinstance(exp1, list) and isinstance(exp2, ExpNode):
        return semantically_equiv(exp1, [exp2])
    elif isinstance(exp1, ExpNode) and isinstance(exp2, list):
        return semantically_equiv([exp1], exp2)
    else:
        # print([exp1,exp2])
        # if isinstance(exp1, ExpNode) and isinstance(exp2,ArgOr):
        #    print(f"exp1: {exp1}     exp2: {exp2}")
        #    print(exp1 == exp2)
        if exp1 == HOLE or exp2 == HOLE:
            return True
        return exp1 == exp2


class ExpNode(object):
    def __init__(self, op, children):
        self.op = op
        self.children = children  # children can be a list of ExpNode and CellCoordinates

    def __eq__(self, other):
        def exact_equiv(exp1, exp2):
            if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
                return (exp1.op == exp2.op and len(exp1.children) == len(exp2.children)
                        and all([exact_equiv(exp1.children[i], exp2.children[i])
                                 for i in range(len(exp1.children))]))
            else:
                # then it's a leaf note, both exp are coordinates
                return exp1 == exp2
        if isinstance(other, ArgOr):
            for e in other.arguments:
                if self.__eq__(e):
                    return True
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
                    out.update(e.to_flat_list())
                else:
                    out.add(e)
        res = self.children
        output = set()
        add_leaves(res, output)
        return output

    def get_flat_ops(self):
        res = set()
        if isinstance(self.op, ArgOr):
            res.update(self.op.to_flat_list())
        elif isinstance(self.op, list):
            res.update(self.op)
        else:
            res.add(self.op)
        for e in self.children:
            if isinstance(e, ExpNode):
                res |= e.get_flat_ops()
        # print(res)
        return res

    def randomize(self):
        # with_operator = random.randrange(3)
        new_op = self.op
        new_exp = []
        # if with_operator == 0:
        #     new_op = HOLE
        for e in self.children:
            p_miss_cell = random.randrange(2)
            p_miss_arg = random.randrange(2)
            if isinstance(e, ExpNode) and p_miss_arg == 0:  # 1/2
                new_exp += [e.randomize()]
            if not isinstance(e, ExpNode) and p_miss_cell == 0:  # 1/2
                new_exp += [e]

        return ExpNode(new_op, new_exp)




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
                    ]))
    print(cell.to_dict())


class ArgOr:
    """this class represent some arguments that are alternatives to each other
        used for comparing traces"""

    def __init__(self, arguments):
        self.arguments = arguments  # a list of (note, coordinate_x, coordinate_y)

    def __hash__(self):
        total_hash = 0
        for e in self.arguments:
            total_hash += hash(e)
        return total_hash

    def __eq__(self, other):
        if not isinstance(other, ArgOr):
            return other in self.arguments
        return [1 for i in self.arguments for j in other.arguments if i == j] != []
        #return self.contains(other) or other.contains()

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
        res = set()
        for e in self.arguments:
            if isinstance(e, ExpNode) or isinstance(e, ArgOr):
                res.update(e.to_flat_list())
            else:
                res.add(e)
        return res


