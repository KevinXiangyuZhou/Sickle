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


def semantically_equiv(exp1, exp2):
    # exp2 is our target expression and exp1 is the output expression
    # looser check if the target cell contain some unknown parts
    if exp1 == HOLE:
        return True
    if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
        if exp1.op != HOLE and exp2.op != HOLE and exp1.op != exp2.op:
            sum_ops = ["cumsum", "sum", "lambda x, y: x + y"]
            if not (exp1.op in sum_ops and exp2.op in sum_ops):  # add ambiguity to operators
                return False
        if HOLE == exp1.children:
            return True
        """
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
        """
        return semantically_equiv(exp1.children, exp2.children)
    elif isinstance(exp1, list) and isinstance(exp2, list):
        used = []
        if UNKNOWN in exp2:  # we should check for containment
            for c2 in exp2:
                if c2 == UNKNOWN:  # skip the indicator
                    continue
                exist = False
                for i in range(len(exp1)):
                    if semantically_equiv(exp1[i], c2) and i not in used:
                        exist = True
                        used.append(i)
                if not exist:
                    return False
            return True
        else:  # the trace in user example is exact
            if len(exp1) != len(exp2):
                return False
            for c2 in exp2:
                exist = False
                for i in range(len(exp1)):
                    if semantically_equiv(exp1[i], c2) and i not in used:
                        exist = True
                        used.append(i)
                if not exist:
                    return False
            return len(used) == len(exp1)  # everything in exp1 is been used

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
        self.children = children  # children can be a set of ExpNode or CellCoordinates

    def __hash__(self):
        return hash(str(self.to_dict()))

    def __repr__(self):
        return str((self.op, self.children))

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
        new_op = self.op
        new_exp = []
        children_count = 0
        for e in self.children:
            p_miss_cell = random.randrange(4)
            p_miss_arg = random.randrange(2)
            # p_miss_operator = random.randrange(4)
            # if isinstance(e, ExpNode) and p_miss_arg == 0:  # 1/10
            if isinstance(e, ExpNode):
                new_exp += [e.randomize()]
            elif children_count >= 1 and p_miss_cell == 1:  # 1/10
                if UNKNOWN not in new_exp:
                    new_exp += [UNKNOWN]
            else:
                new_exp += [e]
                children_count += 1

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
        return hash(str(self.arguments))

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


