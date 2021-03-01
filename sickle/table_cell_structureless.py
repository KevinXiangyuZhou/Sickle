# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"
"""
this class represents a cell stored in the data frame with its trace
"""
class SimpleCell:
    def __init__(self, value, argument, operator, attribute):
        self.value = value
        self.argument = argument  # a list of (note, coordinate_x, coordinate_y)
        self.operator = operator
        self.attribute = attribute

    def get_value(self):
        return self.value

    def matches(self, other):
        # looser check if the target cell contain some uninstantiated parts
        if other.operator == HOLE and other.argument == HOLE:
            return True
        elif other.operator == HOLE:
            return self.is_sublist(self.argument, other.argument)
        elif other.argument == HOLE:
            return self.is_sublist(self.operator, other.operator)
        # firstly, check this argument is a subset of other's argument
        # we assume that if argument is not None then operator should not be None
        if self.argument is not None and self.operator is not None:
            if self.is_sublist(self.argument, other.argument) \
                    and self.is_sublist(self.operator, other.operator):
                return True
        # if self.operator is not None and self.operator != other.operator:
        #    return False
        # next, if we do not have trace to do comparison
        # we check if we can find some values that map
        elif self.value is not None and self.value == other.value:
            return True
        return False

    """check if everything in lst1 is contained by lst2"""
    def is_sublist(self, lst1, lst2):
        if not lst1:
            return True
        for i in range(len(lst1)):
            # for each value in lst1
            # check if it is exist in lst2
            exist = False
            for j in range(len(lst2)):
                if isinstance(lst2[j], ArgOr):
                    if lst2[j].contains(lst1[i]):
                        exist = True
                elif lst1[i] == lst2[j]:
                    exist = True
            # we did not find this value in lst2
            # so return false
            if not exist:
                return False
        return True

    def to_dict(self):
        return {
            "value": self.value,
            "argument": self.argument,
            "operator": self.operator,
            "attribute": self.attribute
        }

    def get_attribute(self):
        return self.attribute

    def get_argument(self):
        if self.argument == HOLE:
            return [HOLE]
        return self.argument.copy()

    def get_operator(self):
        if self.operator == HOLE:
            return [HOLE]
        return self.operator.copy()

    def to_stmt(self):
        return f"<{self.value}, {self.operator}, {self.argument}>"
