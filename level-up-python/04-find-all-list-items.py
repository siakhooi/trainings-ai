def index_all(list1, v, parentlist=None):
    result=[]
    if parentlist is None:
        parentlist = []
    for index, value in enumerate(list1):
        if value == v:
            result.append([*parentlist, index])
        elif isinstance(value,list):
            result.extend(index_all(value, v, [*parentlist, index]))
    return result


# commands used in solution video for reference
if __name__ == '__main__':
    example = [[[1, 2, 3], 2, [1, 3]], [1, 2, 3]]
    print(index_all(example, 2))  # [[0, 0, 1], [0, 1], [1, 1]]
    print(index_all(example, [1, 2, 3]))  # [[0, 0], [1]]
