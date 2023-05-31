import numpy as np


def rindex_array(inarray):
    array_index = [40, 2, 3, 0, 1, 14, 15, 12, 13, 4, 5, 8, 24, 20, 21, 10, 11, 6, 7, 22, 23, 16, 17, 19, 18, 9, 25, 26,
                   27, 28, 29, 31, 30]
    l = len(array_index)
    L = len(inarray)
    L = L // l * l
    outarray = np.zeros((1, L))
    for j in range(L):
        i = j + 1
        d = i // l
        ind = i % l
        if ind == 0:
            d = d - 1
            ind = l
            outarray[0, d * l + array_index[ind - 1] + 2 - 1] = inarray[i - 1]
        elif ind == 1:
            outarray[0, d * l + 1 - 1] = inarray[i - 1]
        else:
            outarray[0, d * l + array_index[ind - 1] + 2 - 1] = inarray[i - 1]

    return outarray
