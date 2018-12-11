import numpy as np
import matplotlib.pyplot as plt
import pprint


def shift(register, matrix, g):
    newreg = np.zeros(len(register))
    for i in range(len(g) - 1):
        for j in range(len(g) - 1):
            if matrix[i, j]:
                newreg[i] += register[j]
    return newreg % 2


def bit_correlation(a, b):
    res = 0

    for i in range(len(a)):
        if a[i] == b[i]:
            res += 1
        else:
            res -= 1
    return res

def process(g, matrix_degree, sourcebit):
    # g = [1, 0, 0, 1, 0, 1]
    res = {}
    g_degree = len(g) - 1

    matrix = np.zeros((g_degree, g_degree), dtype="int")
    matrix[:, -1] = g[:-1]

    for i in range(g_degree - 1):
        matrix[i + 1, i] = 1

    for i in range(1, matrix_degree):
        matrix = matrix.dot(matrix)

    s = set()

    register = [0] * g_degree
    register[0] = 1

    m_sequence = []

    for i in range(2**g_degree - 1):
        # print(np.asarray(register, dtype="int"))
        m_sequence.append(register[-1 - sourcebit])
        register = shift(register, matrix, g)
        s.add(tuple([int(i) for i in register]))
        print(tuple([int(i) for i in register]))

    res["m-seq"] = [int(i) for i in m_sequence]
    res["field_elems"] = s

    m_sequence = np.asarray(m_sequence, dtype="int")

    data = []

    for i in range(2*(2**g_degree) - 1):
        data.append((bit_correlation(m_sequence, np.roll(m_sequence, -i))))

    plt.plot(range(2*(2**g_degree) - 1), data)

    plt.show()

    return res


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    # g = [1, 1, 1, 1, 0, 1]
    g = [1, 1, 0, 0, 0, 0, 1]
    res1 = process(g, 1, 0)


    print(res1["m-seq"])

    # data = []
    # for i in range( 2* (len(res1["m-seq"]) - 1)):
    #     data.append(bit_correlation(res1["m-seq"], np.roll(res3["m-seq"], i)))
    # plt.plot(range( 2* (len(res1["m-seq"]) - 1)), data)
    # plt.show()

# x^5 + x^3 + 1