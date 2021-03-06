"""
@file
@brief Mathemical functions.
"""


def decompose_permutation(perm):
    """
    Decomposes a permutation into transitions.

    :param perm: permutation (integers)
    :return: list of tuples

    .. note::
        The function does not check *perm* is a permutation.
        If the input value is wrong, the execution could
        end up in an infinite loop.

    .. runpython::
        :showcode:

        import pprint
        from deeponnxcustom.tools.math_helper import decompose_permutation

        pprint.pprint(decompose_permutation((1, 0)))
        pprint.pprint(decompose_permutation((2, 0, 1)))
        pprint.pprint(decompose_permutation((1, 4, 2, 3, 0)))

    Function @see fn apply_transitions applies this series
    of transitions.
    """
    perm = list(perm)
    transitions = []
    while True:
        index = -1
        for i, p in enumerate(perm):
            if p != i:
                index = i
                break
        if index == -1:
            break
        while perm[index] != index:
            a, b = index, perm[index]
            transitions.append((b, a))
            perm[a], perm[b] = perm[b], perm[a]
            index = b

    return list(reversed(transitions))


def apply_transitions(n, transitions):
    """
    Applies a list of transitions (permutations of two elements)
    on the first *n* integers.

    :param n: number of elements in the permutation
    :param transitions: list of transitions
    :return: permuted ensemble
    """
    ens = list(range(n))
    for a, b in transitions:
        ens[a], ens[b] = ens[b], ens[a]
    return ens
