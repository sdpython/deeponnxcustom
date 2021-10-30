"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from deeponnxcustom.tools.math_helper import (
    apply_transitions, decompose_permutation)


class TestMathHelper(ExtTestCase):

    def test_apply_transitions(self):
        self.assertEqual([1, 0], apply_transitions(2, [(0, 1)]))
        self.assertEqual([1, 0, 2], apply_transitions(3, [(0, 1)]))
        self.assertEqual([1, 2, 0], apply_transitions(3, [(0, 1), (1, 2)]))
        self.assertEqual([0, 1, 2], apply_transitions(3, [(0, 1), (1, 0)]))

    def test_decompose_permutation(self):
        perms = [[1, 0, 2], [1, 2, 0], [0, 1], [1, 0], (2, 0, 1),
                 (1, 3, 2, 0), (2, 3, 1, 0), (2, 0, 3, 1)]
        for perm in perms:
            with self.subTest(perm=perm):
                res = decompose_permutation(perm)
                back = apply_transitions(len(perm), res)
                self.assertEqual(list(perm), back)


if __name__ == "__main__":
    unittest.main()
