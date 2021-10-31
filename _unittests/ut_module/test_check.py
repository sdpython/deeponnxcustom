"""
@brief      test log(time=0s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from deeponnxcustom import check


class TestCheck(ExtTestCase):
    """Test style."""

    def test_check(self):
        check()


if __name__ == "__main__":
    unittest.main()
