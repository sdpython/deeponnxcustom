"""
@brief      test tree node (time=4s)
"""
import unittest
from pyquickhelper.loghelper import BufferedPrint
from pyquickhelper.pycode import ExtTestCase
from onnxcustom.__main__ import main


class TestCliCheck(ExtTestCase):

    def test_profile_check(self):
        st = BufferedPrint()
        main(args=['check', '--help'], fLOG=st.fprint)
        res = str(st)
        self.assertIn("usage: check", res)


if __name__ == "__main__":
    unittest.main()
