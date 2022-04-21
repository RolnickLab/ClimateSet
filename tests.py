import unittest


class Testing(unittest.TestCase):
    def test_dummy(self):
        a = True
        b = True
        self.assertEqual(a, b)
