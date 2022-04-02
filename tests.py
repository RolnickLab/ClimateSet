import unittest


class Testing(unittest.TestCase):
    def dummy_test(self):
        a = True
        b = True
        self.assertEqual(a, b)
