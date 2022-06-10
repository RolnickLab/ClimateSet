import unittest
import torch
import causal.train.get_ortho_constraint


class Testing(unittest.TestCase):
    def test_dummy(self):
        a = True
        b = True
        self.assertEqual(a, b)

    def test_ortho_constraint(self):
        get_ortho_constraint = causal.train.get_ortho_constraint
        dim = 5

        c = get_ortho_constraint(torch.eye(dim))
        self.assertEqual(c.item(), 0)

        mat = torch.zeros(dim, dim)
        mat[torch.arange(dim), torch.randperm(dim)] = 1
        c = get_ortho_constraint(mat)
        self.assertEqual(c.item(), 0)

        # similar to ws...

        mat = torch.rand(dim, dim)
        c = get_ortho_constraint(mat)
        self.assertGreater(c.item(), 0)
