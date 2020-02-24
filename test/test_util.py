import unittest
from blurnn._util import safe_apply, combine_iterators

class TestUtilFunctions(unittest.TestCase):
    def test_safeapply(self):
        const_func = (lambda _, **kw: 'const')
        is_zero = (lambda x: x == 0)
        self.assertEqual(safe_apply(
            is_zero,
            const_func,
            x=0,
        )(1), 'const')
        self.assertEqual(safe_apply(
            is_zero,
            const_func,
            x=1,
        )(1), 1)

    def test_combine_iterators(self):
        one_to_three = range(3)
        one_to_five = range(5)
        yields = [(0, 0), (1, 1), (2, 2), (None, 3), (None, 4)]
        gen = combine_iterators(one_to_three, one_to_five)
        for value in yields:
            self.assertEqual(gen.__next__(), value)

if __name__ == '__main__':
    unittest.main()