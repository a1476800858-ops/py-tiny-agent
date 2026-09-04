import unittest

try:
    from .math import multiply
except ImportError:
    from workspace.math import multiply


class MultiplyTest(unittest.TestCase):
    def test_cases(self) -> None:
        for values, expected in [((3, 4), 12), ((-3, -4), 12), ((5, -2), -10), ((0, 123), 0), ((456, 0), 0), ((1, 999), 999), ((7, 7), 49), ((1000, 2000), 2_000_000)]:
            with self.subTest(values=values):
                self.assertEqual(multiply(*values), expected)


if __name__ == "__main__":
    unittest.main()
