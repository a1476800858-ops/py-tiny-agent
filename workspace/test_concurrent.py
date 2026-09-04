import unittest

from counter_demo import count_concurrently


class ConcurrentCountingTest(unittest.TestCase):
    def test_count(self) -> None:
        self.assertEqual(count_concurrently(1000), 1000)


if __name__ == "__main__":
    unittest.main()
