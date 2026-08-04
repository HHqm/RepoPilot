import unittest

from src.calculator import safe_divide


class SafeDivideTests(unittest.TestCase):
    def test_regular_division(self):
        self.assertEqual(safe_divide(8, 2), 4)

    def test_zero_divisor_has_domain_error(self):
        with self.assertRaisesRegex(ValueError, "divisor must not be zero"):
            safe_divide(1, 0)


if __name__ == "__main__":
    unittest.main()

