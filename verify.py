"""Run the entire original suite, including real SQLite reopen and rollback."""
import sys
import unittest


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is required")
    suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
