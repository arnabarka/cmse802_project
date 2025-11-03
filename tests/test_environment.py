# tests/test_environment.py
import unittest
import sys
from pathlib import Path


class TestEnvironment(unittest.TestCase):
    """Basic environment and repository structure checks."""

    def test_python_version(self):
        """Confirm Python 3.9 or newer."""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual((major, minor), (3, 9), f"Python version too old: {major}.{minor}")

    def test_project_structure(self):
        """Ensure essential project folders and files exist."""
        root = Path(__file__).resolve().parents[1]
        for p in ["src", "tests", "README.md", "requirements.txt"]:
            self.assertTrue((root / p).exists(), f"Missing {p}")


if __name__ == "__main__":
    unittest.main()
