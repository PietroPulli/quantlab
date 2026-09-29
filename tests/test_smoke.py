"""First test: checks that the package is installed correctly."""

import quantlab


def test_package_imports():
    assert quantlab.__version__ == "0.1.0"
