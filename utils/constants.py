"""Global constants for use across `RPP`."""

import os.path

RPP_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Test directory
TEST_DIR = os.path.join(RPP_ROOT_DIR, "tests")

# Test data
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")