import pytest

import ising


@pytest.fixture
def model():
    return ising.Model()


