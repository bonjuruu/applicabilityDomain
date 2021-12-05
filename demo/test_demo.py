"""This is a demo for pytest.

The idea is that we like to prepare autometic setup between anytest, and it should runs in parallel.
E.g. In every test, if np.random.rand is called, it should alway returns the same value.
Further reading is here: https://docs.pytest.org/en/6.2.x/fixture.html#fixture-availability

Moreover, since we have many methods which have similar behaviour. It's good to
group tests in class, so we might try to reuse them.

To run this demo:
$ python -m pytest -s ./demo/test_demo.py 
"""
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def setup():
    """Setup test. Reset RNG seed."""
    print('Reset seed')
    np.random.seed(1234)


@pytest.fixture
def gen_val(request):
    """Generate 3 random variables

    Note: `request` is an object injected from the decorator. If the method has
    a fixture decorator, every parameter can access the `config` object, and
    they share the same cache.

    'test_cache' is the name of the cache file, which can be found under the `.pytest_cache`
    directory.
    """

    print('Generate data')
    # Save and load variables
    val = request.config.cache.get('test_cache', None)
    if val is None:
        val = np.random.rand(3)
        request.config.cache.set('test_cache', None)
    return val


class TestDummy:
    def test_check1(self, gen_val):
        """Check the length"""
        print('gen_val:', gen_val)
        assert len(gen_val) == 3

    def test_check2(self, gen_val):
        """Check the value"""
        print('gen_val:', gen_val)
        assert np.all((gen_val < 1) & (gen_val > 0))
