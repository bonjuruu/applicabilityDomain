"""Setup file for the ADAD package

To install this local package
    python -m pip install .
To upgrade this package
    python -m pip install --upgrade .

TODO: Add the list of required packages
"""
from setuptools import setup, find_packages

setup(
    name='ADAD',
    description='Applicability Domain & Adversarial Defences',
    packages=find_packages(),
    version='0.0.1',
    python_requires='>=3.6',
)
