from setuptools import setup, find_packages
import seahorse

setup(
    name = "seahorse",
    packages = find_packages(),
    version = seahorse.__version__,
    author = "jsgounot",
    url = 'https://github.com/jsgounot/Seahorse',
    install_requires = ["seaborn"]
)
