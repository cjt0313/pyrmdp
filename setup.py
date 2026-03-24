from setuptools import setup, find_packages

setup(
    name="pyrmdp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "z3-solver",
        "networkx",
        "pyvis",
    ],
    description="Relational MDPs with First-Order Decision Diagrams",
    author="Jim",
)
