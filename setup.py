from setuptools import setup, find_packages

setup(
    name="pyrmdp",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "pyvis",
    ],
    extras_require={
        "z3": ["z3-solver"],
        "llm": ["openai"],
        "all": ["z3-solver", "openai"],
    },
    description="Relational MDPs with First-Order Decision Diagrams",
    author="Jim",
)
