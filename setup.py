#!/usr/bin/env python
from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        packages=find_packages(include=["src", "src.*", "data", "data.*"]),
        include_package_data=True,
    )
