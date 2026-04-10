#!/usr/bin/env python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(
            include=["src", "src.*", "data", "data.*", "config", "config.*", "scripts", "scripts.*"]
        ),
        include_package_data=True,
    )
