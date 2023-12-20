#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="icad",
      version="1.0",
      description="ICAD: Intracranial Atherorclerotic classification",
      author="VHIR",
      author_email="pere.canals@vhir.org",
      packages=find_packages(),
      package_data={"icad": []},
      install_requires=[
    ]
)
