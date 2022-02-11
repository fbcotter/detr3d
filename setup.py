import os
import re

from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read metadata from version file

setup(
    name='detr3d',
    version='0.0.1',
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("detr3d"),
    license="Free To Use",
    keywords="none",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests", "data"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free To Use But Restricted",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
    install_requires=['numpy', 'six', 'torch'],
)

# vim:sw=4:sts=4:et
