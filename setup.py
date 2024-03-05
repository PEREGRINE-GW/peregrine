# setup.py
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="peregrine",
    version="0.0.2",
    packages=find_packages(),
    author="Uddipta Bhardwaj, James Alvey",
    author_email="ubhardwaj.gravity@gmail.com; j.b.g.alvey@uva.nl",
    description="peregrine is a Simulation-based Inference (SBI) library designed to perform analysis on a wide class of gravitational wave signals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
)