from setuptools import setup, find_packages

setup(
    name = "my_toolkit",
    version = "0.1.0",
    packages = find_packages(),
    install_requires = [],
    description = "A personal python library for fitting functions to data",
    author = "Rens van Eck",
    author_email = "rensvaneck@gmail.com",
    url = "https://github.com/Renseck/my_toolkit",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",        
    ],
)