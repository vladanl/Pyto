from setuptools import setup, find_packages

setup(
    name="pyto",
    version="1",
    author="Vladan Lucic",
    author_email="vladan@biochem.mpg.de",
    packages=find_packages(),
    python_requires=">=3.0",
    install_requires=[
        "numpy>=1.18",
        "scipy>=1.41",
        "matplotlib>=3.2",
        "pandas>=1.0"
    ],
    include_package_data=True
)
