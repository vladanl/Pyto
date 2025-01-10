from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_fh:
    long_description = readme_fh.read()

setup(
    name="pyto",
    version="1.10.1",
    author="Vladan Lucic",
    author_email="vladan@biochem.mpg.de",
    description=(
        "Detection and analysis of pleomorphic membrane-bound "
        + "molecular complexes in cryo-electron tomograms"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vladanl/Pyto",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18",
        "scipy>=1.4.1",
        "matplotlib>=3.2",
        "pandas>=1.0",
        "future>=0.18",
        "scikit-learn",
        "scikit-image",
        "statsmodels"
    ],
    include_package_data=True
)
