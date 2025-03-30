
# Preprocessing for subtomo averaging examples #

Contains two examples of particle preprocessing for subtomo averaging

### Notebook extract_tethers_example.ipynb ###

* Procedure for extracting and preprocessing tether particles, and related boundary and segment particles for subtomogram averaging, as presented in:
https://doi.org/10.1101/2024.12.18.629213 .
* Requires pyto presynaptic analysis
* Input data is read from pandas DataFrames

### Notebook extract_tethers_2stars.ipynb ###

* Similar to the above, but has wider applicability because it does not use segment particles (called the Focused particle set in the manuscript)
* Does not require pyto presynaptic analysis
* Input data is read from two relion style star files


### General instructions ###

Further info and instructions are given in the respective notebooks.


