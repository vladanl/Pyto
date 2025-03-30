
# Mapping particles to tomograms #

Executing the example code from the notebook extracts particle location and orentation parameters from a star file and makes a tomogram where a specified glyph (an average density or a generic glyph) is postioned at the extracted coordinates and oriented according to the extracted angles.

Developed for Orozco-Burunda et al, Direct cryo-ET detection of native SNARE and Munc13 protein bridges using AI classification and preprocessing, bioRxiv https://doi.org/10.1101/2024.12.18.629213

## Visualization instructions ##

### ChimeraX ###

To visualize mapped particles using an average density or any previously generated particle or a glyph

* Set the star file, glyph and output parameters and execute the notebook
* Open the generated map in ChimeraX
* Open the corresponding tomogram or segmented membranes in the same ChimeraX session

### Paraview ###

To visualize mapped particles using an existing Paraview glyph (these are all axially symetric, so they show only two of the rotation angles):

* Set the starfile and output parameters, and execute the notebook
* Open the generated csv file in Paraview
* Convert the data to points, chose columns x/y/z_map_a for positions
* Make unit vectors from v_x/y/z columns
* Chose a glyph and map it using the positions and the unit vectors


