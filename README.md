
# Pyto #

### General description ###

Detection and analysis of macromolecular complexes and proteins in 3D cryo-electron images (tomograms). While it was developed for neuronal synapses, it is applicable to other biological systems.

### Specific tasks ###

* The main purpose is the detection and analysis of pleomorphic membrane-bound molecular complexes cryo-electron  tomograms.

* Correlate images obtained from light microscopy and different modes of electron microscopy (transmission and scanning electron microscopy, focused ion beam microscopy). A part of this code is used
in [3DCT package for correlative microscopy](https://github.com/coraxx/3DCT.git), which provides GUI.

* Spatial distribution analysis of particles (such as complexes)

* Preprocessing membrane-bound particles (complexes) for subtomogram averaging.

* Mapping complexes to their exact position and angular orientation for visualization.

* Provides tools that assist general processing of cryo-electron tomograms (frame alignment, 3D reconstruction).


### Dependencies: ###

The current version is written in Python 3. The code was originally written in Python 2 and subsequently converted to Python 3.6. While most of the code is currently compatible with Python 2.7, this will not be enforced in future. 

Major dependencies of this package are:

+ NumPy
+ SciPy
+ Pandas
+ future

In addition, some parts depend on:

+ Matplotlib
+ Sklearn
+ Skimage
+ Statsmodels
+ SymPy: Currently needed only for development
+ PySeg: Needed only if Pyseg colocalization processing is followed by Pyto. Make sure Pyto comes before PySeg in PYTHONPATH because PySeg contains an older version of Pyto
* PyTorch: Needed only for neural net classification


### Installation ###

Put this directory to your PYTHONPATH


### Documentation ###

Please start from [Overview](doc/manuals/overview.pdf).


### Release history ###

* 1.11.1 (17.1.2026, svn r2263)
	* Added membrane / boundary normal vectors
	* Improved preprocessing for averaging and handling pixel size in Layers.rebin()
* 1.11.0 (27.10.2025, svn r2239)
	* Added colocalization example
	* Improved colocalization code and docs for pattern generation and reading star files
* 1.10.3 (7.05.2025, svn 2202)
	* Added AI tether classification code and example
* 1.10.2 (30.03.2025, svn r2181)
	* Added mapping particle code and example
	* Fixed some depreciation related issues for python 3.12
	* Docs improvements, small fixes
* 1.10.1 (07.01.2025, svn r2155)
  	* Added preprocessing for subtomo averaging example
	* Put back sources for manuals
* 1.10.0 (18.12.2024, svn r2140)
	* Added preprocessing for subtomo averaging
	* Improved particle handling (MultiParticlesets)
	* The above are needed for the Tether averaging project
* 1.9.2 (19.03.2024, svn r2102)
  	* Added projection methods for colocalization analysis
	* Added particle extraction from tomos
	* Removed dependence on imp module to make pyseg compatible with Python 3.12+
* 1.9.1 (18.10.2023, svn r2024) 
	* Noted future module dependency
	* Removed references to numpy.testing.Tester (removd from numpy 1.24) 
* 1.9 (7.12.2021, svn r1824)
	* Added colocalization analysis (developed for Martinez-Sanchez et al 2021 "Trans-synaptic assemblies link synaptic vesicles and neuroreceptors" DOI: 10.1126/sciadv.abe6204)
	* Added functionality to extract additional features of segments
* 1.8 (31.10.2021, svn r1771)
    * Added classes for basic geometrical shapes (Plane, Parallelogram)
    * Added bin_crop script
    * Improved analysis: Groups to/from Pandas, correlation analysis and graphs
* 1.7
    * Added presynaptic example
    * Improved docs
    * Small improvements
* 1.6
    * The first public release.


### License ###

Copyright (C) 2010  Vladan Lucic

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.


### Citing ###

Please consider citing us if you use Pyto:

* For general use, segmentation and analysis: Lučić V, Fernández-Busnadiego R, Laugks U and Baumeister W, 2016. Hierarchical detection and analysis of macromolecular complexes in cryo-electron tomograms using Pyto software. J Struct Biol. 196(3):503-514. http://dx.doi.org/10.1016/j.jsb.2016.10.004.

* For colocalization analysis, please cite: Daniel H. Orozco-Borunda, Antonio Martinez-Sanchez, Vladan Lucic, 2025, Spatial organization of assemblies of protein complexes by colocalization analysis BIORXIV/2025/684783.

* For 3D to 2D correlation, please cite: Arnold, J., J. Mahamid, V. Lucic, A. d. Marco, J.-J. Fernandez, Laugks, H.-A. Mayer, Tobias, W. Baumeister, and J. Plitzko, 2016. Site-specific cryo-focused ion beam sample preparation guided by 3-dimensional correlative microscopy. Biophysical Journal 110:860-869. http://dx.doi.org/10.1016/j.bpj.2015.10.053.

* For all other correlative work: Fukuda, Y., N. Schrod, M. Schaffer, L. R. Feng, W. Baumeister, and V. Lucic, 2014. Coordinate transformation based cryo-correlative methods for electron tomography and focused ion beam milling. Ultramicroscopy 143:15– 23. http://dx.doi.org/10.1016/j.ultramic.2013.11.008.

Thank you.


### Author ###

Vladan Lucic (vladan@biochem.mpg.de), Max Planck Institute for Biochemistry

