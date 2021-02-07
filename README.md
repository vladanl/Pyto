
# Pyto #

### Description ###

The main purpose of Pyto package is to provide tools for the detection and analysis of pleomorphic membrane-bound molecular complexes in 3D images (tomograms). It is designed for cryo-electron tomograms of neuronal synapses, but it can be used for other biological systems.

In addition, Pyto can be used to correlate images obtained from light microscopy and different modes of electron microscopy (transmission and scanning electron microscopy, focused ion beam microscopy). A part of this code is used 
in [3DCT package for correlative microscopy](https://github.com/coraxx/3DCT.git), which provides GUI.

Finaly, tools that assist general processing of cryo-electron tomograms (frame alignment, 3D reconstruction) are also provided. 


### Dependencies: ###

The current version is written in Python 3. The code was originally written in Pythin 2 and subsequently converted to Python 3.6. While most of the code is currently compatible with Python 2.7, this will not be enforced in future. 

The major dependencies of this package are:

* NumPy
+ SciPy

In addition, some parts depend on:

+ Matplotlib
+ Pandas


### Installation ###

Put this directory to your PYTHONPATH


### Documentation ###

Please start from [Overview](doc/manuals/overview.pdf).


### Release history ###

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

* For 3D to 2D correlation, please cite: Arnold, J., J. Mahamid, V. Lucic, A. d. Marco, J.-J. Fernandez, Laugks, H.-A. Mayer, Tobias, W. Baumeister, and J. Plitzko, 2016. Site-specific cryo-focused ion beam sample preparation guided by 3-dimensional correlative microscopy. Biophysical Journal 110:860-869. http://dx.doi.org/10.1016/j.bpj.2015.10.053.

* For all other correlative work: Fukuda, Y., N. Schrod, M. Schaffer, L. R. Feng, W. Baumeister, and V. Lucic, 2014. Coordinate transformation based cryo-correlative methods for electron tomography and focused ion beam milling. Ultramicroscopy 143:15– 23. http://dx.doi.org/10.1016/j.ultramic.2013.11.008.

Thank you.


### Author ###

Vladan Lucic (vladan@biochem.mpg.de), Max Planck Institute for Biochemistry

