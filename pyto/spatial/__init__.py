"""
Tools for spatial analysis

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
__version__ = "$Revision$"

from .coloc_analysis import ColocAnalysis
from . import test

from numpy.testing import Tester
test = Tester().test
