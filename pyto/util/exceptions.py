"""
Defines custom exceptions.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


class ReferenceError(Exception):
    """
    Exception raised when the reference used for inferrence cannot be found.
    """
    pass
