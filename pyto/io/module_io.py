"""
Contains class ModuleIO that loads a module

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import importlib


class ModuleIO(object):
    """
    Usage from an external program like a python shell or a notebook:

      >>> mod_io = ModuleIO(calling_dir=os.getcwd())
      >>> module_of_interest = mod_io.load(path=relative_path_from_external)
    """

    def __init__(self, calling_dir=''):
        """
        """
        self.calling_dir = calling_dir
    
    def load(self, path, preprocess=False):
        """
        Loads and returns a module that is located at the specified path.

        Intended for loading modules that hold results of a project, which 
        are likely located outside the python path.

        If arg preprocess is true, function module.main() is executed, where
        module is the loaded module.

        Arguments:
          - path: module path, can be absolute or relative to the external
          calling program
          - preprocess: flag indicating if function main() is executed

        Returns the loaded module
        """

        # figure out the path
        if not os.path.isabs(path):
            abs_path = os.path.normpath(
                os.path.join(self.calling_dir, path))
        else:
            abs_path = path
            
        # import work
        _, work_name = os.path.split(abs_path)
        work_base, _ = os.path.splitext(work_name)
        spec = importlib.util.spec_from_file_location(work_base, abs_path)
        work = spec.loader.load_module(spec.name)

        # work preprocessing
        if preprocess:
            work.main()

        return work
