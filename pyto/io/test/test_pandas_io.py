"""
Tests class PandasIO

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import shutil
import pickle
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pyto.io.pandas_io import PandasIO


class TestPandasIO(np_test.TestCase):
    """
    Tests PandasIO
    """

    def setUp(self):
        """Setup file paths and tables
        """

        self.do_tear_down = True
        
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.base_1 = 'dir_1/table_1'
        self.base_2 = 'dir_2/table_1.pkl'
        self.base_2_json = 'dir_2/table_2_json.pkl'
        self.table_1 = pd.DataFrame(
            {'number': [1, 2, 3], 'letter': ['a', 'b', 'c']}, index=[7, 8, 9]) 
        self.table_11 = pd.DataFrame(
            {'number': [11, 22, 33], 'letter': ['aa', 'bb', 'cc']},
            index=[7, 8, 9]) 

    def test_write_read(self):
        """Tests write() and read() methods.
        """

        # Note: arg calling_dir has to be the dir of this file because the
        # specified read/write pats are relative to this file's dir
        
        # standard
        PandasIO.write(
            table=self.table_1, base=self.base_1, calling_dir=self.current_dir, 
            file_formats=['pkl', 'hdf5', 'json'], verbose=False)
        actual = PandasIO.read(
            calling_dir=self.current_dir, base=self.base_1, verbose=False)
        np_test.assert_equal(self.table_1.equals(actual), True)

        # save only json
        PandasIO.write(
            table=self.table_1, base=self.base_2, calling_dir=self.current_dir,
            file_formats=['json'], verbose=False)
        actual = PandasIO.read(
            calling_dir=self.current_dir, base=self.base_2, verbose=False)
        np_test.assert_equal(self.table_1.equals(actual), True)

    def test_write_read_table(self):
        """Tests write_table() and read_table() methods.
        """

        # Note: arg calling_dir has to be the dir of this file because the
        # specified read/write pats are relative to this file's dir
        
        # check calling_dir initialization
        pio = PandasIO(
            calling_dir=None, file_formats=['pkl', 'hdf5', 'json'],
            verbose=False)
        np_test.assert_equal(pio.calling_dir, os.getcwd())
        pio = PandasIO(file_formats=['pkl', 'hdf5', 'json'], verbose=False)
        np_test.assert_equal(pio.calling_dir, os.getcwd())

        # try all possibilities
        pio = PandasIO(
            calling_dir=self.current_dir, file_formats=['pkl', 'hdf5', 'json'],
            verbose=False)
        pio.write_table(table=self.table_1, base=self.base_1)
        actual = pio.read_table(base=self.base_1)
        np_test.assert_equal(self.table_1.equals(actual), True)
        
        # overwrite the previous
        pio.write_table(table=self.table_11, base=self.base_1, overwrite=True)
        actual = pio.read_table(base=self.base_1)
        np_test.assert_equal(self.table_11.equals(actual), True)

    def test_write_read_table_json(self):
        """Tests write_table() and read_table() for json 
        """
        #self.do_tear_down = False
        # save only json
        pio = PandasIO(
            calling_dir=self.current_dir, file_formats=['json'], verbose=False)
        pio.write_table(table=self.table_1, base=self.base_2_json)
        np_test.assert_equal(
            os.path.exists(os.path.join(self.current_dir, self.base_2_json)),
            True)
        actual = pio.read_table(base=self.base_2_json)
        np_test.assert_equal(self.table_1.equals(actual), True)

        # try to overwrite the previous when overwrite=False
        np_test.assert_raises(
            ValueError, pio.write_table,
            table=self.table_11, base=self.base_2_json, overwrite=False)

        # check write / read with repeated index values
        pio = PandasIO(
            calling_dir=self.current_dir, file_formats=['json'], verbose=False)
        table = self.table_1.copy()
        table.index = np.ones(table.shape[0], dtype='int') * 5
        pio.write_table(table=table, base=self.base_2_json, overwrite=True)
        table2 = pio.read_table(base=self.base_2_json)
        np_test.assert_equal((table2.index.to_numpy() == 5).all(), True)
        np_test.assert_array_equal(table2.to_numpy(), table.to_numpy())

    def test_get_pickle_path(self):
        """ Tests get_pickle_path()
        """

        self.do_tear_down = False

        # relative paths
        pio = PandasIO(calling_dir='call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z'),
            ('call_dir/x/y/z.pkl', 'x/y/z.pkl'))
        pio = PandasIO(calling_dir='call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z.pkl'),
            ('call_dir/x/y/z.pkl', 'x/y/z.pkl'))
        pio = PandasIO(calling_dir='call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z_json.pkl'),
            ('call_dir/x/y/z.pkl', 'x/y/z.pkl'))
        pio = PandasIO(calling_dir='call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z', json=True),
            ('call_dir/x/y/z_json.pkl', 'x/y/z_json.pkl'))
        pio = PandasIO(calling_dir='call_dir/dir_2')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z.pkl', json=True),
            ('call_dir/dir_2/x/y/z_json.pkl', 'x/y/z_json.pkl'))

        # absolute paths
        pio = PandasIO(calling_dir='/abs/call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='x/y/z', json=True),
            ('/abs/call_dir/x/y/z_json.pkl', 'x/y/z_json.pkl'))
        pio = PandasIO(calling_dir='call_dir')
        np_test.assert_equal(
            pio.get_pickle_path(base='/abs/x/y/z', json=True),
            ('/abs/x/y/z_json.pkl', '/abs/x/y/z_json.pkl'))
        
    def tearDown(self, mode=None):
        """Remove tables
        """
        
        #self.do_tear_down = False
        if self.do_tear_down:
        
            # remove files
            try:
                if (mode is None) or (mode == 1):
                    shutil.rmtree(os.path.dirname(os.path.join(
                        self.current_dir, self.base_1)))
            except FileNotFoundError:
                pass
            try:
                if (mode is None) or (mode == 2):
                    shutil.rmtree(os.path.dirname(os.path.join(
                        self.current_dir, self.base_2)))
            except FileNotFoundError:
                pass

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPandasIO)
    unittest.TextTestRunner(verbosity=2).run(suite)
       
        

    
