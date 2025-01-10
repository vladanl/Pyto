"""

Tests module analysis.groups.

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from builtins import zip
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import pickle
import os.path
import sys
import unittest

import numpy as np
import numpy.testing as np_test
import scipy
import pandas as pd

import pyto.util.scipy_plus
from pyto.analysis.catalog import Catalog
from pyto.analysis.groups import Groups
from pyto.analysis.observations import Observations
import pyto.scene.test.common as scene_cmn
from pyto.analysis.test import common


# set output
out = sys.stdout  # tests output, but prints lots of things
out = None        # clean output but doesn't test print

class TestGroups(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Makes Groups object.
        """
        self.groups = self.makeInstance()
        self.groups2 = self.makeInstance2()

    def testGetSet(self):
        """
        Test setting, getting and deleting items/attributes using both
        dictionary and attribute forms.
        """

        # make instance
        groups = Groups()
        groups.a = 1
        groups['b'] = 2
        groups._c = 3

        np_test.assert_equal(groups.a, 1)
        np_test.assert_equal(groups['a'], 1)
        np_test.assert_equal(groups.b, 2)
        np_test.assert_equal(groups['b'], 2)
        np_test.assert_equal(groups._c, 3)
        np_test.assert_equal(groups.get('_c', None) is None, True)

        np_test.assert_equal(list(groups.keys()), ['a', 'b'])

        delattr(groups, 'b')
        np_test.assert_equal(groups.get('b') is None, True)
        raised = False
        try:
            groups.b
        except KeyError:
            raised = True
        np_test.assert_equal(raised, True)
        np_test.assert_equal(groups.pop('a'), 1)
        np_test.assert_equal(groups.get('a') is None, True)
        raised = False
        try:
            groups.b
        except KeyError:
            raised = True
        np_test.assert_equal(raised, True)

    @classmethod
    def makeInstance(cls):
        """
        Makes and returns a Groups object.
        """

        groups = Groups()
        groups.ga = Observations()
        groups.ga.identifiers = ['ia1', 'ia3', 'ia5']
        groups.ga.scalar = [2, 6, 10]
        groups.ga.ids = [2*np.arange(2), np.arange(6), np.array([])]
        groups.ga.vector = [2*np.arange(2) + 1, np.arange(6) + 1,
                               np.array([])]
        groups.ga.properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups.ga.indexed.update(['ids', 'vector'])
        groups['gb'] = Observations()
        groups['gb'].identifiers = ['ib1', 'ib2', 'ib3']
        groups['gb'].scalar = [2, 4, 6]
        groups['gb'].ids = [2*np.arange(2), np.array([]), np.arange(4)]
        groups['gb'].vector = [2*np.arange(2) + 1,
                               np.array([]), np.arange(4) + 1]
        groups['gb'].properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups['gb'].indexed.update(['ids', 'vector'])
        groups['gc'] = Observations()
        groups['gc'].identifiers = []
        groups['gc'].scalar = []
        groups['gc'].ids = [np.array([])]
        groups['gc'].vector = [np.array([])]
        groups['gc'].properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups['gc'].indexed.update(['ids', 'vector'])
        groups['gd'] = Observations()
        groups.gd.identifiers = ['id6']
        groups.gd.scalar = [12]
        groups.gd.ids = [np.arange(5) + 1]
        groups.gd.vector = [np.arange(5) + 2]
        groups.gd.properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups.gd.indexed.update(['ids', 'vector'])

        return groups

    @classmethod
    def makeSymmetricalInstance(cls):
        """
        Makes instance that has same identifiers for all groups.
        """

        groups = Groups()
        groups.ga = Observations()
        groups.ga.identifiers = ['i1', 'i3', 'i5']
        groups.ga.scalar = [2, 6, 10]
        groups.ga.ids = [2*np.arange(2), np.arange(6), np.array([])]
        groups.ga.vector = [2*np.arange(2) + 1, np.arange(6) + 1,
                               np.array([])]
        groups.ga.properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups.ga.indexed.update(['ids', 'vector'])
        groups['gb'] = Observations()
        groups['gb'].identifiers = ['i1', 'i3', 'i5']
        groups['gb'].scalar = [2, 4, 6]
        groups['gb'].ids = [2*np.arange(2), np.array([]), np.arange(4)]
        groups['gb'].vector = [2*np.arange(2) + 1,
                               np.array([]), np.arange(4) + 1]
        groups['gb'].properties.update(['scalar', 'identifiers',
                                        'ids', 'vector'])
        groups['gb'].indexed.update(['ids', 'vector'])

        return groups

    @classmethod
    def makeSameIdInstance(cls):
        """
        Makes and returns a Groups object where observations have same indices.
        """

        groups = Groups()
        groups.ga = Observations()
        groups.gb = Observations()

        groups.ga.identifiers = ['ia1', 'ia3', 'ia5']
        groups.ga.ids = 3*[np.array([1, 3, 5])]
        groups.ga.vector = [np.array([2,4,6]), np.array([3,5,7]),
                            np.array([4,6,8])]
        groups.ga.properties.update(['identifiers', 'ids', 'vector'])
        groups.ga.indexed.update(['ids', 'vector'])

        groups.gb.identifiers = ['ib1', 'ib2']
        groups.gb.ids = 2*[np.array([1, 2, 3])]
        groups.gb.vector = [np.array([10,20,30]), np.array([20,40,60])]
        groups.gb.properties.update(['identifiers', 'ids', 'vector'])
        groups.gb.indexed.update(['ids', 'vector'])

        return groups

    @classmethod
    def makeInstance2(cls):
        """
        Makes and returns a Groups object that has more values.
        """

        groups = Groups()

        groups.ga = Observations()
        groups.ga.setValue(
            name='ids', identifier='ia1', value=[1,3,5,7,9], indexed=True)
        groups.ga.setValue(
            name='xxx', identifier='ia1',
            value=np.array([1.1, 2.1, 3.1, 4.1, 5.1]), indexed=True)
        groups.ga.setValue(
            name='ids', identifier='ia2', value=[3,4,5], indexed=True)
        groups.ga.setValue(
            name='xxx', identifier='ia2',
            value=np.array([0.1, 1.1, 2.1]), indexed=True)
        groups.ga.setValue(
            name='ids', identifier='ia3', value=[2,3,5,7,8], indexed=True)
        groups.ga.setValue(
            name='xxx', identifier='ia3',
            value=np.array([3.6, 4.6, 5.6, 6.6, 7.6]), indexed=True)

        groups.gb = Observations()
        groups.gb.setValue(
            name='ids', identifier='ib1', value=[1,3,4,5,6], indexed=True)
        groups.gb.setValue(
            name='xxx', identifier='ib1',
            value=np.array([1.3, 2.3, 3.3, 4.3, 5.3]), indexed=True)
        groups.gb.setValue(
            name='ids', identifier='ib2', value=[2,4,6,8,10], indexed=True)
        groups.gb.setValue(
            name='xxx', identifier='ib2',
            value=np.array([3.3, 4.3, 5.3, 6.3, 7.3]), indexed=True)
        groups.gb.setValue(
            name='ids', identifier='ib3', value=[2,4,5,7,9,10], indexed=True)
        groups.gb.setValue(
            name='xxx', identifier='ib3',
            value=np.array([4.7, 5.7, 6.7, 7.7, 8.7, 6.7]), indexed=True)
        groups.gb.setValue(
            name='ids', identifier='ib4', value=[], indexed=True)
        groups.gb.setValue(
            name='xxx', identifier='ib4',
            value=np.array([]), indexed=True)

        return groups

    @classmethod
    def makeStats(cls):
        """
        Makes and returns a statistics-like Groups object, as retured
        by doStats()
        """

        groups = Groups()

        for bin in ['0-2', '2-4', '4-10']:
            groups[bin] = Observations()
            groups[bin].setValue(
                name='ids', identifier='g1', value=[0], indexed=True)
            groups[bin].setValue(
                name='ids', identifier='g2', value=[0], indexed=True)
            groups[bin].setValue(
                name='ids', identifier='g3', value=[0], indexed=True)
        groups['0-2'].setValue(
            name='probability', identifier='g1', value=[0.4], indexed=True)
        groups['2-4'].setValue(
            name='probability', identifier='g1', value=[0.5], indexed=True)
        groups['4-10'].setValue(
            name='probability', identifier='g1', value=[0.1], indexed=True)
        groups['0-2'].setValue(
            name='probability', identifier='g2', value=[0.2], indexed=True)
        groups['2-4'].setValue(
            name='probability', identifier='g2', value=[0.4], indexed=True)
        groups['4-10'].setValue(
            name='probability', identifier='g2', value=[0.4], indexed=True)
        groups['0-2'].setValue(
            name='probability', identifier='g3', value=[0.], indexed=True)
        groups['2-4'].setValue(
            name='probability', identifier='g3', value=[0.4], indexed=True)
        groups['4-10'].setValue(
            name='probability', identifier='g3', value=[0.6], indexed=True)

        return groups

    def make_tables(self):
        """
        Makes Pandas tables
        """

        # data (alpha.alpha_2 contains no indexed data)
        ids = {
            'alpha' : {'alpha_1' : range(5), 'alpha_4' : np.arange(1,8,2)},
            'bravo' : {
                'bravo_1' : np.arange(1,6), 'bravo_2' : np.arange(1,8,2),
                'bravo_3' : np.arange(0,21,5)}}
        vector_1 = [
            1, 1, 1, 3, 3,
            1, 3, 5, 7,
            1, 1, 3, 5, 5,
            3, 5, 7, 9,
            3, 3, 5, 7, 7]
        vector_2 = np.array(vector_1) + 10
        scalar_data = {
            'alpha' : {
                'alpha_1': range(5), 'alpha_2': None,
                'alpha_4': np.arange(1,8,2)},
            'bravo' : {
                'bravo_1' : np.arange(1,6), 'bravo_2' : np.arange(1,8,2),
                'bravo_3' : np.arange(0,21,5)}}
        scalar_1 = [1, 9, 2, 3, 4, 5]
        scalar_2 = np.array(scalar_1) + 10

        # make indexed data table and put group names, identifiers and ids
        indexed_table = None
        for g_name in ids:
            for ident in ids[g_name]:
                local_data = pd.DataFrame({
                    'group' : g_name,
                    'identifiers' : ident,
                    'ids' : ids[g_name][ident]})
                try:
                    indexed_table = pd.concat(
                        [indexed_table, local_data], axis=0, ignore_index=True)
                except NameError:
                    indexed_table = local_data

        # add vector data
        indexed_table = pd.concat(
            [indexed_table, pd.Series(vector_1, name='vector_1')], axis=1)
        indexed_table = pd.concat(
            [indexed_table, pd.Series(vector_2, name='vector_2')], axis=1)

        # make scalar data table and put group names and identifiers
        scalar_table = None
        for g_name in scalar_data:
            local_data = pd.DataFrame({
                'group' : g_name,
                'identifiers' : list(scalar_data[g_name].keys())})
            try:
                scalar_table = pd.concat(
                    [scalar_table, local_data], axis=0, ignore_index=True)
            except NameError:
                scalar_table = local_data

        # add scalar data
        scalar_table = pd.concat(
            [scalar_table, pd.Series(scalar_1, name='scalar_1')], axis=1)
        scalar_table = pd.concat(
            [scalar_table, pd.Series(scalar_2, name='scalar_2')], axis=1)

        return indexed_table, scalar_table

    def test_skip_name(self):
        """
        """

        # check _skip_name and default_skip_name set in a new instance
        grs = Groups()
        np_test.assert_equal(grs._skip_name, '_skip')
        np_test.assert_equal(grs.default_skip_name(), '_skip')

        # mimics the case when
        grs = Groups(pickle_debug=True)
        np_test.assert_equal(grs._skip_name, '_skip')
        np_test.assert_equal(grs.default_skip_name(), '_skip')
        
    def test_get_indexed_data(self):
        """
        Tests get_indexed_data()

        Problem: ids and vector converted to float
        """

        grs_1 = self.makeInstance()
        data = grs_1.indexed_data
        np_test.assert_equal(
            data.columns.tolist(), ['group', 'identifiers', 'ids', 'vector'])
        np_test.assert_equal(
            data.group.tolist(),
            8 * ['ga'] + 6 * ['gb'] + 5 * ['gd'])
        np_test.assert_equal(
            data.identifiers.unique().tolist(),
            ['ia1', 'ia3', 'ib1', 'ib3', 'id6'])
        np_test.assert_equal(
            data.ids.tolist(),
            ([0, 2] + list(range(6)) + [0, 2] + list(range(4))
             + list(range(1, 6))) )
        #print(data.ids)
        # Need to fix
        dtype = data.ids.dtype
        #np_test.assert_equal(np.issubdtype(dtype, int), True)
        np_test.assert_equal(
            data.vector.tolist(),
            ([1,3] + list(range(1,7)) + [1,3] + list(range(1,5))
             + list(range(2,7))))
        #print(data.vector)
        #np_test.assert_equal(np.issubdtype(data.vector.dtype, int), True)

        # no indexed data saved as None (shouldn't really happen)
        gr = self.makeInstance()
        gr.ga.setValue(identifier='ia5', name='ids', value=None)
        gr.ga.setValue(identifier='ia5', name='vector', value=None)
        idata = gr.indexed_data
        np_test.assert_equal((idata['identifiers'] == 'ia5').any(), False)

    def test_get_scalar_data(self):
        """
        Tests get_scalar_data()
        """

        grs_1 = self.makeInstance()
        data = grs_1.scalar_data
        np_test.assert_equal(
            data.columns.tolist(), ['group', 'identifiers', 'scalar'])
        np_test.assert_equal(
            data['group'].tolist(),
            3*['ga'] +  3*['gb'] + ['gd'])
        np_test.assert_equal(
            data['identifiers'].tolist(),
            ['ia1', 'ia3', 'ia5', 'ib1', 'ib2', 'ib3', 'id6'])
        np_test.assert_equal(
            data.scalar.tolist(), [2, 6, 10] +  [2, 4, 6] +  [12])

        # no indexed data saved as None (shouldn't really happen)
        gr = self.makeInstance()
        gr.ga.setValue(identifier='ia5', name='ids', value=None)
        gr.ga.setValue(identifier='ia5', name='vector', value=None)
        sdata = gr.scalar_data
        idata = gr.indexed_data
        np_test.assert_equal(len(sdata[sdata['identifiers'] == 'ia5']), 1)
        np_test.assert_equal(len(idata[idata['identifiers'] == 'ia5']), 0)

    def test_from_pandas(self):
        """
        Tests from_pandas()
        """

        # make tables
        indexed_table, scalar_table = self.make_tables()

        # check directly
        #print(indexed_table)
        groups = Groups.from_pandas(indexed=indexed_table, scalar=scalar_table)
        np_test.assert_equal(
            groups.alpha.properties,
            set(['identifiers', 'ids', 'vector_1', 'vector_2',
                 'scalar_1', 'scalar_2']))
        np_test.assert_equal(
            groups.bravo.properties,
            set(['identifiers', 'ids', 'vector_1', 'vector_2',
                 'scalar_1', 'scalar_2']))
        np_test.assert_equal(
            groups.alpha.indexed, set(['ids', 'vector_1', 'vector_2']))
        np_test.assert_equal(
            groups.bravo.indexed, set(['ids', 'vector_1', 'vector_2']))
        np_test.assert_equal(
            groups.alpha.identifiers, ['alpha_1', 'alpha_2', 'alpha_4'])
        np_test.assert_equal(
            groups.bravo.identifiers, ['bravo_1', 'bravo_2', 'bravo_3'])
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_1', name='ids'),
            range(5))
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_2', name='ids'),
            np.array([]))
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_4', name='ids'),
            np.arange(1,8,2))
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_1', name='ids'),
            np.arange(1,6))
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_2', name='ids'),
            np.arange(1,8,2))
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_3', name='ids'),
            np.arange(0,21,5))
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_1', name='vector_1'),
            [1, 1, 1, 3, 3])
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_2', name='vector_1'),
            np.array([]))
        np_test.assert_equal(
            groups.alpha.getValue(identifier='alpha_4', name='scalar_2'),
            12)
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_2', name='vector_1'),
            [3, 5, 7, 9])
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_3', name='vector_2'),
            [13, 13, 15, 17, 17])
        np_test.assert_equal(
            groups.bravo.getValue(identifier='bravo_1', name='scalar_1'),
            3)

        # convert back to pandas and check
        np_test.assert_equal(
            np.array((groups.indexed_data == indexed_table).all()).all(),
            True)
        np_test.assert_equal(
            np.array((groups.scalar_data == scalar_table).all()).all(),
            True)

        # test experiments without indexed properties
        gr = self.makeInstance()
        sdata = gr.scalar_data
        idata = gr.indexed_data
        gr_2 = Groups.from_pandas(scalar=sdata, indexed=idata)
        np_test.assert_equal(
            gr_2.ga.getValue(name='ids', identifier='ia5'), np.array([]))
        np_test.assert_equal(
            gr_2.ga.getValue(name='vector', identifier='ia5'), np.array([]))

    def testAddGroups(self):
        """
        Tests addGroups()
        """

        grs_1 = self.makeInstance()
        grs_2 = self.makeInstance2()
        grs_3 = pyto.analysis.Groups()
        grs_3.ga3 = grs_2.ga
        grs_3.gb3 = grs_2.gb

        # test exception
        np_test.assert_raises(ValueError, grs_1.addGroups, grs_2)

        # test added
        grs_1.addGroups(groups=grs_3)
        np_test.assert_equal(
            grs_1.ga.getValue(name='scalar', identifier='ia3'), 6)
        np_test.assert_equal(
            grs_1.gd.getValue(name='vector', identifier='id6'),
            np.arange(5) + 2)
        np_test.assert_equal(
            grs_1.ga3.getValue(name='ids', identifier='ia2'), [3,4,5])
        np_test.assert_equal(
            grs_1.ga3.getValue(name='xxx', identifier='ia3'),
            np.array([3.6, 4.6, 5.6, 6.6, 7.6]))
        np_test.assert_equal(
            grs_1.gb3.getValue(name='ids', identifier='ib1'), [1,3,4,5,6])
        np_test.assert_equal(
            grs_1.gb3.getValue(name='xxx', identifier='ib3'),
            np.array([4.7, 5.7, 6.7, 7.7, 8.7, 6.7]))

        # test copy
        grs_1 = self.makeInstance()
        grs_2 = self.makeInstance2()
        grs_3 = pyto.analysis.Groups()
        grs_3.ga3 = grs_2.ga
        grs_3.gb3 = grs_2.gb
        grs_1.addGroups(groups=grs_3, copy=True)
        np_test.assert_equal(
            grs_1.gb3.getValue(name='ids', identifier='ib2'), [2,4,6,8,10])
        grs_3.gb3.setValue(name='ids', identifier='ib2', value=[1,1,1,1,1])
        np_test.assert_equal(
            grs_1.gb3.getValue(name='ids', identifier='ib2'), [2,4,6,8,10])
        np_test.assert_equal(
            grs_3.gb3.getValue(name='ids', identifier='ib2'), [1,1,1,1,1])

        # test wo copy
        grs_1 = self.makeInstance()
        grs_2 = self.makeInstance2()
        grs_3 = pyto.analysis.Groups()
        grs_3.ga3 = grs_2.ga
        grs_3.gb3 = grs_2.gb
        grs_1.addGroups(groups=grs_3, copy=False)
        np_test.assert_equal(
            grs_1.gb3.getValue(name='ids', identifier='ib2'), [2,4,6,8,10])
        grs_3.gb3.setValue(name='ids', identifier='ib2', value=[1,1,1,1,1])
        np_test.assert_equal(
            grs_1.gb3.getValue(name='ids', identifier='ib2'), [1,1,1,1,1])
        np_test.assert_equal(
            grs_3.gb3.getValue(name='ids', identifier='ib2'), [1,1,1,1,1])

    def testExperiments(self):
        """
        Tests experiments() generator
        """

        groups_obj = self.makeInstance()

        group_names = set([])
        identifiers = set([])
        for group_name, identifier, experiment in groups_obj.experiments():

            group_names.add(group_name)
            identifiers.add(identifier)

            if group_name == 'ga':
                if identifier == 'ia1':
                    np_test.assert_equal(experiment.getValue(name='scalar'), 2)
                    np_test.assert_equal(experiment.getValue(name='vector'),
                                         2*np.arange(2) + 1)
                if identifier == 'ia3':
                    np_test.assert_equal(experiment.getValue(name='vector'),
                                         np.arange(6) + 1)
                if identifier == 'ia5':
                    np_test.assert_equal(experiment.getValue(name='scalar'), 10)
                    np_test.assert_equal(experiment.getValue(name='ids'),
                                         np.array([]))
            if group_name == 'gb':
                if identifier == 'ib3':
                    np_test.assert_equal(experiment.getValue(name='ids'),
                                         np.arange(4))

        # test if all groups and identifiers taken into account
        desired = set(groups_obj.keys())
        desired.remove('gc')
        np_test.assert_equal(group_names, desired)
        all_idents = set([])
        for group in list(groups_obj.values()):
            all_idents.update(set(group.identifiers))
        np_test.assert_equal(identifiers, all_idents)

        # test argument groups
        group_names = set([])
        identifiers = set([])
        for group_name, identifier, experiment in groups_obj.experiments(
            categories=['ga','gc']):
            group_names.add(group_name)
            identifiers.add(identifier)
        np_test.assert_equal(group_names, set(['ga']))

        # test argument identifiers
        group_names = set([])
        identifiers = []
        for group_name, identifier, experiment in groups_obj.experiments(
            identifiers=['ib1','ib3', 'id6']):
            group_names.add(group_name)
            identifiers.append(identifier)
        np_test.assert_equal(group_names, set(['gb', 'gd']))
        np_test.assert_equal(identifiers, ['ib1','ib3', 'id6'])

        # test argument identifiers with different order
        group_names = set([])
        identifiers = []
        for group_name, identifier, experiment in groups_obj.experiments(
            identifiers=['ib3','id6', 'ib1']):
            group_names.add(group_name)
            identifiers.append(identifier)
        np_test.assert_equal(group_names, set(['gb', 'gd']))
        np_test.assert_equal(identifiers, ['ib3','id6', 'ib1'])

    def testRegroup(self):
        """
        Tests regroup()
        """

        # make instance and add a property to be used as a new category
        groups = self.makeInstance()
        groups.ga.setValue(identifier='ia1', name='color', value='red')
        groups.ga.setValue(identifier='ia3', name='color', value='green')
        groups.ga.setValue(identifier='ia5', name='color', value='blue')
        groups.gb.setValue(identifier='ib1', name='color', value='red')
        groups.gb.setValue(identifier='ib2', name='color', value='green')
        groups.gb.setValue(identifier='ib3', name='color', value='blue')
        groups.gd.setValue(identifier='id6', name='color', value='red')

        # regroup by color
        new = groups.regroup(name='color')
        np_test.assert_equal(set(new.keys()), set(['red', 'blue', 'green']))
        np_test.assert_equal(set(new.red.identifiers),
                             set(['ia1', 'ib1', 'id6']))
        np_test.assert_equal(set(new.blue.identifiers), set(['ia5', 'ib3']))
        np_test.assert_equal(set(new.green.identifiers), set(['ia3', 'ib2']))

        # regroup by color with identifiers
        new = groups.regroup(name='color',
                             identifiers=['id6', 'ia5', 'ia1', 'ib2', 'ia3'])
        np_test.assert_equal(set(new.keys()), set(['red', 'blue', 'green']))
        np_test.assert_equal(new.red.identifiers, ['id6', 'ia1'])
        np_test.assert_equal(new.blue.identifiers, ['ia5'])
        np_test.assert_equal(new.green.identifiers, ['ib2', 'ia3'])
        np_test.assert_equal(new['red'].getValue(
                identifier='id6', name='vector'), np.arange(5) + 2)
        np_test.assert_equal(new['red'].getValue(
                identifier='ia1', name='ids'), 2*np.arange(2))
        np_test.assert_equal(new['blue'].getValue(identifier='ia5', name='ids'),
                             np.array([]))
        np_test.assert_equal(new['green'].getValue(
                identifier='ib2', name='scalar'), 4)
        np_test.assert_equal(new['green'].getValue(
                identifier='ia3', name='vector'), np.arange(6) + 1)

        # regroup by a non-string value
        new = groups.regroup(name='scalar')
        np_test.assert_equal(set(new.keys()),
                             set(['2', '4', '6', '10', '12']))
        np_test.assert_equal(set(new['2'].identifiers), set(['ia1', 'ib1']))
        np_test.assert_equal(set(new['4'].identifiers), set(['ib2']))
        np_test.assert_equal(set(new['6'].identifiers), set(['ia3', 'ib3']))
        np_test.assert_equal(set(new['10'].identifiers), set(['ia5']))
        np_test.assert_equal(set(new['12'].identifiers), set(['id6']))
        np_test.assert_equal(new['2'].getValue(identifier='ia1', name='vector'),
                             2*np.arange(2) + 1)
        np_test.assert_equal(new['4'].getValue(identifier='ib2', name='vector'),
                             np.array([]))
        np_test.assert_equal(new['6'].getValue(identifier='ib3', name='vector'),
                             np.arange(4) + 1)
        np_test.assert_equal(new['6'].getValue(identifier='ia3', name='ids'),
                             np.arange(6))
        np_test.assert_equal(new['10'].getValue(identifier='ia5',
                                                name='scalar'), 10)
        np_test.assert_equal(new['12'].getValue(identifier='id6', name='ids'),
                             np.arange(5) + 1)

    def testApply(self):
        """
        Test apply()
        """

        groups = self.makeSymmetricalInstance()
        def plus(x, y): return x + y

        # all groups
        groups.apply(funct=plus, args=['vector'], kwargs={'y':10}, name='new')
        np_test.assert_equal(groups.ga.new, [[11,13], [11,12,13,14,15,16], []])
        np_test.assert_equal(groups.gb.new, [[11,13], [], [11,12,13,14]])
        np_test.assert_equal('new' in groups.ga.properties, True)
        np_test.assert_equal('new' in groups.gb.properties, True)

        # one group, more args
        groups.apply(funct=np.subtract, args=['new', 'vector'], name='new2',
                     categories=['gb'])
        np_test.assert_equal(getattr(groups.ga, 'new2', None) is None, True)
        np_test.assert_equal(groups.gb.new2, [[10,10], [], [10,10,10,10]])
        np_test.assert_equal('new2' in groups.ga.properties, False)
        np_test.assert_equal('new2' in groups.gb.properties, True)

    def testAddData(self):
        """
        Tests addData()
        """

        # make two instances
        groups = self.makeInstance()
        groups_2 = self.makeInstance()
        def plus(x, y): return x + y
        groups_2.apply(funct=plus, args=['vector'], kwargs={'y':10},
                       name='vector_2')

        groups.addData(source=groups_2, names=['vector_2'])
        for g_name in list(groups.keys()):
            np_test.assert_equal('vector_2' in groups[g_name].properties, True)
            np_test.assert_equal('vector_2' in groups[g_name].indexed, True)
            for ident in groups[g_name].identifiers:
                np_test.assert_equal(
                    groups[g_name].getValue(property='vector_2',
                                            identifier=ident),
                    groups_2[g_name].getValue(property='vector_2',
                                              identifier=ident))

        # change names
        groups.addData(source=groups_2, names={'vector_2' : 'vector_3'})
        for g_name in list(groups.keys()):
            np_test.assert_equal('vector_3' in groups[g_name].properties, True)
            np_test.assert_equal('vector_3' in groups[g_name].indexed, True)
            for ident in groups[g_name].identifiers:
                np_test.assert_equal(
                    groups[g_name].getValue(property='vector_3',
                                            identifier=ident),
                    groups_2[g_name].getValue(property='vector_2',
                                              identifier=ident))

    def testRemove(self):
        """
        Tests remove()
        """

        # non-repeating identifiers, groups not specified
        groups_obj = self.makeInstance()
        groups_obj.remove(identifiers=['ia1', 'ib2'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia3', 'ia5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib1', 'ib3'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeInstance()
        groups_obj.remove(identifiers=['ia1', 'ib2'], groups=['ga', 'gb'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia3', 'ia5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib1', 'ib3'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeInstance()
        groups_obj.remove(identifiers=['ia1', 'ib1', 'ib2'], groups=['gb'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia1', 'ia3', 'ia5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib3'])

        # repeating identifiers, groups not specified
        groups_obj = self.makeSymmetricalInstance()
        groups_obj.remove(identifiers=['i1', 'i5'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['i3'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['i3'])

        # repeating identifiers, groups specified
        groups_obj = self.makeSymmetricalInstance()
        groups_obj.remove(identifiers=['i1', 'i5'], groups=['ga'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['i3'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['i1', 'i3', 'i5'])

    def testKeep(self):
        """
        Tests keep()
        """

        # non-repeating identifiers, groups not specified
        groups_obj = self.makeInstance()
        groups_obj.keep(identifiers=['ia1', 'ib2', 'ib3'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia1'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib2', 'ib3'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeInstance()
        groups_obj.keep(identifiers=['ia1', 'ib2', 'ib3'], groups=['ga', 'gb'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia1'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib2', 'ib3'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeInstance()
        groups_obj.keep(identifiers=['ia1', 'ib2', 'ib3'], groups=['gb'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['ia1', 'ia3', 'ia5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib2', 'ib3'])

        # non-repeating identifiers, groups specified, remove groups
        groups_obj = self.makeInstance()
        groups_obj.keep(identifiers=['ia1', 'ib2', 'ib3'], groups=['gb'],
                        removeGroups=True)
        np_test.assert_equal('ga' in groups_obj, False)
        np_test.assert_equal('gb' in groups_obj, True)
        np_test.assert_equal(groups_obj.gb.identifiers, ['ib2', 'ib3'])

        # non-repeating identifiers, groups not specified
        groups_obj = self.makeSymmetricalInstance()
        groups_obj.keep(identifiers=['i1', 'i5'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['i1', 'i5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['i1', 'i5'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeSymmetricalInstance()
        groups_obj.keep(identifiers=['i5'], groups=['gb'])
        np_test.assert_equal(groups_obj.ga.identifiers, ['i1', 'i3', 'i5'])
        np_test.assert_equal(groups_obj.gb.identifiers, ['i5'])

        # non-repeating identifiers, groups specified
        groups_obj = self.makeSymmetricalInstance()
        groups_obj.keep(identifiers=['i5'], groups=['gb'], removeGroups=True)
        np_test.assert_equal('ga' in groups_obj, False)
        np_test.assert_equal('gb' in groups_obj, True)
        np_test.assert_equal(groups_obj.gb.identifiers, ['i5'])

    def testJoinExperiments(self):
        """
        Tests joinExperiments()
        """

        # mode join, scalar (not indexed), all groups
        obs = self.groups.joinExperiments(name='scalar', mode='join')
        np_test.assert_equal(obs.properties,
                             set(['scalar', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['scalar', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'scalar'),
                             np.array([2, 6, 10]))
        np_test.assert_equal(obs.getValue('gb', 'scalar'),
                             np.array([2, 4, 6]))
        np_test.assert_equal(obs.getValue('gc', 'scalar'),
                             np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'scalar'), np.array([12]))
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.array([1,2,3]))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.array([1,2,3]))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.array([1]))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                             ['ia1','ia3','ia5'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                             ['ib1','ib2','ib3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'), ['id6'])

        # mode join, scalar (not indexed), selected groups
        obs = self.groups.joinExperiments(name='scalar', mode='join',
                                          groups=['ga', 'gd'])
        np_test.assert_equal(obs.properties,
                             set(['scalar', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['scalar', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'scalar'),
                             np.array([2, 6, 10]))
        np_test.assert_equal(obs.getValue('gd', 'scalar'), np.array([12]))

        # mode join, scalar (not indexed), selected groups including
        # placeholder group
        some_groups = ['ga', self.groups._skip_name, 'gd']
        desired_groups = ['ga', 'gd']
        obs = self.groups.joinExperiments(
            name='scalar', mode='join', groups=some_groups)
        np_test.assert_equal(obs.properties,
                             set(['scalar', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['scalar', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers), set(desired_groups))
        np_test.assert_equal(obs.getValue('ga', 'scalar'),
                             np.array([2, 6, 10]))
        np_test.assert_equal(obs.getValue('gd', 'scalar'), np.array([12]))

        # mode join, indexed variable, all groups
        obs = self.groups.joinExperiments(name='vector', mode='join')
        np_test.assert_equal(obs.properties,
                             set(['vector', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['vector', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(
            obs.getValue('ga', 'vector'),
            np.append(2*np.arange(2) + 1, np.arange(6) + 1))
        np_test.assert_equal(
            obs.getValue('gb', 'vector'),
            np.append(2*np.arange(2) + 1, np.arange(4) + 1))
        np_test.assert_equal(obs.getValue('gc', 'vector'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'vector'),
                             np.arange(5) + 2)
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.arange(1,9))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.arange(1,7))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.arange(1,6))
        np_test.assert_equal(
            obs.getValue('ga', 'idNames'),
            ['ia1_0', 'ia1_2',
             'ia3_0', 'ia3_1', 'ia3_2', 'ia3_3', 'ia3_4', 'ia3_5'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                       ['ib1_0', 'ib1_2', 'ib3_0', 'ib3_1', 'ib3_2', 'ib3_3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'),
                             ['id6_1', 'id6_2', 'id6_3', 'id6_4', 'id6_5'])

        # mode join, indexed variable, selected identifiers
        some_idents = ['ia1', 'ia5', 'ib3', 'id6']
        obs = self.groups.joinExperiments(name='vector', mode='join',
                                          identifiers=some_idents)
        np_test.assert_equal(obs.properties,
                             set(['vector', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['vector', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'vector'),
                             2*np.arange(2) + 1)
        np_test.assert_equal(obs.getValue('gb', 'vector'),
                             np.arange(4) + 1)
        np_test.assert_equal(obs.getValue('gc', 'vector'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'vector'),
                             np.arange(5) + 2)
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.arange(1,3))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.arange(1,5))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.arange(1,6))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                              ['ia1_0', 'ia1_2'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                       ['ib3_0', 'ib3_1', 'ib3_2', 'ib3_3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'),
                             ['id6_1', 'id6_2', 'id6_3', 'id6_4', 'id6_5'])

        # mode join, >1 variable, selected identifiers
        some_idents = ['ia1', 'ia5', 'ib3', 'id6']
        obs = self.groups.joinExperiments(name=['vector', 'scalar'],
                                          mode='join', identifiers=some_idents)
        np_test.assert_equal(obs.properties,
                             set(['vector', 'ids', 'idNames', 'identifiers',
                                  'scalar']))
        np_test.assert_equal(obs.indexed,
                             set(['vector', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'vector'),
                             2*np.arange(2) + 1)
        np_test.assert_equal(obs.getValue('gb', 'vector'),
                             np.arange(4) + 1)
        np_test.assert_equal(obs.getValue('gc', 'vector'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'vector'),
                             np.arange(5) + 2)
        np_test.assert_equal(obs.getValue('ga', 'scalar'), [2, 10])
        np_test.assert_equal(obs.getValue('gb', 'scalar'), [6])
        np_test.assert_equal(obs.getValue('gc', 'scalar'), [])
        np_test.assert_equal(obs.getValue('gd', 'scalar'), [12])
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.arange(1,3))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.arange(1,5))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.arange(1,6))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                              ['ia1_0', 'ia1_2'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                       ['ib3_0', 'ib3_1', 'ib3_2', 'ib3_3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'),
                             ['id6_1', 'id6_2', 'id6_3', 'id6_4', 'id6_5'])

        # mode mean, not indexed
        obs = self.groups.joinExperiments(name='scalar', mode='mean')
        np_test.assert_equal(obs.properties,
                             set(['scalar', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['scalar', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'scalar'), [2, 6, 10])
        np_test.assert_equal(obs.getValue('gb', 'scalar'), [2, 4, 6])
        np_test.assert_equal(obs.getValue('gc', 'scalar'), [])
        np_test.assert_equal(obs.getValue('gd', 'scalar'), [12])

        # mode mean, indexed
        obs = self.groups.joinExperiments(name='vector', mode='mean')
        np_test.assert_equal(obs.properties,
                             set(['vector', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['vector', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'vector'), [2, 3.5])
        np_test.assert_equal(obs.getValue('gb', 'vector'), [2, 2.5])
        np_test.assert_equal(obs.getValue('gc', 'vector'), [])
        np_test.assert_equal(obs.getValue('gd', 'vector'), [4])
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.array([1,2]))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.array([1,2]))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.array([1]))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                             ['ia1','ia3'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                             ['ib1','ib3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'), ['id6'])

        # mode mean, indexed, removeEmpty=False
        obs = self.groups.joinExperiments(name='vector', mode='mean',
                                          removeEmpty=False)
        np_test.assert_equal(obs.properties,
                             set(['vector', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(obs.indexed,
                             set(['vector', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'vector'), [2, 3.5, np.NaN])
        np_test.assert_equal(obs.getValue('gb', 'vector'), [2, np.NaN, 2.5])
        np_test.assert_equal(obs.getValue('gc', 'vector'), [])
        np_test.assert_equal(obs.getValue('gd', 'vector'), [4])
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.array([1,2, 3]))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.array([1,2, 3]))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.array([1]))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                             ['ia1','ia3','ia5'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                             ['ib1','ib2','ib3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'), ['id6'])

        # mode mean_bin, indexed
        obs = self.groups.joinExperiments(
            name='vector', mode='mean_bin', bins=[0,2,5,10], fraction=1,
            fraction_name='fraction')
        np_test.assert_equal(
            obs.properties,
            set(['fraction', 'ids', 'idNames', 'identifiers']))
        np_test.assert_equal(
            obs.indexed, set(['fraction', 'ids', 'idNames']))
        np_test.assert_equal(set(obs.identifiers),
                             set(['ga', 'gb', 'gc', 'gd']))
        np_test.assert_equal(obs.getValue('ga', 'fraction'), [0.5, 0.5])
        np_test.assert_equal(obs.getValue('gb', 'fraction'), [0.5, 0.75])
        np_test.assert_equal(obs.getValue('gc', 'fraction'), [])
        np_test.assert_equal(obs.getValue('gd', 'fraction'), [0.6])
        np_test.assert_equal(obs.getValue('ga', 'ids'), np.array([1,2]))
        np_test.assert_equal(obs.getValue('gb', 'ids'), np.array([1,2]))
        np_test.assert_equal(obs.getValue('gc', 'ids'), np.array([]))
        np_test.assert_equal(obs.getValue('gd', 'ids'), np.array([1]))
        np_test.assert_equal(obs.getValue('ga', 'idNames'),
                             ['ia1','ia3'])
        np_test.assert_equal(obs.getValue('gb', 'idNames'),
                             ['ib1','ib3'])
        np_test.assert_equal(obs.getValue('gc', 'idNames'), [])
        np_test.assert_equal(obs.getValue('gd', 'idNames'), ['id6'])

        # another group, modes 'join', 'mean' and 'mean_bin'
        obs = self.groups2.joinExperiments(name='xxx', mode='join')
        desired = ([1.1, 2.1, 3.1, 4.1, 5.1] + [0.1, 1.1, 2.1]
                   + [3.6, 4.6, 5.6, 6.6, 7.6])
        np_test.assert_equal(
            obs.getValue(identifier='ga', name='xxx'), desired)
        obs = self.groups2.joinExperiments(name='xxx', mode='mean')
        np_test.assert_equal(
            obs.getValue(identifier='ga', name='xxx'), [3.1, 1.1, 5.6])
        np_test.assert_equal(
            obs.getValue(
                identifier='gb', name='xxx'), [3.3, 5.3, 6.7])
        obs = self.groups2.joinExperiments(
            name='xxx', mode='mean_bin', bins=[0,2,4,6], fraction=1,
            fraction_name='fract')
        np_test.assert_equal(
            obs.getValue(identifier='ga', name='fract'), [2/5., 1/3., 1/5.])
        np_test.assert_equal(
            obs.getValue(identifier='gb', name='fract'), [2/5., 1/5., 0])
        obs = self.groups2.joinExperiments(
            name='xxx', mode='mean_bin', bins=[0,2,4,6], fraction=0,
            fraction_name='frac', removeEmpty=False)
        np_test.assert_equal(
            obs.getValue(identifier='ga', name='frac'), [1/5., 2/3., 0.])
        np_test.assert_equal(
            obs.getValue(identifier='gb', name='frac'),
            [1/5., 0., 0, np.NaN])
        np_test.assert_equal(
            obs.getValue('ga', 'idNames'), ['ia1','ia2', 'ia3'])
        np_test.assert_equal(
            obs.getValue('gb', 'idNames'), ['ib1','ib2', 'ib3', 'ib4'])

    def testJoinAndStats(self):
        """
        Tests joinAndStats()
        """

        # mode join, scalar
        stats = self.groups.joinAndStats(name='scalar', reference='ga',
                                         mode='join', test='t', out=out)
        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'data', 'mean', 'std', 'n',
                 'sem', 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(stats.testSymbol, ['t'] * 4),
        np_test.assert_equal(stats.getValue('ga', 'data'), [2, 6, 10])
        np_test.assert_equal(stats.getValue('ga', 'mean'), 6)
        np_test.assert_almost_equal(stats.getValue('ga', 'std'), 4)
        np_test.assert_equal(stats.getValue('ga', 'n'), 3)
        np_test.assert_almost_equal(stats.getValue('ga', 'testValue'), 0)
        np_test.assert_almost_equal(stats.getValue('ga', 'confidence'), 1)
        np_test.assert_equal(stats.getValue('gb', 'data'), [2, 4, 6])
        np_test.assert_equal(stats.getValue('gb', 'mean'), 4)
        np_test.assert_almost_equal(stats.getValue('gb', 'std'), 2)
        np_test.assert_equal(stats.getValue('gb', 'n'), 3)
        np_test.assert_almost_equal(
            stats.getValue('gb', 'testValue'),
            scipy.stats.ttest_ind([2,4,6], [2,6,10])[0])
        np_test.assert_almost_equal(
            stats.getValue('gb', 'confidence'),
            scipy.stats.ttest_ind([2,4,6], [2,6,10])[1])
        np_test.assert_equal(stats.getValue('gc', 'data'), [])
        np_test.assert_equal(np.isnan(stats.getValue('gc', 'mean')), True)
        try:
            np_test.assert_almost_equal(stats.getValue('gc', 'std'), 0)
        except AssertionError:
            np_test.assert_almost_equal(stats.getValue('gc', 'std'), np.nan)
        np_test.assert_equal(stats.getValue('gc', 'n'), 0)
        np_test.assert_equal(
            np.isnan(stats.getValue('gc', 'testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.getValue('gc', 'confidence')), True)
        np_test.assert_equal(stats.getValue('gd', 'data'), [12])
        np_test.assert_equal(stats.getValue('gd', 'mean'), 12)
        try:
            np_test.assert_almost_equal(stats.getValue('gd', 'std'), 0)
        except AssertionError:
            np_test.assert_almost_equal(stats.getValue('gd', 'std'), np.nan)
        np_test.assert_equal(stats.getValue('gd', 'n'), 1)
        np_test.assert_almost_equal(
            stats.getValue('gd', 'testValue'),
            scipy.stats.ttest_ind([12], [2,6,10])[0])
        np_test.assert_almost_equal(
            stats.getValue('gd', 'confidence'),
            scipy.stats.ttest_ind([12], [2,6,10])[1])

        # mode join, vector
        stats = self.groups.joinAndStats(name='vector', reference='ga',
                                    mode='join', test='mannwhitney', out=out)
        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(stats.testSymbol, ['u'] * 4)
        np_test.assert_equal(stats.getValue('ga', 'data'), [1,3,1,2,3,4,5,6])
        np_test.assert_equal(stats.getValue('gb', 'data'), [1,3,1,2,3,4])
        np_test.assert_equal(stats.getValue('gc', 'data'), [])
        np_test.assert_equal(stats.getValue('gd', 'data'), [2,3,4,5,6])

        # mode join, vector, groups
        some_groups = ['ga', 'gd']
        stats = self.groups.joinAndStats(
            name='vector', reference='ga', groups=some_groups, mode='join',
            test='mannwhitney', out=out)
        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(set(stats.identifiers), set(some_groups))
        np_test.assert_equal(stats.testSymbol, ['u'] * len(some_groups))
        np_test.assert_equal(stats.getValue('ga', 'data'), [1,3,1,2,3,4,5,6])
        np_test.assert_equal(stats.getValue('gd', 'data'), [2,3,4,5,6])

        # mode join, vector, some groups including a plcaholder group
        some_groups = ['ga', self.groups._skip_name, 'gd']
        desired_groups = ['ga', 'gd']
        stats = self.groups.joinAndStats(
            name='vector', reference='ga', groups=some_groups, mode='join',
            test='mannwhitney', out=out)
        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(set(stats.identifiers), set(desired_groups))
        np_test.assert_equal(stats.testSymbol, ['u'] * len(desired_groups))
        np_test.assert_equal(stats.getValue('ga', 'data'), [1,3,1,2,3,4,5,6])
        np_test.assert_equal(stats.getValue('gd', 'data'), [2,3,4,5,6])

        # mode mean, vector
        stats = self.groups.joinAndStats(name='vector', reference='ga',
                                         mode='mean', test='kruskal', out=out)
        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'data', 'mean', 'std', 'n',
                 'sem', 'testValue', 'confidence', 'testSymbol', 'reference']))
        np_test.assert_equal(stats.testSymbol, ['h'] * 4)
        np_test.assert_equal(stats.getValue('ga', 'data'), [2, 3.5])
        np_test.assert_equal(stats.getValue('gb', 'data'), [2, 2.5])
        np_test.assert_equal(stats.getValue('gc', 'data'), [])
        np_test.assert_equal(stats.getValue('gd', 'data'), [4])

        # mode byIndex
        group = self.makeSameIdInstance()
        stats = group.joinAndStats(name='vector', mode='byIndex', out=out)

        np_test.assert_equal(
            stats.properties,
            set(['identifiers', 'ids', 'data', 'mean', 'std', 'sem', 'n']))
        np_test.assert_equal(stats.indexed, set(['ids', 'mean', 'std', 'sem']))

        np_test.assert_equal(stats.getValue('ga', 'ids'), [1, 3, 5])
        np_test.assert_almost_equal(stats.getValue('ga', 'mean'),
                                    [3, 5, 7])
        np_test.assert_almost_equal(stats.getValue('ga', 'std'), [1, 1, 1])
        np_test.assert_almost_equal(stats.getValue('ga', 'sem'),
                                    3 * [1./np.sqrt(3)])
        np_test.assert_almost_equal(stats.getValue('ga', 'n'), 3)

        np_test.assert_equal(stats.getValue('gb', 'ids'), [1, 2, 3])
        np_test.assert_almost_equal(stats.getValue('gb', 'mean'),
                                    [15, 30, 45])
        np_test.assert_almost_equal(
                stats.getValue('gb', 'std'),
                [np.sqrt(50), np.sqrt(200), np.sqrt(450)])
        np_test.assert_almost_equal(
                stats.getValue('gb', 'sem'),
                [np.sqrt(50/2.), np.sqrt(100), np.sqrt(450/2.)])
        np_test.assert_almost_equal(stats.getValue('gb', 'n'), 2)

        # another group, join no bins
        stats = self.groups2.joinAndStats(
            name='xxx', mode='join', out=out, test='t', reference='ga')
        desired_ga = [1.1, 2.1, 3.1, 4.1, 5.1, 0.1, 1.1, 2.1, 3.6, 4.6, 5.6,
                      6.6, 7.6]
        desired_gb = [1.3, 2.3, 3.3, 4.3, 5.3, 3.3, 4.3, 5.3, 6.3, 7.3,
                      4.7, 5.7, 6.7, 7.7, 8.7, 6.7]
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='data'), desired_ga)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), desired_gb)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='mean'), 46.8 / 13.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='mean'), 83.2 / 16.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='n'), 13)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='std'),
            np.std(desired_ga, ddof=1))
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='sem'),
            np.std(desired_ga, ddof=1) / np.sqrt(13))
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='testValue'),
            scipy.stats.ttest_ind(desired_gb, desired_ga)[0])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='confidence'), 1)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='confidence'),
            scipy.stats.ttest_ind(desired_ga, desired_gb)[1])
        np_test.assert_equal(
            stats.getValue(identifier='gb', name='testSymbol'), 't')

        # another group, mean no bins
        stats = self.groups2.joinAndStats(name='xxx', mode='mean', out=out)
        desired = [3.3, 5.3, 6.7]
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='data'), [3.1, 1.1, 5.6])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='mean'), 9.8 / 3.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), desired)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='mean'), 15.3 / 3.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='n'), 3)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='std'),
            np.std(desired, ddof=1))
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='sem'),
            np.std(desired, ddof=1) / np.sqrt(3))

        # another group, join + bins
        stats = self.groups2.joinAndStats(
            name='xxx', mode='join', bins=[0,2,4,10], fraction=1, out=out,
            test='chi2', reference='gb')
        data_gb = [1.3, 2.3, 3.3, 4.3, 5.3, 3.3, 4.3, 5.3, 6.3, 7.3, 4.7,
             5.7, 6.7, 7.7, 8.7, 6.7]
        histogram_ga = [3, 4, 6]
        histogram_gb = [1, 3, 12]
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='histogram'), histogram_ga)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='fraction'), 4 / 13.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), data_gb)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='histogram'), histogram_gb)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='probability'),
            [1/16., 3/16., 12/16.])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='fraction'), 3 / 16.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='ids'), [0,2,4])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='n'), 16)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='testValue'),
            pyto.util.scipy_plus.chisquare_2(histogram_ga, histogram_gb)[0])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='confidence'),
            pyto.util.scipy_plus.chisquare_2(histogram_ga, histogram_gb)[1])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='confidence'), 1.)

        # another group, mean + bins
        stats = self.groups2.joinAndStats(
            name='xxx', mode='mean', bins=[0,2,4,10], fraction=2, out=out)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), [3.3, 5.3, 6.7])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='histogram'), [1, 1, 1])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='fraction'), 1/3.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='histogram'), [0, 1, 2])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='fraction'), 2/3.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='probability'),
            [0., 1/3., 2/3.])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='ids'), [0,2,4])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='n'), 3)

        # another group, mean_bin, frac 0
        stats = self.groups2.joinAndStats(
            name='xxx', mode='mean_bin', bins=[0,2,4,10], fraction=0, out=out,
            test='t', reference='ga')
        desired_ga = [0.2, 2/3., 0]
        desired_gb = [0.2, 0, 0]
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='data'), desired_ga)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='mean'), 13/45.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), desired_gb)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='mean'), 0.2/3.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='n'), 3)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='std'),
            np.std(desired_ga, ddof=1))
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='sem'),
            np.std(desired_ga, ddof=1) / np.sqrt(3))
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='testValue'),
            scipy.stats.ttest_ind(desired_gb, desired_ga)[0])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='confidence'),
            scipy.stats.ttest_ind(desired_gb, desired_ga)[1])


        # another group, mean_bin, frac 1
        stats = self.groups2.joinAndStats(
            name='xxx', mode='mean_bin', bins=[0,2,4,10], fraction=1, out=out)
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='data'), [0.4, 1/3., 0.2])
        np_test.assert_almost_equal(
            stats.getValue(identifier='ga', name='mean'), 14/45.)
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='data'), [0.4, 0.2, 0])
        np_test.assert_almost_equal(
            stats.getValue(identifier='gb', name='mean'), 0.6/3.)

    def testIsTransposable(self):
        """
        Tests isTransposable()
        """

        nonsym = self.makeInstance()
        np_test.assert_equal(nonsym.isTransposable(), False)
        sym = self.makeSymmetricalInstance()
        np_test.assert_equal(sym.isTransposable(), True)

    def testTranspose(self):
        """
        Tests transpose()
        """

        # check raising exception when not symmetrical
        #np_test.assert_raises(ValueError, self.groups.transpose())

        # make instance and add references
        sym = self.makeSymmetricalInstance()
        for group_name, group in list(sym.items()):
            ref_g_a = ['gb', 'ga', 'ga']
            ref_g_b = ['gb', 'gb', 'ga']
            if group_name == 'ga':
                reference_g = ref_g_a
            elif group_name == 'gb':
                reference_g = ref_g_b
            for ident, ref, ref_g in zip(['i1', 'i3', 'i5'],
                                         ['i3', 'i1', 'i1'], reference_g):
                group.setValue(identifier=ident, property='reference',
                               value=ref)
                group.setValue(identifier=ident, property='referenceGroup',
                               value=ref_g)

        # transpose
        transp = sym.transpose()

        # check group names, identifiers, properties and indexed
        np_test.assert_equal(set(transp.keys()), set(sym['gb'].identifiers))
        for t_name, t_group in list(transp.items()):
            np_test.assert_equal(t_group.properties,
                              set(['scalar', 'ids', 'vector', 'identifiers',
                                   'reference', 'referenceGroup']))
            np_test.assert_equal(t_group.indexed, set(['ids', 'vector']))
            np_test.assert_equal(t_group.identifiers, list(sym.keys()))

        # check scalar
        np_test.assert_equal(
            transp.i1.getValue(identifier='ga', property='scalar'), 2)
        np_test.assert_equal(
            transp.i1.getValue(identifier='gb', property='scalar'), 2)
        np_test.assert_equal(
            transp.i3.getValue(identifier='ga', property='scalar'), 6)
        np_test.assert_equal(
            transp.i3.getValue(identifier='gb', property='scalar'), 4)
        np_test.assert_equal(
            transp.i5.getValue(identifier='ga', property='scalar'), 10)
        np_test.assert_equal(
            transp.i5.getValue(identifier='gb', property='scalar'), 6)

        # check vector
        np_test.assert_equal(
            transp.i1.getValue(identifier='ga', property='vector'),
            2*np.arange(2) + 1)
        np_test.assert_equal(
            transp.i1.getValue(identifier='gb', property='vector'),
            2*np.arange(2) + 1)
        np_test.assert_equal(
            transp.i3.getValue(identifier='ga', property='vector'),
            np.arange(6) + 1)
        np_test.assert_equal(
            transp.i3.getValue(identifier='gb', property='vector'), [])
        np_test.assert_equal(
            transp.i5.getValue(identifier='ga', property='vector'), [])
        np_test.assert_equal(
            transp.i5.getValue(identifier='gb', property='vector'),
            np.arange(4) + 1)

        # check reference
        np_test.assert_equal(
            transp.i1.getValue(identifier='ga', property='reference'), 'gb')
        np_test.assert_equal(
            transp.i1.getValue(identifier='ga', property='referenceGroup'),
            'i3')
        np_test.assert_equal(
            transp.i3.getValue(identifier='ga', property='reference'), 'ga')
        np_test.assert_equal(
            transp.i3.getValue(identifier='ga', property='referenceGroup'),
            'i1')
        np_test.assert_equal(
            transp.i5.getValue(identifier='ga', property='reference'), 'ga')
        np_test.assert_equal(
            transp.i5.getValue(identifier='ga', property='referenceGroup'),
            'i1')
        np_test.assert_equal(
            transp.i1.getValue(identifier='gb', property='reference'), 'gb')
        np_test.assert_equal(
            transp.i1.getValue(identifier='gb', property='referenceGroup'),
            'i3')
        np_test.assert_equal(
            transp.i3.getValue(identifier='gb', property='reference'), 'gb')
        np_test.assert_equal(
            transp.i3.getValue(identifier='gb', property='referenceGroup'),
            'i1')
        np_test.assert_equal(
            transp.i5.getValue(identifier='gb', property='reference'), 'ga')
        np_test.assert_equal(
            transp.i5.getValue(identifier='gb', property='referenceGroup'),
            'i1')

        # transpose of transpose
        sym = self.makeSymmetricalInstance()
        transp = sym.transpose()
        tt = transp.transpose()
        np_test.assert_equal(set(tt.keys()), set(sym.keys()))
        for tt_name, tt_group in list(tt.items()):
            np_test.assert_equal(tt_group.properties, sym[tt_name].properties)
            np_test.assert_equal(tt_group.indexed, sym[tt_name].indexed)
            np_test.assert_equal(tt_group.identifiers, sym[tt_name].identifiers)
            np_test.assert_equal(tt_group.scalar, sym[tt_name].scalar)
            np_test.assert_equal(tt_group.vector, sym[tt_name].vector)
            np_test.assert_equal(tt_group.ids, sym[tt_name].ids)

    def testDoStatsBetweenExperiments(self):
        """
        Tests doStats(between='experiments')
        """

        # symmetrical instance
        sym = self.makeSymmetricalInstance()

        # between 'experiments', string reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference='i3', out=out)
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                     'testValue', 'confidence', 'testSymbol', 'reference',
                     'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['t'] * 3)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='mean'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='std'), np.sqrt(2))
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='n'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'), 0)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'), 1)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [1,3])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i1',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i1',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='referenceGroup'), 'gb')

        # one group without identifiers
        sym2 = self.makeSymmetricalInstance()
        sym2['gc'] = Observations()
        stats = sym2.doStats(name='vector', test='t', between='experiments',
                            reference='i3', out=out)
        np_test.assert_equal(stats.gc.identifiers, [])
        np_test.assert_equal(stats.gc.data, [])
        np_test.assert_equal(stats.gc.mean, [])

        # between 'experiments', groups and identifiers, string reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference='i3', out=out,
                            groups=['ga'], identifiers=['i3', 'i5'])
        np_test.assert_equal(list(stats.keys()), ['ga'])
        np_test.assert_equal(stats.ga.identifiers, ['i3', 'i5'])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])

        # between 'experiments', identifiers, string reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference='i3', out=out,
                            groups=['ga', 'gb'], identifiers=['i3', 'i5'])
        np_test.assert_equal(set(stats.keys()), set(['ga', 'gb']))
        for stats_group in list(stats.values()):
            np_test.assert_equal(stats_group.identifiers, ['i3', 'i5'])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])

        # between 'experiments', string reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference='i3', groups=['ga'], out=out)
        np_test.assert_equal(list(stats.keys()), ['ga'])
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                     'testValue', 'confidence', 'testSymbol', 'reference',
                     'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['t'] * 3)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='mean'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='std'), np.sqrt(2))
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='n'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'), 0)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'), 1)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='referenceGroup'), 'ga')

        # between 'experiments', simple dictionary reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference={'ga':'i3', 'gb':'i5'}, out=out)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'), 0)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'), 1)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [1,3])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4])[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4])[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i5')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i5')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='testValue'), 0)
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='confidence'), 1)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='reference'), 'i5')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='referenceGroup'), 'gb')

        # between 'experiments', dictionary of dictionaries reference
        stats = sym.doStats(name='vector', test='t', between='experiments',
                            reference={'ga':{'i1':'i3', 'i3':'i1', 'i5':'i1'},
                                       'gb':{'i1':'i5', 'i3':'i1', 'i5':'i1'}},
                            out=out)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'),
            -scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4,5,6])[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i1')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='reference'), 'i1')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [1,3])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='testValue'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4])[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4])[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i5')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i1')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='testValue'),
            -scipy.stats.ttest_ind([1,3], [1,2,3,4])[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='confidence'),
            scipy.stats.ttest_ind([1,3], [1,2,3,4])[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='reference'), 'i1')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='referenceGroup'), 'gb')

    def testDoStatsBetweenGroups(self):
        """
        Tests doStats(between='groups')
        """

        # symatrical instance
        sym = self.makeSymmetricalInstance()

        # between 'groups', string reference
        stats = sym.doStats(name='vector', test='kruskal', between='groups',
                            reference='ga', out=out)
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                     'testValue', 'confidence', 'testSymbol', 'reference',
                     'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['h']*3)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='mean'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='std'), np.sqrt(2))
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='n'), 2)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([1,3]), np.array([1,3]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([1,3]), np.array([1,3]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.ga.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='reference'), 'i5')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [1,3])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([1,3]), np.array([1,3]))[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([1,3]), np.array([1,3]))[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i3',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i5',
                                          property='testValue')), True)
        np_test.assert_equal(
            np.isnan(stats.gb.getValue(identifier='i5',
                                          property='confidence')), True)
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='reference'), 'i5')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i5', property='referenceGroup'), 'ga')

        # between 'experiments', groups and identifiers, string reference
        stats = sym.doStats(name='vector', test='t', between='groups',
                            reference='ga', out=out,
                            groups=['ga'], identifiers=['i3', 'i5'])
        np_test.assert_equal(list(stats.keys()), ['ga'])
        np_test.assert_equal(stats.ga.identifiers, ['i3', 'i5'])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])

        # between 'experiments', identifiers, string reference
        stats = sym.doStats(name='vector', test='t', between='groups',
                            reference='ga', out=out,
                            groups=['ga', 'gb'], identifiers=['i3', 'i5'])
        np_test.assert_equal(set(stats.keys()), set(['ga', 'gb']))
        for stats_group in list(stats.values()):
            np_test.assert_equal(stats_group.identifiers, ['i3', 'i5'])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i5', property='data'), [1,2,3,4])

        # between 'groups', simple dictionary reference
        sym2 = self.makeSymmetricalInstance()
        sym2.gb.setValue(identifier='i1', property='vector',
                         value=np.array([2,5]))
        sym2.gb.setValue(identifier='i3', property='vector',
                         value=np.array([3,7,8]))
        stats = sym2.doStats(name='vector', test='kruskal', between='groups',
                            reference={'i1':'gb', 'i3':'ga', 'i5':'gb'},
                             out=out)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([1,3]), np.array([2,5]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([1,3]), np.array([2,5]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [2, 5])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([2,5]), np.array([2,5]))[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([2,5]), np.array([2,5]))[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [3,7,8])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='testValue'),
            scipy.stats.kruskal(np.array([3,7,8]),
                                np.array([1,2,3,4,5,6]))[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='confidence'),
            scipy.stats.kruskal(np.array([3,7,8]),
                                np.array([1,2,3,4,5,6]))[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'ga')

        # between 'groups', dictionary of dictionaries reference
        stats = sym2.doStats(name='vector', test='kruskal', between='groups',
                            reference={'i1':{'ga':'gb', 'gb':'ga'},
                                       'i3':{'ga':'ga', 'gb':'ga'},
                                       'i5':{'ga':'gb', 'gb':'gb'}},
                             out=out)
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([1,3]), np.array([2,5]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([1,3]), np.array([2,5]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='referenceGroup'), 'gb')
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='testValue'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[0])
        np_test.assert_almost_equal(
            stats.ga.getValue(identifier='i3', property='confidence'),
            scipy.stats.kruskal(np.array([1,2,3,4,5,6]),
                                np.array([1,2,3,4,5,6]))[1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='data'), [2, 5])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='testValue'),
            scipy.stats.kruskal(np.array([2,5]), np.array([3,1]))[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i1', property='confidence'),
            scipy.stats.kruskal(np.array([2,5]), np.array([3,1]))[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='reference'), 'i1')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i1', property='referenceGroup'), 'ga')
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='data'), [3,7,8])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='testValue'),
            scipy.stats.kruskal(np.array([3,7,8]),
                                np.array([1,2,3,4,5,6]))[0])
        np_test.assert_almost_equal(
            stats.gb.getValue(identifier='i3', property='confidence'),
            scipy.stats.kruskal(np.array([3,7,8]),
                                np.array([1,2,3,4,5,6]))[1])
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='reference'), 'i3')
        np_test.assert_equal(
            stats.gb.getValue(identifier='i3', property='referenceGroup'), 'ga')

    def testDoStatsHistoBetweenExperiments(self):
        """
        Tests doStats(between='experiments', bins=...)
        """

        # symmetrical instance
        sym = self.makeSymmetricalInstance()

        # between 'experiments', string reference
        stats = sym.doStats(name='vector', test='chi2', bins=[0,2,10],
                            fraction=1, between='experiments',
                            reference='i3', out=out)
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'histogram', 'probability', 'ids',
                     'fraction', 'n', 'testValue', 'confidence', 'testSymbol',
                     'reference', 'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['chi2'] * 3)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='data'), [1, 3])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='histogram'), [1, 1])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='probability'),
            [0.5, 0.5])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i1', property='fraction'), 0.5)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='data'), [1,2,3,4,5,6])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='histogram'), [1, 5])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='probability'),
            [1./6, 5./6])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i3', property='fraction'), 5./6)
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='data'), [])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='histogram'), [0, 0])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='probability'),
            [np.nan, np.nan])
        np_test.assert_equal(
            stats.ga.getValue(identifier='i5', property='fraction'), np.nan)

    def testCountHistogram(self):
        """
        Test countHistogram()
        """
        # symmetrical instance
        sym = self.makeSymmetricalInstance()

        # string reference
        stats = sym.countHistogram(test='chi2', reference='i3', out=out)
        for g_name, group in list(stats.items()):
            np_test.assert_equal(
                stats[g_name].properties,
                set(['identifiers', 'count',
                     'fraction', 'n', 'testValue', 'confidence', 'testSymbol',
                     'reference']))
            np_test.assert_equal(stats[g_name].testSymbol, ['chi2'] * 3)
        np_test.assert_equal(
            stats.ga.getValue(property='count', identifier='i1'), 2)
        np_test.assert_equal(
            stats.gb.getValue(property='count', identifier='i1'), 2)
        np_test.assert_equal(
            stats.ga.getValue(property='count', identifier='i3'), 6)
        np_test.assert_equal(
            stats.gb.getValue(property='count', identifier='i3'), 0)
        np_test.assert_equal(
            stats.ga.getValue(property='count', identifier='i5'), 0)
        np_test.assert_equal(
            stats.gb.getValue(property='count', identifier='i5'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='n', identifier='i1'), 4)
        np_test.assert_equal(
            stats.gb.getValue(property='n', identifier='i1'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='n', identifier='i3'), 6)
        np_test.assert_equal(
            stats.gb.getValue(property='n', identifier='i3'), 6)
        np_test.assert_equal(
            stats.ga.getValue(property='n', identifier='i5'), 4)
        np_test.assert_equal(
            stats.gb.getValue(property='n', identifier='i5'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i3'), 1.)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i3'), 0.)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i5'), 0.)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i5'), 1.)
        np_test.assert_equal(
            stats.ga.getValue(property='confidence', identifier='i1'),
            pyto.util.scipy_plus.chisquare_2([2,2], [6,0])[1])
        np_test.assert_equal(
            stats.ga.getValue(property='confidence', identifier='i5'),
            pyto.util.scipy_plus.chisquare_2([0,4], [6,0])[1])

        # string reference
        ref = {'i1' : 'i1', 'i3' : 'i1', 'i5' : 'i1'}
        stats = sym.countHistogram(test='chi2', reference=ref, out=out)
        np_test.assert_equal(
            stats.ga.getValue(property='confidence', identifier='i3'),
            pyto.util.scipy_plus.chisquare_2([2,2], [6,0])[1])
        np_test.assert_equal(
            stats.ga.getValue(property='confidence', identifier='i5'),
            pyto.util.scipy_plus.chisquare_2([0,4], [2, 2])[1])

        # other name
        stats = sym.countHistogram(name='vector', test='chi2',
                                   reference='i5', out=out)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i3'), 1.)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i3'), 0.)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i5'), 0.)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i5'), 1.)
        np_test.assert_equal(
            stats.gb.getValue(property='confidence', identifier='i1'),
            pyto.util.scipy_plus.chisquare_2([2,2], [0, 4])[1])
        np_test.assert_equal(
            stats.gb.getValue(property='confidence', identifier='i3'),
            pyto.util.scipy_plus.chisquare_2([0,4], [6, 0])[1])

        # some identifiers
        stats = sym.countHistogram(name='vector', test='chi2', out=out,
                                   reference='i5', identifiers=['i1', 'i5'])
        np_test.assert_equal(stats.ga.identifiers, ['i1', 'i5'])
        np_test.assert_equal(
            stats.ga.getValue(property='count', identifier='i1'), 2)
        np_test.assert_equal(
            stats.gb.getValue(property='count', identifier='i1'), 2)
        np_test.assert_equal(
            stats.ga.getValue(property='count', identifier='i5'), 0)
        np_test.assert_equal(
            stats.gb.getValue(property='count', identifier='i5'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='n', identifier='i1'), 4)
        np_test.assert_equal(
            stats.gb.getValue(property='n', identifier='i1'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='n', identifier='i5'), 4)
        np_test.assert_equal(
            stats.gb.getValue(property='n', identifier='i5'), 4)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i1'), 0.5)
        np_test.assert_equal(
            stats.ga.getValue(property='fraction', identifier='i5'), 0.)
        np_test.assert_equal(
            stats.gb.getValue(property='fraction', identifier='i5'), 1.)
        np_test.assert_equal(
            stats.ga.getValue(property='confidence', identifier='i1'),
            pyto.util.scipy_plus.chisquare_2([2,2], [0, 4])[1])

    def test_histo_to_cdf(self):
        """
        Tests histo_to_cdf()
        """

        stats = self.makeStats()
        bins = [1, 3, 7]
        cdf_tab = stats.histo_to_cdf(bins, num_bin_label='bin_label')
        np_test.assert_equal(cdf_tab.index.to_list(), bins)
        np_test.assert_almost_equal(cdf_tab['g1'].to_list(), [0.4, 0.9, 1])
        np_test.assert_almost_equal(cdf_tab['g2'].to_list(), [0.2, 0.6, 1])
        np_test.assert_almost_equal(cdf_tab['g3'].to_list(), [0., 0.4, 1])
        
    def testJoinExperimentsList(self):
        """
        Tests joinExperimentsList()
        """

        # prepare
        sym = self.makeSymmetricalInstance()
        sym_p = self.makeSymmetricalInstance()
        def plus(x, y): return x + y
        sym_p.apply(funct=plus, args=['vector'], kwargs={'y':10},
                    name='vector')
        g_list = [sym, sym_p]
        joined = Groups.joinExperimentsList(
            list=g_list, listNames=['orig', 'orig_p'], name='vector',
            mode='join')
        joined_orig = sym.joinExperiments(name='vector', mode='join')
        joined_orig_p = sym_p.joinExperiments(name='vector', mode='join')

        # test metadata
        np_test.assert_equal(set(joined.keys()), set(['orig', 'orig_p']))
        for g_name, groups in list(joined.items()):
            np_test.assert_equal(
                groups.properties,
                set(['idNames', 'ids', 'vector', 'identifiers']))
            np_test.assert_equal(
                groups.indexed, set(['idNames', 'ids', 'vector']))
            np_test.assert_equal(
                set(groups.identifiers), set(['ga', 'gb']))

        # test if the list is converted to groups properly
        np_test.assert_almost_equal(
            joined.orig.getValue('ga', 'vector'),
            joined_orig.getValue('ga', 'vector'))
        np_test.assert_almost_equal(
            joined.orig.getValue('gb', 'vector'),
            joined_orig.getValue('gb', 'vector'))
        np_test.assert_almost_equal(
            joined.orig_p.getValue('ga', 'vector'),
            joined_orig_p.getValue('ga', 'vector'))
        np_test.assert_almost_equal(
            joined.orig_p.getValue('gb', 'vector'),
            joined_orig_p.getValue('gb', 'vector'))

    def testJoinAndStatsList(self):
        """
        Tests joinAndStatsList().
        """

        # prepare
        sym = self.makeSymmetricalInstance()
        sym_p = self.makeSymmetricalInstance()
        def plus(x, y): return x + y
        sym_p.apply(funct=plus, args=['vector'], kwargs={'y':10},
                    name='vector')
        g_list = [sym, sym_p]

        # between groups, reference simple dictionary
        stats = Groups.joinAndStatsList(
            list=g_list, listNames=['orig', 'orig_p'], name='vector',
            mode='join', test='t', between='groups',
            reference={'orig':'ga', 'orig_p':'gb'})
        stats_orig = sym.joinAndStats(name='vector', mode='join', test='t',
                                      reference='ga', out=None)
        stats_orig_p = sym_p.joinAndStats(name='vector', mode='join',
                                          test='t', reference='gb', out=None)

        # test metadata
        np_test.assert_equal(set(stats.keys()), set(['orig', 'orig_p']))
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                     'testValue', 'confidence', 'testSymbol', 'reference',
                     'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['t']*2)
            np_test.assert_equal(
                set(groups.identifiers), set(['ga', 'gb']))

        # test if the list is converted to groups properly
        np_test.assert_almost_equal(
            stats.orig.getValue('ga', 'data'),
            stats_orig.getValue('ga', 'data'))
        np_test.assert_almost_equal(
            stats.orig.getValue('ga', 'std'),
            stats_orig.getValue('ga', 'std'))
        np_test.assert_almost_equal(
            stats.orig.getValue('gb', 'confidence'),
            stats_orig.getValue('gb', 'confidence'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('ga', 'data'),
            stats_orig_p.getValue('ga', 'data'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('gb', 'confidence'),
            stats_orig_p.getValue('gb', 'confidence'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('gb', 'testValue'),
            stats_orig_p.getValue('gb', 'testValue'))

        # between groups, reference simple dictionary, groups
        stats = Groups.joinAndStatsList(
            list=g_list, listNames=['orig', 'orig_p'], name='vector',
            groups=['gb'], mode='join', test='t', between='groups',
            reference={'orig':'gb', 'orig_p':'gb'})
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(groups.testSymbol, ['t'])
            np_test.assert_equal(
                set(groups.identifiers), set(['gb']))
        np_test.assert_almost_equal(
            stats.orig.getValue('gb', 'data'),
            stats_orig.getValue('gb', 'data'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('gb', 'data'),
            stats_orig_p.getValue('gb', 'data'))

        # between list items, reference dictionary of dictionaries
        stats = Groups.joinAndStatsList(
            list=g_list, listNames=['orig', 'orig_p'], name='vector',
            mode='join', test='t', between='list_items',
            reference={'ga':{'orig':'orig_p', 'orig_p':'orig'},
                       'gb':{'orig':'orig', 'orig_p':'orig'}})
        stats_orig = sym.joinAndStats(name='vector', mode='join', out=None)
        stats_orig_p = sym_p.joinAndStats(name='vector', mode='join', out=None)

        # test metadata
        np_test.assert_equal(set(stats.keys()), set(['orig', 'orig_p']))
        for g_name, groups in list(stats.items()):
            np_test.assert_equal(
                groups.properties,
                set(['identifiers', 'data', 'mean', 'std', 'n',  'sem',
                     'testValue', 'confidence', 'testSymbol', 'reference',
                     'referenceGroup']))
            np_test.assert_equal(groups.testSymbol, ['t']*2)
            np_test.assert_equal(
                set(groups.identifiers), set(['ga', 'gb']))

        # test if the list is converted to groups properly
        np_test.assert_almost_equal(
            stats.orig.getValue('ga', 'data'),
            stats_orig.getValue('ga', 'data'))
        np_test.assert_almost_equal(
            stats.orig.getValue('ga', 'testValue'),
            scipy.stats.ttest_ind(
                stats.orig.getValue('ga', 'data'),
                stats.orig_p.getValue('ga', 'data'))[0])
        np_test.assert_equal(
            stats.orig.getValue(identifier='ga', property='reference'), 'ga')
        np_test.assert_equal(
            stats.orig.getValue(identifier='ga', property='referenceGroup'),
            'orig_p')
        np_test.assert_almost_equal(
            stats.orig.getValue('gb', 'data'),
            stats_orig.getValue('gb', 'data'))
        np_test.assert_almost_equal(
            stats.orig.getValue('gb', 'testValue'),
            scipy.stats.ttest_ind(
                stats.orig.getValue('gb', 'data'),
                stats.orig.getValue('gb', 'data'))[0])
        np_test.assert_equal(
            stats.orig.getValue(identifier='gb', property='reference'), 'gb')
        np_test.assert_equal(
            stats.orig.getValue(identifier='gb', property='referenceGroup'),
            'orig')
        np_test.assert_almost_equal(
            stats.orig_p.getValue('ga', 'data'),
            stats_orig_p.getValue('ga', 'data'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('ga', 'testValue'),
            scipy.stats.ttest_ind(
                stats.orig_p.getValue('ga', 'data'),
                stats.orig.getValue('ga', 'data'))[0])
        np_test.assert_equal(
            stats.orig_p.getValue(identifier='ga', property='reference'), 'ga')
        np_test.assert_equal(
            stats.orig_p.getValue(identifier='ga', property='referenceGroup'),
            'orig')
        np_test.assert_almost_equal(
            stats.orig_p.getValue('gb', 'data'),
            stats_orig_p.getValue('gb', 'data'))
        np_test.assert_almost_equal(
            stats.orig_p.getValue('gb', 'testValue'),
            scipy.stats.ttest_ind(
                stats.orig_p.getValue('gb', 'data'),
                stats.orig.getValue('gb', 'data'))[0])
        np_test.assert_equal(
            stats.orig_p.getValue(identifier='gb', property='reference'), 'gb')
        np_test.assert_equal(
            stats.orig_p.getValue(identifier='gb', property='referenceGroup'),
            'orig')

    def testDoCorrelation(self):
        """
        Tests doCorrelation()
        """

        # mode is None
        gs = self.makeInstance()
        def plus(x, y): return x + y
        gs.apply(funct=plus, args=['vector'], kwargs={'y':10}, name='vector_2')

        corr = gs.doCorrelation(xName='vector', yName='vector_2',
                                test='r', mode=None, out=out)
        for g_name, group in list(gs.items()):
            co = corr[g_name]
            new_props = set(['testValue', 'testSymbol', 'confidence'])
            np_test.assert_equal(co.properties.issuperset(new_props), True)
            np_test.assert_equal(co.testSymbol,
                                 ['r']*len(group.identifiers))
            for ident in group.identifiers:
                np_test.assert_equal(
                    group.getValue(property='vector', identifier=ident),
                    co.getValue(property='xData', identifier=ident))
                np_test.assert_equal(
                    group.getValue(property='vector_2', identifier=ident),
                    co.getValue(property='yData',  identifier=ident))
        for ident in ['ia1', 'ia3']:
            np_test.assert_almost_equal(
                corr.ga.getValue(identifier=ident, property='testValue'), 1)
        np_test.assert_equal(
            np.isnan(corr.ga.getValue(identifier='ia5',
                                         property='testValue')), True)
        for ident in ['ib1', 'ib3']:
            np_test.assert_almost_equal(
                corr.gb.getValue(identifier=ident, property='testValue'), 1)
        np_test.assert_equal(
            np.isnan(corr.gb.getValue(identifier='ib2',
                                         property='testValue')), True)
        np_test.assert_almost_equal(corr.gc.identifiers, np.array([]))
        np_test.assert_almost_equal(corr.gc.confidence, np.array([]))

        # mode 'join'
        gs = self.makeInstance()
        gs.ga.scalar_2 = [3, 6, 11]
        gs.ga.properties.add('scalar_2')
        gs.gb.scalar_2 = [3, 6, 11]
        gs.gb.properties.add('scalar_2')
        gs.gc.scalar_2 = []
        gs.gc.properties.add('scalar_2')
        gs.gd.scalar_2 = [13]
        gs.gd.properties.add('scalar_2')
        corr = gs.doCorrelation(
            xName='scalar', yName='scalar_2', test='r', regress=True,
            mode='join', out=out)
        np_test.assert_equal(set(corr.identifiers), set(gs.keys()))
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='xData'),
            gs.ga.scalar)
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='yData'),
            gs.ga.scalar_2)
        desired = scipy.stats.pearsonr(gs.ga.scalar, gs.ga.scalar_2)
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='testValue'), desired[0])
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='confidence'), desired[1])
        for ident in ['ga', 'gb']:
            desired = scipy.stats.linregress(gs[ident].scalar, [3, 6, 11])
            np_test.assert_equal(
                corr.getValue(identifier=ident, property='aRegress'),
                desired[0])
            np_test.assert_equal(
                corr.getValue(identifier=ident, property='bRegress'),
                desired[1])
        for ident in ['gc', 'gd']:
            np_test.assert_equal(
                corr.getValue(identifier=ident, property='aRegress'), np.NaN)
            np_test.assert_equal(
                corr.getValue(identifier=ident, property='bRegress'), np.NaN)

        # mode 'mean'
        gs = self.makeInstance()
        corr = gs.doCorrelation(
            xName='vector', yName='scalar', test='r', mode='mean', out=out)
        np_test.assert_equal(set(corr.identifiers), set(gs.keys()))
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='xData'), [2, 3.5])
        np_test.assert_equal(
            corr.getValue(identifier='ga', property='yData'), [2, 6])
        np_test.assert_equal(
            corr.getValue(identifier='gb', property='xData'), [2, 2.5])
        np_test.assert_equal(
            corr.getValue(identifier='gb', property='yData'), [2, 6])
        np_test.assert_equal(
            corr.getValue(identifier='gc', property='xData'), [])
        np_test.assert_equal(
            corr.getValue(identifier='gc', property='yData'), [])
        np_test.assert_equal(
            corr.getValue(identifier='gd', property='xData'), [4])
        np_test.assert_equal(
            corr.getValue(identifier='gd', property='yData'), [12])

    def testMinMax(self):
        """
        Tests min() and max()
        """

        gs = self.makeInstance()

        #  simple
        np_test.assert_equal(gs.min(name='scalar'), 2)
        np_test.assert_equal(gs.max(name='scalar'), 12)
        np_test.assert_equal(gs.min(name='vector'), 1)
        np_test.assert_equal(gs.max(name='vector'), 6)

        # some nan
        np_test.assert_equal(gs.min(name='scalar', categories=['gc', 'ga']), 2)
        np_test.assert_equal(gs.max(name='scalar', categories=['gc', 'ga']), 10)
        np_test.assert_equal(gs.min(name='scalar', categories=['ga', 'gc']), 2)
        np_test.assert_equal(gs.max(name='scalar', categories=['ga', 'gc']), 10)
        np_test.assert_equal(gs.min(name='vector', categories=['gc', 'ga']), 1)
        np_test.assert_equal(gs.max(name='vector', categories=['gc', 'ga']), 6)

        # all nan
        np_test.assert_equal(
            np.isnan(gs.min(name='scalar', categories=['gc', 'gc'])), True)
        np_test.assert_equal(
            np.isnan(gs.max(name='scalar', categories=['gc', 'gc'])), True)

    def testGetNPositive(self):
        """
        Tests getNPositive()
        """

        gs = self.makeInstance()
        gs.ge = Observations()
        gs.ge.identifiers = ['ie1', 'ie2']
        gs.ge.ids = [list(range(4)), list(range(3))]
        gs.ge.vector = [np.array([1, 0, 0, 1]), np.array([0, 0, 1])]

        gs.getNPositive(name='vector', n_name='n_positive')
        np_test.assert_equal('n_positive' in gs.ga.properties, True)
        np_test.assert_equal('n_positive' in gs.gb.indexed, False)
        np_test.assert_equal(
            gs.ga.getValue(property='n_positive', identifier='ia1'),
            2)
        np_test.assert_equal(
            gs.ga.getValue(property='n_positive', identifier='ia5'),
            0)
        np_test.assert_equal(
            gs.ge.getValue(property='n_positive', identifier='ie1'),
            2)
        np_test.assert_equal(
            gs.ge.getValue(property='n_positive', identifier='ie2'),
            1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGroups)
    unittest.TextTestRunner(verbosity=2).run(suite)
