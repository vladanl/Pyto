"""
Class ClassificationResults can hold and analyze results of multiple 
classifications.

Author: Vladan Lucic
"""

import os
import pickle
import itertools

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import sklearn.metrics
from sklearn.metrics import confusion_matrix, f1_score
#from IPython.core.display import display, HTML

import pyto
from pyto.io.pandas_io import PandasIO


class ClassificationResults():
    """
    Holds and analyzes multiple classifications.
    """

    def __init__(self, data=None, true_label='true'):
        """Saves arguments as attributes.
        """

        self.data = data
        self.true_label = true_label
        #self.overwrite = overwrite

    def save_data(
            self, path, file_formats=['pkl', 'json'], hdf5_name=None,
            overwrite=True, verbose=True, out_desc=''):
        """Saves data (pandas.DataFrame)
        """
        #pdio = PandasIO(
        #    file_formats=file_formats, overwrite=overwrite, verbose=verbose)
        #pdio.write_table(table=self.data, base=path)
        PandasIO.write(
            table=self.data, base=path, file_formats=file_formats,
            hdf5_name=hdf5_name, overwrite=overwrite, verbose=verbose,
            out_desc=out_desc)
        
    @classmethod
    def read_data(
            cls, path, file_formats=['pkl', 'json'], hdf5_name=None,
            true_label='true', verbose=True, out_desc=''):
        """Reads data from a file and returns an instance of this class. 
        """
        #pdio = PandasIO(file_formats=file_formats, verbose=verbose)
        #data = pdio.read_table(base=path)
        data = PandasIO.read(
            base=path, file_formats=file_formats, hdf5_name=hdf5_name,
            verbose=verbose, out_desc=out_desc)
        inst = cls(data=data, true_label=true_label)

        return inst
        
    @property
    def predict_labels(self):
        """Labels corresponding to predicted classifications
        """
        pl = [col for col in self.data.columns if col != self.true_label]
        return pl
        
    def add_classification(self, id_, true, predict, overwrite=False):
        """Adds results of one classification.

        """

        # sanity check
        if self.data is not None:
           if not (self.data[self.true_label] == true).all():
               raise ValueError(
                   "The specified correct (true) data has to be the same as "
                   + "the true data column of this instance")

        # add to data
        if self.data is not None:
            predict_df = pd.DataFrame({id_: predict})
            if overwrite or (id_ not in self.data.columns):
                self.data = pd.concat(
                    [self.data, predict_df.set_index(self.data.index)], axis=1)
            else:
                raise ValueError(
                    f"Column {id_} already exists and self.overwrite is False")
        else:
            self.data = pd.DataFrame({self.true_label: true, id_: predict})

    def multi_confusion_matrix(
            self, id_=None, id_ref=None, reduce=None, dataframe=True):
        """Makes confusion matrices for multiple classifications. 

        Classes of classifications specified by args id_ and id_ref have to
        mach. Therefore, this method is applicable to classification but
        not clustering results.
        """

        if id_ref is None:
            id_ref = self.true_label
        if id_ is None:
            id_ = self.predict_labels
        if isinstance(id_, str):
            id_ = [id_]

        cm_multi = [
            sklearn.metrics.confusion_matrix(
                self.data['true'], self.data[cl_id]) 
            for cl_id in id_]
        cm_multi = np.stack(cm_multi, axis=2)

        if reduce is None:
            result = cm_multi
        else:
            cm = getattr(cm_multi, reduce)(axis=2)
            if dataframe:
                class_inds = range(cm.shape[1])
                result = pd.DataFrame(
                    data=cm, columns=class_inds, index=class_inds)
                result = result.rename_axis('True')
            
        return result
    
    def accuracy_score(self, id_=None, id_ref=None, normalize=True, pairs=False):
        """Calculates accuracy between two classifications.
        """

        # figure out ids and set logic 
        if id_ is None:
            id_ = self.predict_labels
        multi_id = False
        if isinstance(id_, (list, tuple)):
            multi_id = True
        if id_ref is None:
            if pairs:
                id_ref = self.predict_labels
            else:
                id_ref = self.true_label

        # all combination of classes
        if pairs:
            if not multi_id:
                id_ = [id_]
            accu_np = np.zeros((len(id_ref), len(id_))) - 1
            accu = pd.DataFrame(data=accu_np, columns=id_, index=id_ref)
            for id1, id2 in itertools.product(id_, id_ref):
                try:
                    inverse = accu.at[id1, id2]
                    if inverse >= 0:
                        accu.at[id2, id1] = inverse
                        continue
                except KeyError:
                    pass
                if id1 == id2:
                    current_accu = 1
                else:
                    current_accu = self.accuracy_score(
                        id_=id1, id_ref=id2, normalize=normalize)
                accu.at[id2, id1] = current_accu
            return accu                

        # many classes vs one class
        if multi_id and not pairs:
            accu = [
                self.accuracy_score(
                    id_=id_one, id_ref=id_ref, normalize=normalize)
                for id_one in id_]
            return accu

        # one vs one class
        accu = sklearn.metrics.accuracy_score(
            y_true=self.data[id_ref], y_pred=self.data[id_],
            normalize=normalize)
        return accu

    def f1_score(self, id_=None, id_ref=None, average='macro'):
        """Calculates f1 score between two classifications


        """

        if id_ref is None:
            id_ref = self.true_label           
        if id_ is None:
            id_ = self.predict_labels

        if isinstance(id_, (list, tuple)):
            score = [
                sk.metrics.f1_score(
                    y_true=self.data[id_ref], y_predict=self.data[id_one],
                    average=average)
                for id_one in id_]
            return score
            
        score = sk.metrics.f1_score(
            y_true=self.data[id_ref], y_predict=self.data[id_], average=average)
        return score
