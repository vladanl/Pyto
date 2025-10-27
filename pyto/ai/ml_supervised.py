"""
Contains class MLSupervised for supervised ml classification

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
"""

import os
import sys
import subprocess
import importlib
import itertools
import pickle
import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score
try:
    from IPython.core.display import display, HTML  # depreciated
except ImportError:
    from IPython.display import display, HTML
    
    
import pyto
from pyto.io.pandas_io import PandasIO


class MLSupervised:
    """Supervised ML.

    """

    def __init__(
            self, estim=None, cv=None, random_state=18, stratify=None,
            scoring='f1_macro', fit_single=True, print_coefs=True,
            single_print_report=False, cv_print_report=False):
        """Sets arguments.
        """

        self.estim = estim
        self.cv = cv
        self.random_state = random_state
        self.stratify = stratify
        self.scoring = scoring
        self.fit_single = fit_single
        self.print_coefs = print_coefs
        self.single_print_report = single_print_report
        self.cv_print_report = cv_print_report

    def evaluate(self, data, features, target):
        """ Fits data using a classifier, with cross-validation.

        First, the specified data (pandas.DataFrame) is split into features 
        and target(s) and into training and testing data.

        Intended to be used in two ways:

        1) Parameter search with cross-validation (CV). Set by arg
        fit_single=True and arg cv should be None. The specified
        estimator (arg estim) should be 
        a parameter search CV object that was instantiated with a classifier 
        or a pipe containing a classifier. It does the following:
          - The classifier (or a pipe) is fit for all specified parameter 
          combinations, with CV, using the training data. 
          Consequently, if the pipe contains a transformer, the transformer is 
          fit using only the training data (actually only a subset of the 
          training data that is determined by the cross-validation). 
          - Scores and predictions are calculated using the best model, for 
          training and for testing data
          - Prints the best fit results: parameters, coefficient values,
          training and test scores, training and test confusion matrix,
          and the classification report (sk.metrics.classification_report)

        2) Only CV without parameter search. Set by specifying arg cv. The 
        specified estimator (arg estim) has to be a classifier. It does the 
        following if arg fit_single=True:
          - Fits the classifier using the training data
          - Calculates the score and prediction for training and for
          testing data
          - Prints the classifier coefficient values, training and test
          scores, training and test confusion matrix, and the
          classification report (sk.metrics.classification_report)
        It does the following regardless of arg fit_single:
          - CV on all data (that is training and test data together).
          - Calculates the score and the prediction for all data together
          - Prints the score, confusion matrix, and the classification report 
        Warning: the second part should be used only for exploration because 
        the results are not provided for separate testing data.

        Arguments:
          - data: (pandas.DataFrame) all data
          - features: (list) names of data featue columns
          - target: (str or list) names of one or more data target columns
          - random_state: random state used to split data in training
          and test sets
          - estim: estimator that can be a classifier, or a parameter search 
          CV object that was instantiated with a classifier or a pipe
          containing a classifier
          - scoring: name of the scoring function 
          - cv: 
          - fit_single:
          - print_coefs
          - single_print_report
          - cv_print_report
        """

        # set data
        X, y = self.initialize_data(
            data=data, features=features, target=target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.random_state, stratify=self.stratify)

        if self.fit_single:
            clf_best = self.fit_print_single(
                X_train=X_train, y_train=y_train, X_test=X_test,
                y_test=y_test, features=features, clf=self.estim) 

        if self.cv is not None:
            self.fit_print_cv(X=X, y=y, clf=self.estim)

        return clf_best

    def initialize_data(self, data, features, target, scale=False):
        """Separates data into input (features) and output (target) variables.

        Optional scaling is depreciated
        """

        X = data[features]
        y = data[target]

        if scale:
            scaler = sk.preprocessing.StandardScaler().fit(X)
            X = scaler.transform(X)

        return X, y

    def fit_print_single(
            self, X_train, y_train, clf, features, X_test=None, y_test=None,
            f1_average='macro', clf_step='clf'):
        """
        Fits the specified parameter search estimator and makes predictions.

        Intended for parameter search where the estimator passed to
        the parameter search constructor is a classifier or a pipe
        containing a classifier.

        Arguments:
          - X_train, y_train: training features and target(s)
          - X_test, y_test: testing features and target(s)
          - clf: parameter search estimator

        Returns the best estimator
        """

        # fit 
        clf.fit(X_train, y_train)

        # check if param search or in pipeline and get best params
        try:
            clf_best = clf.best_estimator_.named_steps[clf_step]
            search_desc = " for the best estimator"
            best_params = dict([
                (name.removeprefix(f'{clf_step}__'),
                 clf_best.get_params()[name.removeprefix(f'{clf_step}__')])
                for name in clf.get_params()['param_grid'].keys()])
            plain_estimator = False
        except AttributeError:
            try:
                clf_best = clf.best_estimator_
                search_desc = " for the best estimator"
                best_params = dict([
                    (name, clf_best.get_params()[name])
                    for name in clf.get_params()['param_grid'].keys()])
                plain_estimator = False
            except AttributeError:
                clf_best = clf
                search_desc = ''
                plain_estimator = True

        # print best estimator params
        if not plain_estimator:
            print(f'Best estimator: {best_params}')

        # print coefficient values, works for different estimators
        if self.print_coefs:

            # plain logreg and param search 
            try:
                coefs_np = clf_best.coef_
            except AttributeError:
                pass
            else:
                columns = ["Class " + str(x) for x in range(coefs_np.shape[0])]
                try:
                    coefs = pd.DataFrame(
                        coefs_np.transpose(), columns=columns, index=features)
                except ValueError:
                    coefs = pd.DataFrame(coefs_np.transpose(), columns=columns)

            # other estimators
            try:
                coefs = pd.DataFrame(
                    {"Importance": clf_best.feature_importances_},
                    index=clf_best.feature_names_in_)
            except AttributeError as err:
                #print(err)
                pass
            try:
                coefs = pd.DataFrame(
                    {'Scores': clf_best.scores_, 'P values': clf_best.pvalues_},
                    index=clf_best.feature_names_in_)
            except AttributeError as err:
                #print(err)
                pass

            # print nicely
            try:
                display(HTML(coefs.to_html()))
            except UnboundLocalError:
                pass

        # print scores
        try:
            score_train = clf.score(X_train, y_train)
            print(f"Training score{search_desc}: {score_train:6.3f}")
        except AttributeError as err:
            pass
        else:
            if (X_test is not None) and (y_test is not None):
                score_test = clf.score(X_test, y_test)
                print(f"Test score{search_desc}: {score_test:6.3f}")

        # try to predict on train and test data
        try:
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)
        except (AttributeError, ValueError) as err:
            pred_train = None
            pred_test = None

        # confusion matrix
        if pred_train is not None:
            f1_train = f1_score(y_train, pred_train, average=f1_average)
            f1_test = f1_score(y_test, pred_test, average=f1_average)
            print(
                f"f1 {f1_average} scores{search_desc}: train {f1_train:6.3f}, "
                + f"test {f1_test:6.3f}")
            self.print_cm(
                y_train=y_train, pred_train=pred_train, 
                y_test=y_test, pred_test=pred_test)

        # stats report 
        if self.single_print_report:
            print(f"Single test report{search_desc}:")
            target_names = list(map(str, clf_best.classes_)) 
            print(sk.metrics.classification_report(
                y_true=y_test, y_pred=pred_test, target_names=target_names))

        return clf_best

    def fit_print_cv(self, X, y, clf, cv=5):
        """
        Probably depreciated
        """

        # cross-validation to get the score
        sc = cross_val_score(clf, X, y, cv=cv, scoring=self.scoring)
        print(
            f"CV {self.scoring} score: mean = {sc.mean():6.3f}, "
            + f"std = {sc.std():6.3f}")

        # cross-validation to get predictions
        pred = cross_val_predict(clf, X, y, cv=cv)    

        # confusion matrix
        print("CV confusion matrix:")
        self.print_cm(y_train=y, pred_train=pred)

        # stats report 
        if self.single_print_report:
            print("CV report:")
            target_names = list(map(str, clf.classes_)) 
            print(sk.metrics.classification_report(
                y_true=y, y_pred=pred, target_names=target_names))


    @classmethod
    def confusion_matrix_smooth(cls, y_true, y_pred):
        """Smooth confusion matrix.

        Can be used when predicted values are probability-like.

        """
        cm_list = [
            y_pred[y_true==cl_ind].sum(axis=0)
            for cl_ind in range(y_pred.shape[-1])]
        cm = np.vstack(cm_list)
        return cm

    @classmethod
    def f1_score_smooth(cls, y_true, y_pred, average='macro'):
        """F1 score for the case when predicted values are probability-like.

        """
        return cls.fbeta_score_smooth(
            y_true=y_true, y_pred=y_pred, beta=1, average=average)

    @classmethod
    def fbeta_score_smooth(cls, y_true, y_pred, beta, average='macro'):
        """F1 score for the case when predicted values are probability-like.

        The specified predicted values contain probabilities (or some similar 
        measure of the predicted class belonging) instead of hard class
        asignments like in sklearn.metrics.fbeta_score().

        Works for multi-class classification.

        Arguments:
          - y_true: (1d array) true (target) values
          - y_pred: (2d array, N samples x N classes) predicted values
          - beta: parameter that determines the weight of recall
          - average: ('macro', 'weighted' or None) determines how scores for 
          individual classes are combined. If None, all class scores
          are returned.
        """

        # get f scores for all classes separately
        cm = cls.confusion_matrix_smooth(y_true=y_true, y_pred=y_pred)
        score_class = (
            (1 + beta**2) * cm.diagonal()
            / (beta**2 * cm.sum(axis=1) + cm.sum(axis=0)))

        # average class f score
        if average is None:
            score = score_class
        elif average == 'macro':
            score = score_class.mean()
        elif average == 'weighted':
            score = (score_class * cm.sum(axis=1)).sum() / cm.sum()
        else:
            raise ValueError(
                f"Sorry, average value {average} is not implemented. "
                + "Currently implemented are None, 'macro' and 'weighted'.")

        return score

    @classmethod
    def print_smooth_cm(
            cls, y_train, pred_train, y_test=None, pred_test=None, 
            file=None, desc=os.linesep):
        """Print smooth confusion matrix.

        Shortcut for print_cm(smooth=True).
        """
        cls.print_cm(
            y_train=y_train, pred_train=pred_train, y_test=y_test,
            pred_test=pred_test, smooth=True, file=file, desc=desc)

    @classmethod
    def print_cm(
            cls, y_train, pred_train, y_test=None, pred_test=None,
            smooth=False, file=None, desc=os.linesep):
        """Print standard (hard) or smooth confusion matrix.

        Confusion matrix is printed for data specified by args y_train and
        pred_train. In additon, if specified, confusion matrix is also printed 
        for another set of data (args y_test and pred_test). The latter case is 
        meant for training, when both training and testing data are present. 

        Arg smooth determines if the standard confusion matrix is printed
        (based on hard class assignement), or the smooth confusion matrix
        (based on class probabilities)

        Arguments:
          - y_train: (N samples if smmoth is False, N samples x N classes if
          smooth is True) expected classes
          - pred_train: (N samples) predicted classes
          - y_test: (N sampless if smmoth is False, N samples x N classes if
          smooth is True) expected classes for the additional data
          (default None)
          - pred_test: (N samples) predicted classes for the additional 
          data (default None)
          - smooth: Flag indicating if the stadard (False) or the smooth
          confusion matrix is printed (default False)
          - file: output file name or descriptor
        """

        if smooth:
            cm_train = cls.confusion_matrix_smooth(
                y_true=y_train, y_pred=pred_train)
        else:
            cm_train = confusion_matrix(y_true=y_train, y_pred=pred_train)
        class_inds = range(cm_train.shape[1])
        cm_df = pd.DataFrame(
            cm_train, index=class_inds,
            columns=map(str, class_inds)).rename_axis('True')

        if pred_test is not None:
            if smooth:
                cm_test = cls.confusion_matrix_smooth(
                    y_true=y_test, y_pred=pred_test)
            else:
                cm_test = confusion_matrix(y_true=y_test, y_pred=pred_test)
            cm_test_df = pd.DataFrame(
                cm_test, index=class_inds,
                columns=map(str, class_inds)).rename_axis('True')
            cm_df = pd.merge(
                cm_df, cm_test_df, left_index=True, right_index=True,
                suffixes=("_train", "_test"))

        with pd.option_context('display.float_format', '{:,.2f}'.format):
            if file is None:
                display(HTML(cm_df.to_html()))
            else:
                print(desc, file=file)
                print(cm_df, file=file)

    @classmethod
    def plot_param_search_cv(
            cls, name, param_search=None, df=None, transform_name='clf__',
            x_log=False, increment=0.1,
            labels={}, y_label='mean_test_score', yerr_label='std_test_score',
            linestyle='solid', title='', ax=None):
        """Plots multi-parameter data one one graph.

        Meant for plotting results from a multidimensional parameter search,
        or values from multi-column dataframes. All data is plotted on the
        same graph in the following way.

        Args y_label and yerr_label indicate variables that are plotted on
        y-axis (value and errorbar, respectively).

        The value of the first parameter (element of arg name) is plotted
        on the x-axis. All combinations of all other parameter values
        are plotted in different colors.

        To avoid overlap between plotted points corresponding to different
        parameter combinations that have the same x-axis value, the points
        are shifted along the x-axis by an relative amount specified by
        arg increment.  

        Arguments:
          - param_search: (dict) cv parameter search object, as cv_results_
          attribute of a param search
          - name: names of all parameters used in the search, the first one
          is plotted on the x axis, it has to have a numerical value
          - df: (pd.DataFrame) table used when param_search is None
          - transform_name: param search keys prefix used when parameter search
          is done on a pipe that specifies the transformer to which
          the parameter belongs (see parameter grid key format)  
          - x_log: flag indicating whether log scale is used on the x-axis
          - increment: (float) proportion factor for x-axis shifts relative to 
          spacing between x-axis values 
          - linestyle: matplotlib linestyle ("None" for no line)
          - labels: (dict) defines a conversion of parameter names to a simpler 
          string that us used for graph legend
        """

        if ax is None:
            fig, ax = plt.subplots()

        if param_search is not None:
            if isinstance(param_search.cv_results_, dict):
                data = pd.DataFrame(param_search.cv_results_)
        elif df is not None:
            data = df
        else:
            raise ValueError("Arg param_search or df has to be specified")

        if isinstance(name, str) or (
                isinstance(name, (list, tuple)) and (len(name) == 1)):
            x_name = name
            if x_name in data.columns:
                x_name_col = f'{x_name}'
            elif f'param_{x_name}' in data.columns:
                x_name_col = f'param_{x_name}'
            elif f'param_clf_{x_name}' in data.columns:
                x_name_col = f'param_clf__{x_name}'
            elif f'param_{transform_name}{x_name}' in data.columns:
                x_name_col = f'param_{transform_name}{x_name}'
            else:
                raise ValueError(
                    f"Column corresponding to {x_name} could not be found")

            ax.errorbar(
                data=data, x=x_name_col, y=y_label,
                yerr=yerr_label, marker='x', linestyle=linestyle)

        elif isinstance(name, (list, tuple)):

            name = name.copy()
            x_name = name.pop(0)
            line_names = name

            # get param names for different search types
            if x_name in data.columns:
                x_name_col = f'{x_name}'
                line_name_cols = [f'{nam}' for nam in line_names]
            elif f'param_{x_name}' in data.columns:
                x_name_col = f'param_{x_name}'
                line_name_cols = [f'param_{nam}' for nam in line_names]
            elif f'param_clf__{x_name}' in data.columns:
                x_name_col = f'param_clf__{x_name}'
                line_name_cols = [f'param_clf__{nam}' for nam in line_names]
            elif f'param_{transform_name}{x_name}' in data.columns:
                x_name_col = f'param_{transform_name}{x_name}'
                line_name_cols = [
                    f'param_{transform_name}{nam}' for nam in line_names]
            else:
                raise ValueError(
                    f"Column corresponding to {x_name} could not be found")

            # get unique values of all line values
            # converting to str to avoid problems when a value is a list
            line_values = [
                data[nam].apply(str).unique() for nam in line_name_cols]

            # loop over all combinations of line values
            for ind, comb in enumerate(itertools.product(*line_values)):

                # get data for the current combination
                cond = np.logical_and.reduce(
                    [data[nam].apply(str) == com for nam, com
                     in zip(line_name_cols, comb)])
                line_data = data[cond]
                x_data = line_data[x_name_col].to_numpy(dtype=float)

                # make label
                comb_converted = [labels.get(str(val), val) for val in comb]
                label_list = [
                    f'{labels.get(nam, nam)}={com}, '
                    for nam, com in zip(line_names, comb_converted)]
                lab = ''.join(label_list)[:-2]

                # shift x-values
                if x_log:
                    x_data = np.log(x_data)
                x_data_unique = np.unique(x_data)
                x_data_inc = (
                    (x_data_unique.max() - x_data_unique.min())
                    / (len(x_data_unique) - 1))
                x_data_shift = x_data + x_data_inc * increment * ind
                if x_log:
                    x_data_shift = np.exp(x_data_shift)

                ax.errorbar(
                    x=x_data_shift, y=line_data[y_label],
                    yerr=line_data[yerr_label],
                    marker='x', linestyle=linestyle, label=lab)

            ax.legend()

        if x_log:
            ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(x_name)
        ax.set_title(f"Param search {title}");

        return ax

    @classmethod
    def save_predictions(
            cls, train_df, predict_df, y_predict_class, target, code,
            output_dir=None, file_formats=['json']):
        """Saves combined predictions and annotations.

        Arguments:
          - train_df: (pandas.DataFrame) training (annotated) data
          - predict_df: (pandas.DataFrame) prediction (non-annotated) data
          - y_predict_class: (1d array) predicted classes (correspond 
          to predict_df)
          - target: column name that contains annotation / prediction code
          - code: ML model identifier, used to form output table file name 
          - output_dir: (default None) output table directory, if None no 
          table is written
          - file_formats: (default ['json']) pyto.io.PandasIO file format  
        """

        predict_df[target] = y_predict_class
        out_df = (pd
            .concat([predict_df, train_df], verify_integrity=True)
            .sort_index())
        if output_dir is not None:
            out_path = os.path.join(output_dir, code)
            PandasIO.write(
                out_df, out_path, file_formats=file_formats, 
                out_desc='predictions and annotations')

        return out_df

    @classmethod
    def get_labeled(cls, df, column):
        """Extracts rows that contain values in the specified columns.

        Keeps rows of df that contain values that are not isnull() for all
        specified columns (arg column)
        
        Arguments:
          - df: (pandas.DataFrame) data table
          - column: (str or list) one or more column names

        Returns (pandas.DataFrame) table composed of a subset of the
        rows of table (arg) df.
        """

        if not isinstance(column, list):
            column = [column]

        res = df[
            df[column]
            .isnull()
            .map(np.logical_not)
            .apply(np.logical_and.reduce, axis=1)]

        return res


