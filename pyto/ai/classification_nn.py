"""
Classes and functions for tether feature classification by neural nets

ToDo: Move to Pyto when it gets stable

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import importlib
import pickle
import inspect

import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.core.display import HTML

import torch
from torch.utils.data import DataLoader

import sklearn as sk
import sklearn.metrics

import pyto
from pyto.io.pandas_io import PandasIO
from .ml_supervised import MLSupervised
from .full_connect import FullConnect
from .features_dataset_nn import FeaturesDatasetNN


class ClassificationNN(object):
    """Classification by features using neural nets.

    Originally deeloped for classification of tethers
    """

    def __init__(
            self, loss_fn, optimizer, n_epochs, batch_size, device=None, 
            model=None, n_features=None, activation=None, batch_norm=None,
            dropout=None, init=None, last_layer=None, last_l1=False,
            lr=None, class_weight=None, f1_average=None, f_beta=1,
            train_dataloader=None, test_dataloader=None, 
            log_file=None, n_batches_print=0,
            cm_epochs_print=0, report_epochs_print=0,
            save_path=None, save_epochs=[]):
        """Saves arguments as attributes and sets additional attributes.

        Class weight is passed as weight argument to loss function. It is
        determined from arguments in this order:
          - directly from class_weight argument, if arg class_weight is
          not None
          - if arg train_dataloader is not None, calculated from that data
          - None if both class_weight and train_dataloader args are None
        In any case the determined value is saved as self._class_weight. The 
        reason for this convoluted behavior is to be able to reproduce this 
        instance by using attributes that were set from arguments.

        Arguments:
          - n_epochs: number of epochs
          - batch_size: batch size
          - loss_fn: loss function class or an instance thereof
          - class_weight: class weight for loss function
          - optimizer: optimizer class of an instance of thereof
          - lr: learning rate
          - model: (FullConnect) Instance of the neural net model. If it is
          specified, model parameters (n_features, activation, batch_norm, 
          dropout, init, last_layer and last_l1) are ignored. Intended for 
          single runs, not compatible with parameter search.
          - n_features, activation, batch_norm, dropout, init, last_layer,
          last_l1: model parameters
          - device: device on which the model is run
          - train_dataloader: training dataloader, used only to determine
          class weight for loss function. Training data need to be passed
          directly to run() or fit()
          - test_dataloader: test dataloader, used for testing
          - log_file: log file path, or None for stdout
          - f1_average: type of multiclass f1 score (passed to 
          sklearn.metrics.f1_score)
          - n_batches_print: interval for printing training loss in the
          number of batches
          - cm_epochs_print, report_epochs_print: interval for printing 
          confusion matrix and classification report, respectively, in the 
          number of epochs. To print these only at the end of the run set
          these to n_epochs, and to avoid printing to 0 
          - save_path: path where this instance is pickled at the end of the
          run, None for not saving it
          - save_epochs: (list) epoch numbers for which this instance
          is pickled
        """

        #
        self._estimator_type = "classifier"
        
        # save arguments as attributes
        # Note that this method also saves model params as attributes of this
        # instance. The only reason is that self.set_model (below) can use
        # these attributes to make a model.
        self.save_args()

        # model needs to be set before optimizer
        self.set_model(model)
        
        # figure out class weight and set loss_fn
        self._class_weight = self.get_class_weight(
            class_weight=class_weight, dataloader=train_dataloader)
        self.set_loss_fn()

        # set optimizer after model and lr are set
        self.set_optimizer()

        # set log file
        #self.set_log_file(log_file=log_file)

        # hyperparameter names
        self.model_param_names = [
            'n_features', 'activation', 'batch_norm', 'dropout', 'init',
            'last_layer', 'last_l1']
        self.loss_fn_param_names = [
            'loss_fn', 'class_weight', 'train_dataloader', 'test_dataloader']
        self.optimizer_param_names = ['optimizer', 'lr']
        self.run_param_names = [
            'n_epochs', 'batch_size', 'device', 'f1_average', 'f_beta']
        self.io_param_names = [
            'log_file', 'n_batches_print',
            'cm_epochs_print', 'report_epochs_print', 'save_path',
            'save_epochs']
        
        # scores tables
        columns = ['epoch', 'loss', 'accuracy']
        self.train_scores = pd.DataFrame(columns=columns)
        self.test_scores = pd.DataFrame(columns=columns)

        # current epoch
        self.epoch = 0

    def save_args(self, ignore=[]):
        """Saves calling function (meant for __init__) arguments as attributes
        """
        frame = inspect.currentframe().f_back
        args, _, _, local_vars = inspect.getargvalues(frame)
        [setattr(self, name, local_vars[name]) for name in args
         if name not in ['self'] + ignore] 

    def get_class_weight(self, class_weight=None, dataloader=None):
        """Returns class weights from arguments or attributes.

        Arg class_weight has precedence over arg dataloader and specified
        arguments have precedence over the corresponding attributes. That is:
          - if arg class_weight is specified, that value is returned
          - if dataloader is specified, it is used to calculate class weight
          - otherwise None is returned

        Arguments:
          - class_weight: array of class weights, the higher the number the 
          more important the corresponding class is, need not be normalized
          - dataloader: data loader containing target training data (class
          indices), expected to be ints
        """

        if class_weight is not None:
            return torch.tensor(class_weight)
        
        if dataloader is None:
            return None

        try:
            cl_count = np.bincount(
                np.hstack([tar.cpu() for _, tar in iter(dataloader)]))
        except TypeError:
            # just in case dataloader has floats, as it happens when
            # data are read from json DataFrame
            cl_count = np.bincount(
                np.hstack(
                    [tar.cpu().to(torch.int) for _, tar in iter(dataloader)]))
        class_weight = torch.tensor(
            cl_count.max() / cl_count, dtype=torch.float, device=self.device)

        return class_weight
            
    def set_loss_fn(self, loss_fn=None):
        """
        Sets loss function object (self._loss_fn_obj)

        Loss function can be specified in two forms:
          - instance of loss function class (doesn't use self._class_weight)
          - loss function class, in which case a new instance of this class
          is screated using class weight (self._class_weight)

        If arg loss_fn is None, self.loss_fn is used.
        """
        if loss_fn is None:
            loss_fn = self.loss_fn
        if isinstance(loss_fn, type):
            self._loss_fn_obj = loss_fn(weight=self._class_weight)
        else:
            self._loss_fn_obj = loss_fn
            
    def set_model(self, model):
        """Sets or makes a model (neural net).

        Sets self.model to arg model if not None, or makes a model from
        model parameters (already set as attributes)
        """

        if model is None:
            self.model = FullConnect(
                n_features=self.n_features, activation=self.activation,
                batch_norm=self.batch_norm, dropout=self.dropout,
                init=self.init, last_layer=self.last_layer,
                last_l1=self.last_l1).to(self.device)
            #print(
            #    f'set_model() None '
            #    + f'{[param.is_cuda for param in self.model.parameters()]}')
        else:
            self.model = model
        
    def set_optimizer(self, optimizer=None, lr=None):
        """Sets optimizer.

        If arg optimizer is None, self.optimizer is used.

        Optimizer can be specified in two forms:
          - instance of an optimizer class, such as nn.CrossEntropyLoss(...) 
          - optimizer class, such as nn.CrossEntropyLoss; in this case the
          class is instantiated
        In both cases self._optimizer is set to the optimizer instance.

        """
        if lr is None:
            lr = self.lr
        if optimizer is None:
            optimizer = self.optimizer
        if optimizer is not None:
            if isinstance(optimizer, type):
                self._optimizer = optimizer(self.model.parameters(), lr=lr) 
            else:
                self._optimizer = optimizer
                
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        """Sets _device

        Set _device to value, or if None, set to 'cuda' if possible,
        otherwise to 'cpu'
        """
        
        if value is None:
            value = "cuda" if torch.cuda.is_available() else "cpu"

        if value == 'cuda':
            cuda_count = torch.cuda.device_count()
            if cuda_count >= 3:
                value = f'cuda:{cuda_count-2}'
            elif cuda_count == 2:
                value = f'cuda:1'
                
        self._device = value

    def set_log_file(self, log_file):
        """Open log file if not None.
        """
        
        if log_file is None:
            self._log_file = None
            return

        # make directory if needed
        dir_, name = os.path.split(log_file)
        try:
            os.makedirs(dir_)
        except FileExistsError:
            pass
        
        base, ext = os.path.splitext(name)
        base_split = base.rsplit('_', 1)
        if (len(base_split) > 1) and base_split[1].isdigit():

            # find file that doesn't exist yet
            num = int(base_split[1])
            while os.path.isfile(log_file):
                num += 1
                name_plus = f"{base_split[0]}_{num}{ext}"
                log_file = os.path.join(dir_, name_plus)

            # immediately write something to the file
            fd = open(log_file, mode='w')
            fd.write(f"Instance id: {id(self)}" + os.linesep)
            os.fsync(fd)

            # bookkeeping
            self.log_file = log_file
            self._log_file = fd
        
        else:
            self.log_file = log_file
            self._log_file = open(log_file, mode='w')
            
    def get_params(self, deep=True):
        """Returns dictionary of parameters.

        Needed for sklearn cross-validation parameter search.

        For example, in order to clone this instance, 
        sklearn.model_selection.GridSearchCV(this_object).fit() calls 
        this function to get parameters and then passes these parameters to
        the constructor to make a new instance. 

        Important: Attribute self.model (neural net) should not be returned, 
        because neural nets should be recreated from parameters. 

        Arguments:
          - deep: no influence, only for sklearn compatibility
        """

        result = {}

        # read model params from self.model
        if self.model is not None:
            for name in self.model_param_names:
                result[name] = getattr(self.model, name)

        # read other params from attributes of this instance
        param_names = (
            self.loss_fn_param_names + self.optimizer_param_names
            + self.run_param_names + self.io_param_names)
        for name in param_names:
            result[name] = getattr(self, name)

        return result
            
    def set_params(self, **params):
        """Sets the specified and their dependent parameters as attributes.

        If any of the given model params is set (self.model_param_names), make 
        a new model (self.model).
        """

        model_params = {}
        optimizer_flag = False
        loss_fn_flag = False

        # set attributes directly from the specified params
        for name, value in params.items():
            if name in self.model_param_names:
                model_params[name] = value
            elif name in self.optimizer_param_names:
                setattr(self, name, value)
                optimizer_flag = True
            elif name in self.loss_fn_param_names:
                setattr(self, name, value)
                loss_fn_flag = True
            else:
                setattr(self, name, value)

        # set attributes that depend on specified params
        # Important: model needs to be updated before optimizer
        if len(model_params) > 0:
            self.model.set_params(**model_params)  # makes new model
            self.model.to(self.device)
            #print(
            #    f'set_params() '
            #    + f'{[param.is_cuda for param in self.model.parameters()]}')
        if loss_fn_flag:
            self.set_loss_fn()
        if optimizer_flag:
            self.set_optimizer()

        #self.set_log_file(log_file=self.log_file)

        return self
        
    def fit(self, X, y):
        """ Trains the model.
        
        Needed for compatibility with scikits learn cross-validation param
        serarch.

        Test data (self.test_dataloader) is used only to generate and evaluate
        predictions. These test results are not used to compare different 
        folds and parameter combinations (in order to find the best parameters, 
        for example).

        Arguments:
          - X, y: ndarrays of training feature and target data, respectively
        """

        # make dataloader
        train_data = FeaturesDatasetNN(X=X, y=y, device=self.device)
        train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True)
        
        # set class weight and loss function object because class weight 
        # depends on the target data (arg y)
        if self.class_weight is None:
            self._class_weight = self.get_class_weight(
                dataloader=train_dataloader)
            self.set_loss_fn()

        # to conform with sklearn
        self.classes_ = np.unique(y)
        
        self.run(train_dataloader=train_dataloader)

        return self

    def predict(
            self, X, y=None, cm_print=True, report_print=True,
            probability=False):
        """ Makes predictions and returns predicted classes.

        Makes prediction using the current model (self.model).
        
        Needed for compatibility with scikits learn cross-validation param
        serarch.

        Essentially a bare-bones version of self.test().

        Expected values (arg y) are used only for printing confusion
        matrix and classification report. If None, confusion matrix
        and classification report are not printed.
        
        Argument:
          - X: (np.ndarray N samples x N features) test features
          - y: (nd_array) expected values
          - cm_print: flag inficating if confusion matrix is printed
          - report_print: flag inficating if classification report is printed
          - probability: flag indicating whether probability is also returned
          (default False)

        Returns (pred, pred_class):
          - pred: (N samples x N classes) probabilistic predictions (only if
          probability=True)
          - pred_class: (N samples) hard-class predictions
        """

        # convert data to np and send to the current (model) device
        X_torch = torch.from_numpy(X).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_torch)
        predictions.to('cpu')
        pred_class_torch = predictions.argmax(dim=1)

        # convert back to np and cpu
        pred_class = pred_class_torch.to('cpu')

        # print confusion matrix
        if (y is not None) and cm_print:
            MLSupervised.print_cm(
                y_train=y, pred_train=pred_class, 
                desc=os.linesep + 'Confusion matrix:')
            MLSupervised.print_smooth_cm(
                y_train=y, pred_train=predictions, 
                desc=os.linesep + 'Smooth confusion matrix:')
            
        # print classification_report
        if (y is not None) and report_print:
            print(os.linesep + 'Train classification report:',
                  file=self._log_file)
            self.classification_report_extended(
                y=y, y_pred=predictions, y_pred_class=pred_class)

        if probability:
            return predictions, pred_class
        else:
            return pred_class
        
    def run(self, train_dataloader, test_dataloader=None, n_epochs=None):
        """
        Training and testing
        """

        # figure out epochs
        if n_epochs is None:
            n_epochs = self.n_epochs
        init_epoch = self.epoch + 1

        # open log file
        self.set_log_file(log_file=self.log_file)
        
        # print info
        print(f"Host: {os.uname()[1]}, Device: {self.device}",
              file=self._log_file)
        params = self.get_params()
        params['init_epoch'] = init_epoch
        print(f"Parameters: {params}", file=self._log_file)
        
        if test_dataloader is None:
            test_dataloader = self.test_dataloader

        for self.epoch in range(init_epoch, n_epochs + init_epoch):

            # setup print
            cm_print_now = (
                (self.cm_epochs_print > 0)
                and (self.epoch % self.cm_epochs_print == 0))
            report_print_now = (
                (self.report_epochs_print > 0)
                and (self.epoch % self.report_epochs_print == 0))

            # train
            self.train(dataloader=train_dataloader)
    
            # test on training data
            desc = (os.linesep + f'Epoch {self.epoch}' + os.linesep
                    + 'Training data')
            pred_train, y_train, self.train_scores, pred_train_class = \
                self.test(
                    dataloader=train_dataloader, scores=self.train_scores,
                    print_=cm_print_now, desc=desc)
 
            # test on test data
            pred_test, y_test, self.test_scores, pred_test_class = self.test(
                dataloader=test_dataloader, scores=self.test_scores,
                print_=cm_print_now, desc='Test data')
 
            # print confusion matrix
            if cm_print_now:
                MLSupervised.print_cm(
                    y_train=y_train, pred_train=pred_train_class, 
                    y_test=y_test, pred_test=pred_test_class,
                    file=self._log_file, desc=os.linesep + 'Confusion matrix:')
                MLSupervised.print_smooth_cm(
                    y_train=y_train, pred_train=pred_train, 
                    y_test=y_test, pred_test=pred_test,
                    file=self._log_file,
                    desc=os.linesep + 'Smooth confusion matrix:')

            # print classification report
            if report_print_now:
                print(os.linesep + 'Test classification report:',
                      file=self._log_file)
                print(
                    sk.metrics.classification_report(
                        pred_test_class, y_test, zero_division=0.0),
                    file=self._log_file)
                f1_macro = MLSupervised.f1_score_smooth(
                    y_true=y_test, y_pred=pred_test, average='macro')
                f1_weighted = MLSupervised.f1_score_smooth(
                    y_true=y_test, y_pred=pred_test, average='weighted')
                print(f"   smooth macro {22*' '} {f1_macro:.2f}",
                      file=self._log_file)
                print(f"smooth weighted {22*' '} {f1_weighted:.2f}",
                      file=self._log_file)
                if self.f_beta != 1:
                    fb_macro = MLSupervised.fbeta_score_smooth(
                        y_true=y_test, y_pred=pred_test, average='macro',
                        beta=self.f_beta)
                    fb_weighted = MLSupervised.fbeta_score_smooth(
                        y_true=y_test, y_pred=pred_test, average='weighted',
                        beta=self.f_beta)
                    print(
                        f"   F{self.f_beta} smooth macro {19*' '} "
                        + f"{fb_macro:.2f}",
                        file=self._log_file)
                    print(
                        f"F{self.f_beta} smooth weighted {19*' '} "
                          + f"{fb_weighted:.2f}", file=self._log_file)
                print(os.linesep, file=self._log_file)

                # save this instance
                if ((self.save_path is not None)
                    and (self.epoch in self.save_epochs)):
                    self.save(epoch=self.epoch)

        # close log file
        try:
            self._log_file.close()
        except AttributeError:
            pass
                                
        # save this instance if not saved already
        if ((self.save_path is not None)
            and (self.epoch not in self.save_epochs)):
            self.save(epoch=self.epoch)

    def train(self, dataloader):
        """ Trains the current models.

        """

        size = len(dataloader.dataset)
        self.model.to(self.device)  # should not be needed but just in case
        self.model.train()

        for batch_ind, (X, y) in enumerate(dataloader):

            # error
            prediction = self.model(X)
            loss = self._loss_fn_obj(input=prediction, target=y)

            # backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if (
                (self.n_batches_print > 0)
                    and (batch_ind % self.n_batches_print == 0)):
                loss, current = loss.item(), batch_ind * len(X)
                if self.log_file is not None:
                    print(
                        f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",
                        file=self._log_file)

    def test(self, dataloader, scores=None, print_=False, desc=''):
        """ Get predictions and scores for the current model and specified data.

        Predictions are calculated at self.device (gpu, if available) and
        then sent to cpu. All returned variables are on cpu.

        Arguments:
          - dataloader: data from which predictions are made
          - scores: (pd.DataFrame) initial scores table
          - print_: flag indicating whether to print accuracy and loss
          - desc: description used for printing 

        Returns tuple of the following:
          - predictions: (N samples x N classes) predicted values, that 
          is outputs of the last network layer 
          - y: (N samples) expected class
          - scores: (pd.DataFrame) updated scores table
          - pred_class: (N samples) predicted classes, that is hard 
          class predictions
        """

        size = len(dataloader.dataset)
        n_batches = len(dataloader)
        self.model.to(self.device)  # should not be needed but just in case
        self.model.eval()

        test_loss, accuracy = 0, 0
        predictions = torch.tensor([], device=self.device)
        y = torch.tensor([], device=self.device)
        with torch.no_grad():
            for X, y_batch in dataloader:
                pred = self.model(X)
                test_loss += self._loss_fn_obj(pred, y_batch).item()  # to cpu
                accuracy += ((pred.argmax(dim=1) == y_batch)
                             .type(torch.float)
                             .sum().item())  # sent to cpu
                predictions = torch.cat((predictions, pred), dim=0)
                y = torch.cat((y, y_batch), dim=0)
 
        # send results to cpu
        predictions = predictions.to('cpu')
        y = y.to('cpu')
                
        # calculate loss and accuracy
        loss_epoch = (test_loss / n_batches)
        accuracy_epoch = (accuracy / size)

        # print epoch results
        if print_:
            print(
                f"{desc} Accuracy: {accuracy_epoch:>0.3f}, "
                + f"Avg loss: {loss_epoch:>8f}", file=self._log_file)

        # calculate scores and find the best class
        scores = self.add_scores(
            y=y, predictions=predictions, loss=loss_epoch,
            accuracy=accuracy_epoch, scores=scores)
        pred_class = predictions.argmax(dim=1)

        return predictions, y, scores, pred_class
    
    def add_scores(self, y, predictions, loss, accuracy, scores=None):
        """Calculates multiclass F-scores and adds them to the scores table.

        The following F-scores are calculated, if self.f1_average is not None:
          - f1
          - fbeta: standard (hard classification) fbeta  
          - fbeta_smooth: smooth version of fbeta
        In all cases averaging done according to self.f1_average and beta is 
        defined by self.f_beta.

        The F-scores, together with the current epoch number, loss and accuracy 
        (obtained from self.epoch, self.loss_epoch and self.accuracy_epoch)
        are used to make a single row table (dataframe). 

        If arg scores is not None, the new row is added to the bottom of the 
        existing scores table. 

        Does not modify attributes of this instance.

        Arguments:
          - y: true values (classes)
          - predictions: outputs of the last network layer (that is, not the 
          hard class predictions)
          - loss, accuracy: current loss and eccuracy
          - scores: table containing scores for previous epochs

        Returns (pandas.DataFrame) table containing the scores calculated here
        as the last row. 
        """

        local_scores = {
            'epoch': self.epoch, 'loss': loss, 'accuracy': accuracy}

        # calculate F1 score
        if self.f1_average is not None:
            predicted_class = predictions.argmax(dim=1).numpy()
            f1 = sklearn.metrics.f1_score(
                y, predicted_class, average=self.f1_average)
            local_scores[f'f1_{self.f1_average}'] = f1
            fbeta = sklearn.metrics.fbeta_score(
                y, predicted_class, average=self.f1_average, beta=self.f_beta)
            local_scores[f'f{self.f_beta}_{self.f1_average}'] = fbeta

            f1_smooth = MLSupervised.f1_score_smooth(
                y_true=y, y_pred=predictions, average=self.f1_average)
            local_scores[f'f1_smooth_{self.f1_average}'] = f1_smooth
            fbeta_smooth = MLSupervised.fbeta_score_smooth(
                y_true=y, y_pred=predictions, beta=self.f_beta,
                average=self.f1_average)
            local_scores[
                f'f{self.f_beta}_smooth_{self.f1_average}'] = fbeta_smooth
            
        # add to scores table (works fine when scores is None)
        #scores = scores.append(local_scores, ignore_index=True)
        if (scores is None) or scores.empty:
            scores =  pd.Series(local_scores).to_frame().T
        else:
            scores = pd.concat(
                [scores, pd.Series(local_scores).to_frame().T],
                ignore_index=True)
 
        return scores

    def save(self, path=None, epoch=None):
        """Pickles this object.

        The name of the pickle is specified by arg path, or use 
        self.save_path if arg path is None.

        If arg epoch is specified, suffix containing the epoch number is
        added just before the extension.

        ToDo: If the file base (the path after the directory and before 
        extension ends in '_n', substitutes n by n+1. This is important 
        when this method is called repeatedly, for example by parameter 
        search.

        Arguments:
          - path: pickle path 
          - epoch: current epoch
        """

        # model to cpu
        self.model.to(torch.device('cpu'))

        # find path
        if path is None:
            path = self.save_path

        # make dir and pickle
        dir_ = os.path.split(path)[0]
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        if epoch is not None:
            root, ext = os.path.splitext(path)
            path = root + f'_epoch-{epoch}' + ext
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)
        print(f'Classifier pickled as {path}')

        # bring back to original device
        self.model.to(self.device)

    @classmethod
    def load(cls, path):
        """Loads pickled instance of this class given by path.
        """

        # load
        with open(path, 'rb') as fd:
            obj = pickle.load(fd, encoding='latin1')
        print(f'Classifier loaded from {path}')

        # model to cuda if needed    
        if isinstance(obj.device, str) or  obj.device.type == 'cuda':
            obj.model = obj.model.to(obj.device)

        return obj

    def classification_report_extended(self, y, y_pred, y_pred_class):
        """Prints classification reports including the smooth results.

        Prints F_beta measures, where beta is defined as self.f_beta.

        Prints meassures for both hard classes and smooth (probability)
        classes.

        Arguments:
          y: (N samples) expected classes
          y_pred: (N samples x N classes) predicted class probabilities
          y_pred_class: (N samples) predicted hard classes
        """

        #print(os.linesep + 'Test classification report:',
        #      file=self._log_file)
        print(sk.metrics.classification_report(
            y_pred_class, y), file=self._log_file)
        f1_macro = MLSupervised.f1_score_smooth(
            y_true=y, y_pred=y_pred, average='macro')
        f1_weighted = MLSupervised.f1_score_smooth(
            y_true=y, y_pred=y_pred, average='weighted')
        print(f"   smooth macro {22*' '} {f1_macro:.2f}",
              file=self._log_file)
        print(f"smooth weighted {22*' '} {f1_weighted:.2f}",
              file=self._log_file)
        if self.f_beta != 1:
            fb_macro = MLSupervised.fbeta_score_smooth(
                y_true=y, y_pred=y_pred, average='macro',
                beta=self.f_beta)
            fb_weighted = MLSupervised.fbeta_score_smooth(
                y_true=y, y_pred=y_pred, average='weighted',
                beta=self.f_beta)
            print(
                f"   F{self.f_beta} smooth macro {19*' '} "
                + f"{fb_macro:.2f}",
                file=self._log_file)
            print(
                f"F{self.f_beta} smooth weighted {19*' '} "
                  + f"{fb_weighted:.2f}", file=self._log_file)
        print(os.linesep, file=self._log_file)
           
    def plot(self, fig=None, axs=None):
        """Plots results
        """

        if fig is None:
            fig, axs = plt.subplots(1, 4, figsize=(15, 3))
        
        for ax, label in zip(axs, [
                'loss', 'accuracy', 'f1_macro', 'f1_smooth_macro']):
            ax.scatter(
                self.train_scores['epoch'], self.train_scores[label],
                label='Train')
            ax.scatter(
                self.test_scores['epoch'], self.test_scores[label],
                label='Test')
            ax.set_title(label.capitalize())
        axs[0].legend()
        axs[0].set_ylim(0, axs[0].get_ylim()[1])
        axs[1].set_ylim(0, 1)
        axs[2].set_ylim(0, 1)  
        axs[3].set_ylim(0, 1)

        return fig, axs

    @classmethod
    def make_predictions(
            cls, nn_pkl, df, features, target=None, transformer_func=None,
            cm_print=True, report_print=True):
        """Neural net inference.

        Uses a trained network (arg nn_pkl) to make predicions. The input 
        data (arg df) is transformed using a function given by (arg)
        transformer_func.

        Essentially a wrapper around predict() that allows pandas input.
        
        Prints confusion matrix and classification report.

        Arguments:
          - nn_pkl: path to pickled instance of this class containing
          a trained network
          - df: (pandas.DataFrame) data table
          - features: columns of the data table that contain features
          - target: column of the data table containing target values
          (default None)
          - transformer_func: scikits transformer function
          - cm_print: flag inficating if confusion matrix is printed
          - report_print: flag inficating if classification report is printed

        Returns (pred, pred_class):
          - pred: (N samples x N classes) probabilistic predictions (only if
          probability=True)
          - pred_class: (N samples) hard-class predictions
        """

        X = df[features].to_numpy()
        if transformer_func is not None:
            X_scaled = transformer_func().fit_transform(X)
        else:
            X_scaled = X
            
        if target is not None:
            y = df[target].to_numpy()
        else:
            y = None

        net = cls.load(nn_pkl)
        y_prob, y_class = net.predict(
            X_scaled, y, cm_print=cm_print, report_print=report_print,
            probability=True)

        return y_prob, y_class

    @classmethod
    def save_predictions(
            cls, predict_df, y_predict_class, train_df, target, output_dir,
            nn_code, file_formats=['json']):
        """Saves table containing combined predictions and annotations.

        Table is converted and saved using pyto.io.PandasIO.write(). It
        can be read using pyto.io.PandasIO.read().

        Args output_dir, nn_code: together determine
        Path to the saved data (without extension) is determined as: 
          output_dir/nn_code
        
        Arguments:
          - predict_df: (pandas.DataFrame) data that was used as input
          to make predictions
          - y_predict_class: predicted classes
          - train_df: (pandas.DataFrame) data that was used for training
          - target: name of the column containing predicted data
          - output_dir: directory part of the path to the saved data
          - nn_code: code that defines the network, used as the base
          file name of the path to the saved data
          - file_formats: argument to pyto.io.PandasIO.write(), can be
          left at the default (['json'])
        """

        predict_df[target] = y_predict_class
        out_df = (pd
            .concat([predict_df, train_df], verify_integrity=True)
            .sort_index())
        out_path = os.path.join(output_dir, nn_code)
        PandasIO.write(
            out_df, out_path, file_formats=file_formats, 
            out_desc='predictions and annotations')

        return out_df
