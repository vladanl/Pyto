"""
Pytorch deep net based on fully connected layers

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"
 

from collections import OrderedDict
import inspect

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FullConnect(nn.Module):
    
    def __init__(
            self, n_features, activation='relu', batch_norm=False,
            dropout=None, init=None, last_layer=None, last_l1=False):
        """Saves arguments (params) and makes a net"""
        
        super().__init__()

        # save arguments as attributes
        ignore = []
        self.save_args(ignore=ignore)

        # save these directly because, save_args() doesn't save those that
        # are instances of activation functions
        self.set_active_f(activation)
        self.set_last_layer_f(last_layer)
        # these don't work either
        #self.activation = activation
        #self.last_layer = last_layer
        
        # make net based on params (attributes)
        self.make_net()

    def save_args(self, ignore=[]):
        """Saves calling function (meant for __init__) arguments as attributes.

        Problem: It does not set arguments that are instantiated torch
        activation classes.
        Reason: Has to do with this class inheriting from torch.nn.Module
        and possibly these are torch.nn.Module instances
        Workaround: Set these by calling corresponding functions directly
        """
        frame = inspect.currentframe().f_back
        args, _, _, local_vars = inspect.getargvalues(frame)
        [setattr(self, name, local_vars[name]) for name in args
         if name not in ['self'] + ignore]
            
    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        """Saves activation (self.activation) and sets activation function
        (self._active_f)

        Activation (arg value) can be specified in two ways:
          - String: 'relu', 'elu', or 'leaky_rely'
          - Activation class: nn.ReLU, ...
        """
        #if isinstance(value, str):
        #    self._activation = value
        #else:
        #    self._activation = value.__name__
        self._activation = value
        self.set_active_f(activation=value)

    def set_active_f(self, activation):
        """ Sets self._active_f function from argument
        """

        if isinstance(activation, str):
            if activation == 'relu':
                self._active_f = nn.ReLU()
            elif activation == 'leaky_relu':
                self._active_f = nn.LeakyReLU()
            elif activation == 'elu':
                self._active_f = nn.ELU()
            else:
                raise ValueError(
                    f"Sorry, activation {activation} is not recognized. "
                    + "Acceptable values are 'relu', 'leaky_relu', 'elu' "
                    + "activation classes and instances of activation classes")
        elif inspect.isclass(activation):
            self._active_f = activation()
        else:
            self._active_f = activation  # instance of activation class

    @property
    def last_layer(self):
        return self._last_layer

    @last_layer.setter
    def last_layer(self, value):
        """Sets last_layer function

        Argument:
          - value: 'sofmax', 'sigmoid' or a model layer function
        """

        self._last_layer = value
        self.set_last_layer_f(value=value)

    def set_last_layer_f(self, value):
        """Sets self._last_layer_f
        """
        
        if isinstance(value, str):
            if value == 'softmax':
                self._last_layer_f = nn.Softmax(dim=1)
            elif value == 'sigmoid':
                self._last_layer_f = nn.Sigmoid()
        elif inspect.isclass(value):
            self._last_layer_f = value()
        else:
            self._last_layer_f = value  # function (instance of a layer class)
            
    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, value):
        self._init = value
        self.set_init_f(init=value)
            
    def set_init_f(self, init):
        """ Saves init (self.init) and sets init function (self._init_f)

        Argument init can be specified in two forms:
          - String: 'kaiming_normal', or 'kaiming_uniform'
          - Function, such as torch.nn.init.kaiming_uniform_
        """
        if init is None:
            self._init_f = None
        elif not isinstance(init, str):
            self._init_f = init
            #self._init = init.__name__
        elif init == 'kaiming_normal':
            self._init_f = nn.init.kaiming_normal_
        elif init == 'kaiming_uniform':
            self._init_f = nn.init.kaiming_uniform_
        else:
            raise ValueError(
                "Valid values for argument init are "
                + "None, 'kaiming_normal' and 'kaiming_uniform', as well as "
                + "all appropriate nn.init functions.")        
        
    def set_params(self, **params):
        """Sets specified parameters.

        If any param is specified, makes a new model (self.model).
        """

        update = False
        for name, value in params.items():
            setattr(self, name, value)
            update = True
        if update:
            self.make_net()
            
    def basic_net(self, dropout):
        """Depreciated"""
            
        mods = OrderedDict()
        n_layers = len(self.n_features) - 1

        # make layer stack
        for ind in range(n_layers-1):
            mods[f'fc_{ind}'] = nn.Linear(
                self.n_features[ind], self.n_features[ind+1])
            mods[f'{self.activation}_{ind}'] = self._active_f
            if dropout is not None:
                mods[f'drop_{ind}'] = nn.Dropout(dropout[ind])
        mods[f'fc_{n_layers-1}'] = nn.Linear(
            self.n_features[n_layers-1], self.n_features[n_layers])
        self.stack = nn.Sequential(mods)

        # initialize weights
        self.init_weights()
       
    def make_net(self):
        """Generates net using current parameters and saves it as self.stack"""

        mods = OrderedDict()
        n_layers = len(self.n_features) - 1

        if (self.dropout is not None
            and not isinstance(self.dropout, (list, tuple))):
            dropout = (n_layers - 1) * [self.dropout]

        # make layer stack
        for ind in range(n_layers-1):
            mods[f'fc_{ind}'] = nn.Linear(
                self.n_features[ind], self.n_features[ind+1])
            if self.batch_norm:
                mods[f'bn_{ind}'] = nn.BatchNorm1d(
                    num_features=self.n_features[ind+1])
            mods[f'activation_{ind}'] = self._active_f
            if self.dropout is not None:
                mods[f'drop_{ind}'] = nn.Dropout(dropout[ind])
        mods[f'fc_{n_layers-1}'] = nn.Linear(
            self.n_features[n_layers-1], self.n_features[n_layers])
        if self._last_layer_f is not None:
            mods[f'last'] = self._last_layer_f
        self.stack = nn.Sequential(mods)

        # initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initializes linear layer weights
        """

        if self._init_f is not None:

            # He initializations
            if self._init_f.__name__.startswith('kaiming'):

                # pytorch docs recommend nonlinearity param only for relu*
                # activations
                if (self.activation == 'relu') \
                   or (self.activation == 'leaky_relu'):
                    nonlinearity = self.activation
                else:
                    nonlinearity = 'leaky_relu'

                for layer in self.stack:
                    if isinstance(layer, torch.nn.Linear):
                        self._init_f(layer.weight, nonlinearity=nonlinearity)

            else:
                raise ValueError(
                    "Sorry, only Kaiming (He) weight initializations "
                    + "are implemented.")        
    
    def forward(self, x):
 
        logits = self.stack(x)
        if self.last_l1:
            logits = logits / logits.sum(dim=1).unsqueeze(1)
            
        return logits
        
