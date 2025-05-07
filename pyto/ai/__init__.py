"""
Tools for machine and deep learning

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

from . import feature_transform_tethers
from .ml_supervised import MLSupervised
try:
    from . import pytorch_plus
    from .full_connect import FullConnect
    from .features_dataset_nn import FeaturesDatasetNN
    from .classification_nn import ClassificationNN
except ModuleNotFoundError:
    try:
        torch_missing = False
        torch
    except NameError:
        torch_missing = True
    if torch_missing:
        print("Info: Neural net functionality cannot be used because module "
              + "torch was not found.")
    else:
        raise

