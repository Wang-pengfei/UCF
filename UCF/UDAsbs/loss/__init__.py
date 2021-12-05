from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import SoftEntropy, CrossEntropyLabelSmooth_s, CrossEntropyLabelSmooth_c
from .multisoftmax import MultiSoftmaxLoss
from .invariance import InvNet
__all__ = [
    'TripletLoss',
    'SoftTripletLoss',
    'SoftEntropy',
    'MultiSoftmaxLoss',
    'InvNet',
    'CrossEntropyLabelSmooth_s',
    'CrossEntropyLabelSmooth_c'
]
