"""
    myDatasetEvaluator.py
    Extensions for evaluations.
    Copyright (C) 2016 John Moore

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 """


from collections import OrderedDict
import logging

from picklable_itertools.extras import equizip
import theano
from theano import tensor

from blocks.utils import dict_subset
from blocks.monitoring.aggregation import (_DataIndependent, Mean,
                                           TakeLast, MonitoredQuantity)
from blocks.graph import ComputationGraph
from blocks.utils import reraise_as
import numpy as np
import time

logger = logging.getLogger(__name__)


class myDatasetEvaluator(object):

    def __init__(self, fRR, sharedVars, sharedDataTrain, sharedDataValid):
        self.fRR = fRR
        self.sharedVars=sharedVars
        self.localShared = np.array(0.0, dtype=theano.config.floatX)
        self.sharedDataTrain = sharedDataTrain
        self.sharedDataValid = sharedDataValid

    #switch pointers between x and y
    def switchData(self, x, y):
        for key in x:
            temp = x[key].get_value(borrow=True)
            x[key].set_value(y[key].get_value(borrow=True), borrow=True)
            y[key].set_value(temp, borrow=True)

    def evaluate(self, data_stream):
        startTime = time.time()
        self.switchData(self.sharedDataTrain, self.sharedDataValid)
        for batch in data_stream.get_epoch_iterator(as_dict=True):
            self.fRR(batch['int_stream_From'], batch['int_stream_To'])
        self.switchData(self.sharedDataTrain, self.sharedDataValid)
        endTime = time.time()
        print("time: "+str(endTime-startTime))

        
        MRR = self.sharedVars['sharedMRRSUM'].get_value()/ self.sharedVars['sharedTOTSUM'].get_value()
        self.sharedVars['sharedMRRSUM'].set_value(self.localShared)
        self.sharedVars['sharedTOTSUM'].set_value(self.localShared)
        return {'MRR': 1-MRR}

    def evaluateTest(self, data_stream, sharedDataTest):
        startTime = time.time()
        self.switchData(self.sharedDataTrain, sharedDataTest)
        for batch in data_stream.get_epoch_iterator(as_dict=True):
            self.fRR(batch['int_stream_From'], batch['int_stream_To'])
        self.switchData(self.sharedDataTrain, sharedDataTest)
        endTime = time.time()
        print("time: "+str(endTime-startTime))


        MRR = self.sharedVars['sharedMRRSUM'].get_value()/ self.sharedVars['sharedTOTSUM'].get_value()
        self.sharedVars['sharedMRRSUM'].set_value(self.localShared)
        self.sharedVars['sharedTOTSUM'].set_value(self.localShared)

        return MRR


