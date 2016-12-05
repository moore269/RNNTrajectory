"""
    myDataStreamMonitoring.py
    Extensions for monitoring the training process.
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
 
import logging

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.algorithms import DifferentiableCostMinimizer
from blocks.monitoring.evaluators import AggregationBuffer, MonitoredQuantityBuffer, DatasetEvaluator
from myDatasetEvaluator import *
from multiprocessing import Pool

from blocks.monitoring.aggregation import MonitoredQuantity, take_last


import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import pylab
import time

PREFIX_SEPARATOR = '_'
logger = logging.getLogger(__name__)

import errno    
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def plotMat(params):
    index = params['saveFolder'].rfind("/")
    mkdir_p(params['saveFolder'][:index])
    for key in params['data']:
        x = np.array(params['data'][key]['x'])
        y = np.array(params['data'][key]['y'])
        pylab.plot(x, y, label=key)
        pylab.legend(loc=3, ncol=2)
        pylab.xlabel("iterations "+str(params['iteration']))
        pylab.ylabel("cost")
    pylab.title(params['title'])
    pylab.savefig(params['saveFolder']+"_iter_"+str(params['iteration'])+".png")
    pylab.close()


class myDataStreamMonitoring(SimpleExtension, MonitoringExtension):

    PREFIX_SEPARATOR = '_'

    def __init__(self, data_stream, fRR=None, sharedVars =None, sharedDataTrain=None, sharedDataValid=None, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(myDataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = myDatasetEvaluator(fRR, sharedVars, sharedDataTrain, sharedDataValid)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")

    def predictTest(self, data_stream, sharedDataTest):
        return self._evaluator.evaluateTest(data_stream, sharedDataTest)



class DataStreamMonitoringPlot(SimpleExtension, MonitoringExtension):

    def __init__(self, variables, data_stream, sharedDataTrain, sharedDataActualTest, updates=None, saveEveryXIteration=10, **kwargs):
        super(DataStreamMonitoringPlot, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream
        self.dataTrain = sharedDataTrain
        self.dataTest = sharedDataActualTest
        self.saveEveryXIteration = saveEveryXIteration
        self.curTime=time.time()

    def do(self, callback_name, *args):
        log = self.main_loop.log
        iteration = log.status['iterations_done']
        if iteration % self.saveEveryXIteration==0:    
            print("time took is : "+str(time.time()-self.curTime))
            self.switchData(self.dataTrain, self.dataTest)
            value_dict = self._evaluator.evaluate(self.data_stream)
            self.switchData(self.dataTrain, self.dataTest)
            self.add_records(self.main_loop.log, value_dict.items())


    #switch pointers between x and y
    def switchData(self, x, y):
        for key in x:
            temp = x[key].get_value(borrow=True)
            x[key].set_value(y[key].get_value(borrow=True), borrow=True)
            y[key].set_value(temp, borrow=True)

class myTrainingDataMonitoring(SimpleExtension, MonitoringExtension):

    def __init__(self, variables,saveEveryXIteration, **kwargs):
        self.saveEveryXIteration = saveEveryXIteration
        kwargs.setdefault("before_training", True)
        super(myTrainingDataMonitoring, self).__init__(**kwargs)
        self.add_condition(['after_batch'], arguments=('just_aggregate',))

        self._non_variables = []
        self._variables = []
        for variable_or_not in variables:
            if isinstance(variable_or_not, theano.Variable):
                self._variables.append(variable_or_not)
            elif isinstance(variable_or_not, MonitoredQuantity):
                self._non_variables.append(variable_or_not)
            else:
                raise ValueError("can not monitor {}".format(variable_or_not))

        self._non_variables = MonitoredQuantityBuffer(self._non_variables)
        self._required_for_non_variables = AggregationBuffer(
            [take_last(v) for v in self._non_variables.requires])
        self._variables = AggregationBuffer(
            self._variables, use_take_last=True)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        data, args = self.parse_args(callback_name, args)
        if callback_name == 'before_training':
            if not isinstance(self.main_loop.algorithm,
                              DifferentiableCostMinimizer):
                raise ValueError
            self.main_loop.algorithm.add_updates(
                self._variables.accumulation_updates)
            self.main_loop.algorithm.add_updates(
                self._required_for_non_variables.accumulation_updates)
            self._variables.initialize_aggregators()
            self._required_for_non_variables.initialize_aggregators()
            self._non_variables.initialize_quantities()
        else:
            log = self.main_loop.log
            iteration = log.status['iterations_done']
            if iteration % self.saveEveryXIteration==0:    
                # When called first time at any iterations, update
                # monitored non-Theano quantities
                if (self.main_loop.status['iterations_done'] >
                        self._last_time_called):
                    self._non_variables.aggregate_quantities(
                        list(self._required_for_non_variables
                             .get_aggregated_values().values()))
                    self._required_for_non_variables.initialize_aggregators()
                    self._last_time_called = (
                        self.main_loop.status['iterations_done'])
                # If only called to update non-Theano quantities,
                # do just that
                if args == ('just_aggregate',):
                    return
                # Otherwise, also output current values of from the accumulators
                # to the log.
                self.add_records(
                    self.main_loop.log,
                    self._variables.get_aggregated_values().items())
                self._variables.initialize_aggregators()
                self.add_records(
                    self.main_loop.log,
                    self._non_variables.get_aggregated_values().items())
                self._non_variables.initialize_quantities()



class Plot(SimpleExtension):

    def __init__(self, title, saveFolder, channels, numProcesses, saveEveryXIteration, **kwargs):
        self.saveEveryXIteration=saveEveryXIteration
        self.channels=set(channels)
        self.data={}
        self.pool = Pool(processes=numProcesses)
        self.params = {}
        self.params['saveFolder'] =saveFolder
        self.params['title'] = title

        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        self.iteration = log.status['iterations_done']
        for key, value in log.current_row.items():
            if key in self.channels:
                if key not in self.data:
                    self.data[key]={}
                    self.data[key]['x']=[]
                    self.data[key]['y']=[]
                self.data[key]['x'].append(self.iteration)
                self.data[key]['y'].append(value)
        #print('x: '+str(self.data[key]['x']))
        #print('y: '+str(self.data[key]['y']))
        if self.iteration % self.saveEveryXIteration == 0:
            self.params['data'] = self.data ; self.params['iteration'] = self.iteration ;
            self.pool.map_async(plotMat, [self.params])
            #plotMat(self.params)




