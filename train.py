"""
    train.py
    Main training file
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
import sys
import argparse

import numpy as np
import theano
from theano import function
from theano import tensor as T
from theano import shared

from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar, predicates
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.monitoring import aggregation
from blocks.extensions import saveload
from blocks.extensions.training import TrackTheBest

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab
sys.setrecursionlimit(1000000)

from BlocksModules.myDataStreamMonitoring import *
from BlocksModules.FinishIfNoImprovementEpsilonAfter import *
from BlocksModules.myCheckpoint import myCheckpoint
from Utils.utils import get_metadata, get_stream, get_stream_inGPU, track_best, MainLoop
from Utils.make_dataset import *
from ModelDefinitions.model import nn_fprop, RR_cost
from config import config

# Load config parameters
locals().update(config)

# Main training function.  This function performs most of the legwork in training and saving the model.
def train(args, trial=11, no_valid=False):
    # Creating unique strings to save for experiments.
    data_valid = "data/"+args.data_name+"_trial_"+str(trial)+"_valid_size_"+str(args.train_size)+\
    "_transitions_"+str(args.transitions)
    data_test = data_valid.replace("_valid_size", "_test_size")
    # If we want validation set to match modData of test set
    if modDataValid==1:
        data_valid = data_valid.replace("_trial_", "_"+modData+"_trial_")
        data_test = data_test.replace("_trial_", "_"+modData+"_trial_")
    
    # By default, it is m0
    data_train = "data/"+args.data_name+"_trial_"+str(trial)+"_train_size_"+str(args.train_size)+\
    "_transitions_"+str(args.transitions)

    subStr = "rnn_type_"+args.rnn_type + "_trial_"+str(trial) + "_hiddenSize_"+str(args.hidden_size)+\
    "_numLayers_"+str(args.num_layers)+ \
    "_dropout_"+str(args.dropout)+"_train_size_"+str(args.train_size) + "_transitions_"+str(args.transitions)+\
    "_novalid_"+str(args.no_valid)

    if modData=="m1":
        data_train = data_train.replace("_trial_", "_m1_trial_")
        subStr = subStr.replace("_trial_", "_m1_trial_")
    elif modData=="m3":
        data_train = data_train.replace("_trial_", "_m3_trial_")
        subStr = subStr.replace("_trial_", "_m3_trial_")

        data_valid = "data/"+args.data_name+"_m3_trial_"+str(trial)+"_valid_size_"+str(args.train_size)+\
        "_transitions_"+str(args.transitions)
        data_test = "data/"+args.data_name+"_m3_trial_"+str(trial)+"_test_size_"+str(args.train_size)+\
        "_transitions_"+str(args.transitions)
    
    print("on test: "+subStr)
    # Perform folder prefixing
    prefix_path = models_folder + args.data_name + "/" + subStr +"_tgrad_"+str(args.truncate_gradient)+\
    "_boost_"+bStr(args.boosting)

    load_path2=prefix + load_path
    save_path2=prefix + save_path
    last_path2=prefix + last_path

    plots_output2 = plots_output + args.data_name + "/" + subStr +"_tgrad_"+str(args.truncate_gradient)+\
    "_boost_"+bStr(args.boosting)

    # obtain vocabulary size
    ix_to_char, char_to_ix, vocab_size = get_metadata(data_test.replace("_test", ""))
    print("vocab_size: " + str(vocab_size))

    # Get train, valid, test streams
    sharedDataTrain,  train_stream = get_stream_inGPU(data_train, sharedName='sharedData') 
    train_streamCopy = copy.deepcopy(train_stream)
    sharedDataValid, dev_stream = get_stream_inGPU(data_valid, sharedName='sharedData') 
    valid_streamCopy = copy.deepcopy(dev_stream)
    sharedDataTest, test_stream = get_stream_inGPU(data_test, sharedName='sharedData') 
    test_streamCopy = copy.deepcopy(test_stream)
    
    # Create dummy sums
    sharedMRRSUM = shared(np.array(0.0, dtype=theano.config.floatX))
    sharedTOTSUM = shared(np.array(0.0, dtype=theano.config.floatX))
    sharedSUMVARs = {'sharedMRRSUM': sharedMRRSUM, 'sharedTOTSUM': sharedTOTSUM}

    # Initialize batches
    batch_index_From = T.scalar('int_stream_From', dtype='int32')
    batch_index_To = T.scalar('int_stream_To', dtype='int32')

    # Index theano variables
    x = sharedDataTrain['x'][:,batch_index_From:batch_index_To]
    x.name = 'x'

    x_mask = sharedDataTrain['x_mask'][:,batch_index_From:batch_index_To]
    x_mask.name = 'x_mask'

    x_mask_o = sharedDataTrain['x_mask_o'][:,batch_index_From:batch_index_To]
    x_mask_o.name = 'x_mask_o'

    x_mask_o_mask = sharedDataTrain['x_mask_o_mask'][:,batch_index_From:batch_index_To]
    x_mask_o_mask.name = 'x_mask_o_mask'

    y = sharedDataTrain['y'][:,batch_index_From:batch_index_To]
    y.name = 'y'

    y_mask = sharedDataTrain['y_mask'][:,batch_index_From:batch_index_To]
    y_mask.name = 'y_mask'

    y_mask_o = sharedDataTrain['y_mask_o'][:,batch_index_From:batch_index_To]
    y_mask_o.name = 'y_mask_o'

    y_mask_o_mask = sharedDataTrain['y_mask_o_mask'][:,batch_index_From:batch_index_To]
    y_mask_o_mask.name = 'y_mask_o_mask'

    lens = sharedDataTrain['lens'][:, batch_index_From:batch_index_To]
    lens.name = 'lens'


    # Generate temp shared vars
    tempSharedData = {}
    tempSharedData[theano.config.floatX] = [shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ),
        shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) )]

    tempSharedData['uint8'] = [shared(np.array([[0], [0]], dtype='uint8') ), shared(np.array([[0], [0]], dtype='uint8')), shared(np.array([[0], [0]], dtype='uint8'))]

    # Final mask is due to the generated mask and the input mask
    x_mask_final=x_mask*x_mask_o*x_mask_o_mask
    y_mask_final=y_mask*y_mask_o*y_mask_o_mask

    # Build neural network
    linear_output, cost= nn_fprop(x, x_mask_final, y, y_mask_final, lens, vocab_size, hidden_size, num_layers, rnn_type, boosting=boosting, scan_kwargs={'truncate_gradient': truncate_gradient})

    # Keep a constant in gpu memory
    constant1 = shared(np.float32(1.0))
    cost_int, ymasksum = RR_cost(y, linear_output, y_mask_final, constant1)



    # Validation calculations
    fRR = function(inputs=[theano.In(batch_index_From, borrow=True), theano.In(batch_index_To, borrow=True)], 
        updates=[(sharedMRRSUM, sharedMRRSUM+cost_int ), (sharedTOTSUM, sharedTOTSUM+ymasksum)])

    # COST
    cg = ComputationGraph(cost)

    if dropout > 0:
        # Apply dropout only to the non-recurrent inputs (Zaremba et al. 2015)
        inputs = VariableFilter(theano_name_regex=r'.*apply_input.*')(cg.variables)
        cg = apply_dropout(cg, inputs, dropout)
        cost = cg.outputs[0]

    # Learning algorithm
    step_rules = [RMSProp(learning_rate=rmsPropLearnRate, decay_rate=decay_rate),
                  StepClipping(step_clipping)]
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=CompositeRule(step_rules))

    # Extensions

    # This is for tracking our best result
    trackbest = track_best('valid_MRR', save_path2, last_path2, num_epochs, nepochs, maxIterations, epsilon, tempSharedData)
    
    if onlyPlots:
        prefixes = ["train_cross", "valid_cross", "test_cross"]
        gradient_norm = aggregation.mean(algorithm.total_gradient_norm)
        step_norm = aggregation.mean(algorithm.total_step_norm)
        monitored_vars = [cost, gradient_norm, step_norm]
        #this is faster
        train_monitor = myTrainingDataMonitoring(variables = monitored_vars, prefix=prefixes[0], after_batch=True, saveEveryXIteration = saveEveryXIteration)
        #train_monitor = DataStreamMonitoringPlot(variables=[cost],
        #                    data_stream=train_streamCopy, prefix=prefixes[0], sharedDataTrain=sharedDataTrain, sharedDataActualTest=sharedDataTrain, after_batch=True, saveEveryXIteration = saveEveryXIteration)       
        valid_monitor = DataStreamMonitoringPlot(variables=[cost],
                            data_stream=valid_streamCopy, prefix=prefixes[1], sharedDataTrain=sharedDataTrain, sharedDataActualTest=sharedDataValid, after_batch=True, saveEveryXIteration = saveEveryXIteration)
        test_monitor = DataStreamMonitoringPlot(variables=[cost],
                            data_stream=test_streamCopy, prefix=prefixes[2], sharedDataTrain=sharedDataTrain, sharedDataActualTest=sharedDataTest, after_batch=True, saveEveryXIteration = saveEveryXIteration)
        trackbest = [trackbest[0], trackbest[2], trackbest[3], trackbest[4]]  
        plot = Plot('Live Plotting', saveFolder=plots_output2, channels=['train_cross_cost', 'valid_cross_cost', 'test_cross_cost'], numProcesses= numProcesses, saveEveryXIteration = saveEveryXIteration, after_batch=True)
        extensions =  [train_monitor, valid_monitor, test_monitor, plot,
            Printing(), ProgressBar(),
            ] +  trackbest
    else:
        dev_monitor = myDataStreamMonitoring(after_epoch=True, before_epoch=False,
                            data_stream=dev_stream, prefix="valid", fRR = fRR, sharedVars = sharedSUMVARs, sharedDataTrain=sharedDataTrain, sharedDataValid=sharedDataValid )
        extensions =  [dev_monitor,
            Printing(), ProgressBar(),
            ] +  trackbest

    if learning_rate_decay not in (0, 1):
        extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
                                                 lambda n, lr: np.cast[theano.config.floatX](learning_rate_decay * lr), after_epoch=True, after_batch=False))

    print 'number of parameters in the model: ' + str(T.sum([p.size for p in cg.parameters]).eval())
    # Finally build the main loop and train the model
    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                         model=Model(cost), extensions=extensions)
    main_loop.run()


def plotMat(plotData, name, saveFolder):
    for key in plotData:
        xVals = []
        yVals = []
        for i in range(0, len(plotData[key].data['x'])):
            if not np.isnan(plotData[key].data['y'][i]):
                xVal = plotData[key].data['x'][i]
                yVal = plotData[key].data['y'][i]
                xVals.append(xVal)
                yVals.append(yVal)
        pylab.plot(np.array(xVals), np.array(yVals), label=key)
        pylab.legend(loc=3, ncol=2)
        pylab.xlabel("iterations")
        pylab.ylabel("cost")
    pylab.title(name)
    pylab.savefig(saveFolder)

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run training for RNN Trajectory Project")

    parser.add_argument('--rnn-type', nargs='?', default='lstm',
                        help='Input rnn type. Default is lstm.')

    parser.add_argument('--hidden-size', nargs='?', type=int, default=10,
                        help='Hidden size for RNN. Default is 10.')

    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of input layers. Default is 1.')

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout percentage. Default is no dropout or 0.')

    parser.add_argument('--train-size', type=float, default=0.8,
                        help='Train/Test split. Default is 0.8 or 80\% training.')

    parser.add_argument('--transitions', type=int, default=1000,
                        help='Number of transitions to look at. By default first 1000.')

    parser.add_argument('--mod-data', default="m0", 
                      help='Modify the data. m0 is for regular rnn without modifications. \
                      m1 refers to cutting sequences into equal lengths (RNNC). \
                      m2 refers to taking largest sequences and analyzing just using it. \
                      m3 refers to cutting sequences such that first 20 (or cut size)\
                      is always available and masked. Default is m0.')

    parser.add_argument('--mod-data-valid', type=int, default=0,
                        help='Modify validation set similarly to test set. \
                        Depends on mod-data argument. Default is 0.')

    parser.add_argument('--data-name', default="yoo-choose",
                        help='Data file prefix. Default is yoo-choose.')

    parser.add_argument('--truncate_gradient', type=int, default=-1,
                        help='truncate gradient by how many steps. Default is -1.')
    
    parser.add_argument('--boosting', type=bool, default=False,
                        help='perform boosting objective function. Default is False.')
    
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='RMSprop learning rate. Default is 0.01.')   

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)



