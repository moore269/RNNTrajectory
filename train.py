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

import theano
import numpy as np
from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import StepClipping, GradientDescent, CompositeRule, RMSProp
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar, predicates
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring
from myDataStreamMonitoring import *
from blocks.monitoring import aggregation
from blocks.extensions import saveload
from blocks.extensions.training import TrackTheBest
from utils import get_metadata, get_stream, get_stream_inGPU, track_best, MainLoop
from model import nn_fprop, RR_cost
from config import config
import sys
from make_dataset import *
from theano import function
from theano import tensor as T
from FinishIfNoImprovementEpsilonAfter import *
from theano import shared
from myCheckpoint import myCheckpoint
#from blocks_extras.extensions.plot import Plot 
#bokehsavefrom bokeh.io import save as 
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import pylab
sys.setrecursionlimit(1000000)

# Load config parameters
locals().update(config)

def bStr(boolStr):
    if boolStr:
        return "T"
    else:
        return "F"

# m0 is for regular rnn without modifications
# m1 refers to cutting sequences into equal lengths (RNNC)
# m2 refers to taking largest sequences and analyzing just using it
# m3 refers to cutting sequences such that first 20 (or cut size)
# is always available and masked
def train(rnn_type='lstm', hidden_size=10, num_layers=1, trial=11, dropout=0, train_size=0.8, transitions=10000, modData="m0", modDataValid=0, no_valid=False, data_name="yoo-choose", gpuData=True, truncate_gradient=-1, boosting=False, rmsPropLearnRate = 0.01):
    data_valid = "data/"+data_name+"_trial_"+str(trial)+"_valid_size_"+str(train_size)+"_transitions_"+str(transitions)
    data_test = "data/"+data_name+"_trial_"+str(trial)+"_test_size_"+str(train_size)+"_transitions_"+str(transitions)
    if modDataValid==1:
        data_valid = data_valid.replace("_trial_", "_"+modData+"_trial_")
        data_test = data_test.replace("_trial_", "_"+modData+"_trial_")

    if modData=="m1":
        data_train = "data/"+data_name+"_m1_trial_"+str(trial)+"_train_size_"+str(train_size)+"_transitions_"+str(transitions)
        subStr = "rnn_type_"+rnn_type + "_m1_trial_"+str(trial) + "_hiddenSize_"+str(hidden_size)+"_numLayers_"+str(num_layers)+"_dropout_"+str(dropout)+"_train_size_"+str(train_size) + "_transitions_"+str(transitions)+"_novalid_"+str(no_valid)
    elif modData=="m3":
        data_train = "data/"+data_name+"_m3_trial_"+str(trial)+"_train_size_"+str(train_size)+"_transitions_"+str(transitions)
        subStr = "rnn_type_"+rnn_type + "_m3_trial_"+str(trial) + "_hiddenSize_"+str(hidden_size)+"_numLayers_"+str(num_layers)+"_dropout_"+str(dropout)+"_train_size_"+str(train_size) + "_transitions_"+str(transitions)+"_novalid_"+str(no_valid)
        data_valid = "data/"+data_name+"_m3_trial_"+str(trial)+"_valid_size_"+str(train_size)+"_transitions_"+str(transitions)
        data_test = "data/"+data_name+"_m3_trial_"+str(trial)+"_test_size_"+str(train_size)+"_transitions_"+str(transitions)
    else: 
        data_train = "data/"+data_name+"_trial_"+str(trial)+"_train_size_"+str(train_size)+"_transitions_"+str(transitions)
        subStr = "rnn_type_"+rnn_type + "_trial_"+str(trial) + "_hiddenSize_"+str(hidden_size)+"_numLayers_"+str(num_layers)+"_dropout_"+str(dropout)+"_train_size_"+str(train_size) + "_transitions_"+str(transitions)+"_novalid_"+str(no_valid)
    
    print("on test: "+subStr)
    load_path2=models_folder + data_name + "/" + subStr +"_tgrad_"+str(truncate_gradient)+"_boost_"+bStr(boosting)+ load_path
    save_path2=models_folder + data_name + "/" + subStr +"_tgrad_"+str(truncate_gradient)+"_boost_"+bStr(boosting)+ save_path
    last_path2=models_folder + data_name + "/" + subStr +"_tgrad_"+str(truncate_gradient)+"_boost_"+bStr(boosting)+ last_path
    plots_output2 = plots_output + data_name + "/" + subStr +"_tgrad_"+str(truncate_gradient)+"_boost_"+bStr(boosting)

    #
    ix_to_char, char_to_ix, vocab_size = get_metadata(data_test.replace("_test", ""))
    print("vocab_size: " + str(vocab_size))
    if gpuData:
        sharedDataTrain,  train_stream = get_stream_inGPU(data_train, sharedName='sharedData') 
        train_streamCopy = copy.deepcopy(train_stream)
        sharedDataValid, dev_stream = get_stream_inGPU(data_valid, sharedName='sharedData') 
        valid_streamCopy = copy.deepcopy(dev_stream)
        sharedDataTest, test_stream = get_stream_inGPU(data_test, sharedName='sharedData') 
        test_streamCopy = copy.deepcopy(test_stream)

        sharedMRRSUM = shared(np.array(0.0, dtype=theano.config.floatX))
        sharedTOTSUM = shared(np.array(0.0, dtype=theano.config.floatX))
        sharedSUMVARs = {'sharedMRRSUM': sharedMRRSUM, 'sharedTOTSUM': sharedTOTSUM}

        batch_index_From = T.scalar('int_stream_From', dtype='int32')
        batch_index_To = T.scalar('int_stream_To', dtype='int32')

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


        #generate temp shared vars
        tempSharedData = {}
        tempSharedData[theano.config.floatX] = [shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ),
            shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) ), shared(np.array([[0], [0]], dtype=theano.config.floatX) )]

        tempSharedData['uint8'] = [shared(np.array([[0], [0]], dtype='uint8') ), shared(np.array([[0], [0]], dtype='uint8')), shared(np.array([[0], [0]], dtype='uint8'))]
    else:
        train_stream, train_examples = get_stream(data_train, batch_size) 
        if no_valid:
            dev_stream, dev_examples=get_stream(data_train, batch_size)
        else:
            dev_stream, dev_examples = get_stream(data_valid, batch_size)
        # New Model
        x = T.matrix('x', dtype='uint8')
        x_mask = T.matrix('x_mask', dtype=theano.config.floatX)
        x_mask_o = T.matrix('x_mask_o', dtype=theano.config.floatX)
        x_mask_o_mask = T.matrix('x_mask_o_mask', dtype=theano.config.floatX)

        y = T.matrix('y', dtype='uint8')
        y_mask = T.matrix('y_mask', dtype=theano.config.floatX)
        y_mask_o = T.matrix('y_mask_o', dtype=theano.config.floatX)
        y_mask_o_mask = T.matrix('y_mask_o_mask', dtype=theano.config.floatX)

    #final mask is due to the generated mask and the input mask
    x_mask_final=x_mask*x_mask_o*x_mask_o_mask
    y_mask_final=y_mask*y_mask_o*y_mask_o_mask

    #build neural network
    linear_output, cost= nn_fprop(x, x_mask_final, y, y_mask_final, lens, vocab_size, hidden_size, num_layers, rnn_type, boosting=boosting, scan_kwargs={'truncate_gradient': truncate_gradient})

    #keep a constant in gpu memory
    constant1 = shared(np.float32(1.0))
    cost_int, ymasksum = RR_cost(y, linear_output, y_mask_final, constant1)


    #debug = function(inputs=[batch_index_From, batch_index_To], outputs=[lens])
    #epoch_iterator = (train_stream.get_epoch_iterator(as_dict=True))
    #batch = next(epoch_iterator)
    #print(batch.keys())
    #output = debug(batch['int_stream_From'], batch['int_stream_To'])
    #print('here')
    #validation calculations
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

    #this is for tracking our best result
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

    #trainMRR = dev_monitor.predictTest(train_stream, sharedDataTrain)
    #validMRR = dev_monitor.predictTest(dev_stream, sharedDataValid)
    
    #testMRR = dev_monitor.predictTest(test_stream, sharedDataTest)
    #results = np.array([trainMRR, validMRR, testMRR])
    #np.save(save_path2.replace(".pkl", ".npy"), results)

    #print("train: "+str(trainMRR))
    #print("valid: "+str(validMRR))
    #print("test: "+str(testMRR))


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

#sys.maxint = 9223372036854775807
if __name__ == '__main__':
    #make_trial_data()
    #experiment with hidden sizes
    #python train.py [hidden_size] [num_layers] [validation_size]
    rnn_type=sys.argv[1]
    hidden_size=int(sys.argv[2])
    num_layers=int(sys.argv[3])
    dropout=float(sys.argv[4])
    train_size=float(sys.argv[5])
    transitions=int(sys.argv[6])
    modData=sys.argv[7]
    modDataValid=int(sys.argv[8])
    data_name=sys.argv[9]
    truncate_gradient=int(sys.argv[10])
    boosting = True if not int(sys.argv[11])==0 else False
    rmsPropLearnRate = float(sys.argv[12])

    train(rnn_type=rnn_type, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, 
        train_size=train_size, transitions=transitions, trial=11, modData=modData, no_valid=False, 
        data_name=data_name, modDataValid=modDataValid, truncate_gradient=truncate_gradient, boosting=boosting, 
        rmsPropLearnRate = rmsPropLearnRate)

    #test_vary_hidden(1, 2, num_layers=1)
    #test_vary_hidden(4, 7, num_layers=1)
    #test_vary_hidden(7, 10, num_layers=1)
    
    #test_vary_hidden(13, 14, num_layers=1)
    #test_vary_hidden(14, 15, num_layers=1)
    #test_vary_hidden(15, 16, num_layers=1)
    #test_vary_hidden(16, 17, num_layers=1)
    #test_vary_hidden(17, 18, num_layers=1)
    #test_vary_hidden(19, 20, num_layers=1)
    #test_vary_hidden(20, 21, num_layers=1)
    #test_vary_hidden(21, 22, num_layers=1)





    #test_vary_hidden(1, 4, num_layers=1)
    #test_vary_hidden(4, 7, num_layers=1)
    #test_vary_hidden(7, 10, num_layers=1)
    #test_vary_hidden(10, 13, num_layers=1) 

    #test_vary_dropout(1, 3)
    #test_vary_dropout(3, 5)
    #test_vary_dropout(5, 7)
    #test_vary_dropout(7, 10)

