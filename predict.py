"""
    predict.py
    make predictions after training
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
import os
import time
import argparse

import numpy as np
import theano
from theano import shared

from blocks.bricks import NDimensionalSoftmax
from blocks.extensions.saveload import Load

from Utilities.readData import *
from Utilities.utils import get_metadata, get_stream,get_stream_inGPU_test, track_best, MainLoop, switchStream
from ModelDefinitions.model import RR_cost

from config import config
# Load config parameters
locals().update(config)

# Perform evaluations based on Mean Reciprocal Rank
def evaluateREC(data_file, fRR, model, sharedSUMVARs, localShared, sharedName='sharedData', gpuData=True):
    # by default let's set batch size to 1 for ease of programming
    streamList = get_stream_inGPU_test(data_file, sharedName, model)
    
    streamStart = time.time()
    for streamOb in streamList:
        print("stream time: "+str(time.time() - streamStart))  
        streamStart=time.time()
        data = streamOb[0]
        stream = streamOb[1]
        switchStream(data, model, sharedName)
        
        batchStart=time.time()
        for i, batch in enumerate(stream.get_epoch_iterator(as_dict=True)):
            print(str(i) + " time: "+str(time.time() - batchStart))
            batchStart=time.time()

            bFrom = batch['int_stream_From']
            bTo = batch['int_stream_To']
            fRR(bFrom, bTo)

    recRank = sharedSUMVARs['sharedMRRSUM'].get_value()
    totalPreds = sharedSUMVARs['sharedTOTSUM'].get_value()
    sharedSUMVARs['sharedMRRSUM'].set_value(localShared)
    sharedSUMVARs['sharedTOTSUM'].set_value(localShared)

    print("UnNorm RecRank: "+str(recRank))
    print("totalPreds: "+str(totalPreds))
    print("RecRank: "+str(recRank/totalPreds))
    return recRank/totalPreds

# Load appropriate model file
def evaluation(model_file_path, data_name='', modData="m3", gpuData=True, mTest=False):
    trainInd1=model_file_path.find("_train_size_")+len("_train_size_")
    trainInd2=model_file_path.find("_", trainInd1)
    train_size=float(model_file_path[trainInd1:trainInd2])
    transInd1=model_file_path.find("_transitions_")+len("_transitions_")
    transInd2=model_file_path.find("_", transInd1)
    transitions=int(model_file_path[transInd1:transInd2])
    transInd1=model_file_path.find("_trial_")+len("_trial_")
    transInd2=model_file_path.find("_", transInd1)
    trial=int(model_file_path[transInd1:transInd2])

    mPrefix =""
    if "_m1_" in model_file_path and mTest:
        mPrefix="_m1"
    elif modData=="m3":
        data_train = "data/"+data_name+"_m3_trial_"+str(trial)+"_train_size_"+str(train_size)+"_transitions_"+str(transitions)
        data_valid = "data/"+data_name+"_m3_trial_"+str(trial)+"_valid_size_"+str(train_size)+"_transitions_"+str(transitions)
        data_test = "data/"+data_name+"_m3_trial_"+str(trial)+"_test_size_"+str(train_size)+"_transitions_"+str(transitions)

    #ix_to_char, char_to_ix, vocab_size = get_metadata(data_train.replace("_train", ""))
    data_train = "data/"+data_name+mPrefix+"_trial_"+str(trial)+"_train_size_"+str(train_size)+"_transitions_"+str(transitions)
    data_valid = "data/"+data_name+mPrefix+"_trial_"+str(trial)+"_valid_size_"+str(train_size)+"_transitions_"+str(transitions)
    data_test = "data/"+data_name+mPrefix+"_trial_"+str(trial)+"_test_size_"+str(train_size)+"_transitions_"+str(transitions)

    print 'Loading model from {0}...'.format(model_file_path)
    main_loop = Load(model_file_path)
    #get validation cost
    print 'Model loaded. Building prediction function...'
    model = main_loop.model
    if gpuData:
        batch_index_To, batch_index_From = model.inputs
    else:
        y_mask, y_mask_o, y, x, x_mask, x_mask_o, y_mask_o_mask, x_mask_o_mask = model.inputs

    for var in model.variables:
        if var.name=='linear_output':
            linear_output = var
        if var.name=='y':
            y = var
        if var.name =='y_mask':
            y_mask = var
        if var.name =='y_mask_o':
            y_mask_o = var
        if var.name =='y_mask_o_mask':
            y_mask_o_mask = var

    sharedMRRSUM = shared(np.array(0.0, dtype=theano.config.floatX))
    sharedTOTSUM = shared(np.array(0.0, dtype=theano.config.floatX))
    sharedSUMVARs = {'sharedMRRSUM': sharedMRRSUM, 'sharedTOTSUM': sharedTOTSUM}

    y_mask_final=y_mask*y_mask_o*y_mask_o_mask
    constant1 = shared(np.float32(1.0))
    cost_int, ymasksum = RR_cost(y, linear_output, y_mask_final,  constant1)

    #validation calculations
    fRR = theano.function(inputs=[theano.In(batch_index_From, borrow=True), theano.In(batch_index_To, borrow=True)], 
        updates=[(sharedMRRSUM, sharedMRRSUM+cost_int ), (sharedTOTSUM, sharedTOTSUM+ymasksum)])
    localShared = np.array(0.0, dtype=theano.config.floatX)

    return (evaluateREC(data_train, fRR, model, sharedSUMVARs, localShared), evaluateREC(data_valid, fRR, model, sharedSUMVARs, localShared), evaluateREC(data_test, fRR, model, sharedSUMVARs, localShared))

# pick the best model according to validation measure
def pickBestModel(trial, models_folder, transitions, rnnType='lstm', data_name=''):
    mFiles = os.listdir(models_folder)
    print("in "+models_folder)
    minValCost=10000000.0
    best_model=None
    best_model_file=""
    trial = "trial_"+str(trial)
    for mFile in mFiles:
        if "_novalid_" in mFile and ".pkl" in mFile and trial in mFile and "_transitions_"+str(transitions) in mFile and "rnn_type_"+rnnType in mFile and "train_size_0.8" in mFile:
            print(mFile)
            main_loop = load(models_folder+mFile)
            evaluation(modelFolder+mFile, data_name=data_name)
            if 'dev_cost' in main_loop.log[len(main_loop.log)-1]:
                cur_cost=main_loop.log[len(main_loop.log)-1]['dev_cost'].tolist()
            else:
                cur_cost=main_loop.log[len(main_loop.log)-1]['valid_cost'].tolist()
            if cur_cost < minValCost:
                minValCost=cur_cost
                best_model=main_loop
                best_model_file=mFile
    print(best_model_file)
    return best_model_file

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run training for RNN Trajectory Project")

    parser.add_argument('--hidden-size', nargs='?', type=int, default=10,
                        help='Hidden size for RNN. Default is 10.')

    parser.add_argument('--data-name', default="yoo-choose",
                        help='Data file prefix. Default is yoo-choose.')
  

    return parser.parse_args()

# change model_file_path according to needs
if __name__ == '__main__':
    args = parse_args()
    modelFolder="models/"+args.data_name+"/"
    evaluation(modelFolder+"rnn_type_LSTM_trial_11_hiddenSize_10_numLayers_1_dropout_0.0_train_size_0.8_transitions_10000_novalid_False_tgrad_1_boost_F_best.pkl", data_name=args.data_name)



