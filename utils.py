"""
    utils.py
    Utilities
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
import h5py
import yaml
from fuel.datasets import H5PYDataset, IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme, ConstantScheme
from fuel.transformers import Mapping
from blocks.extensions import saveload, predicates, FinishAfter
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks import main_loop
from fuel.utils import do_not_pickle_attributes
from fuel.transformers import Padding, Unpack, Batch
import numpy as np
import cPickle as pickle
from theano import tensor
from theano import function
from model import nn_fprop
from IntStream import *
from theano import shared
from myCheckpoint import myCheckpoint
from FinishIfNoImprovementEpsilonAfter import *
from config import config

locals().update(config)

#Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []

# helper method for transposing
def transpose_stream(data):
    #return (data[0].T, data[1].T, data[2].T, data[3].T)
    return (data[0].T, data[1].T, data[2].T, data[3].T, data[4].T, data[5].T, data[6].T, data[7].T, data[8].T, data[9].T)

# track the best one
def track_best(channel, save_path, last_path, num_epochs, maxEpochs, maxIterations, epsilon, tempSharedData):
    tracker = TrackTheBest(channel)
    finishNoimprove = FinishIfNoImprovementEpsilonAfter('valid_MRR_best_so_far', epochs=num_epochs, epsilon=epsilon)
    
    lastcheckpoint = myCheckpoint(last_path, tempSharedData)


    checkpoint = myCheckpoint(save_path, tempSharedData, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"],
                             predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    finishAfter = FinishAfter(after_n_epochs=maxEpochs, after_n_batches =maxIterations)

    return [tracker, finishNoimprove, lastcheckpoint, checkpoint, finishAfter]



# metadata - ix_to_char, char_to_ix, and vocab_size
def get_metadata(pFile):
    meta_data=pickle.load(open(pFile+".p", 'r'))
    ix_to_char = meta_data['ix_to_char']
    char_to_ix = meta_data['char_to_ix']
    vocab_size = meta_data['vocab_size']
    return (ix_to_char, char_to_ix, vocab_size) 

# get stream obtains data from the make_dataset.py method
def get_stream(trainXY, batch_size=100):
    #trainXY=genSynXY()
    dataset_train=IndexableDataset(trainXY)
    stream_train_1 = DataStream(dataset=dataset_train, iteration_scheme=ShuffledScheme(examples=dataset_train.num_examples, batch_size=batch_size))
    stream_train_2 = Padding(stream_train_1)
    #stream_train_1.sources=('x_mask_o', 'y_mask_o', 'x', 'y')
    stream_train_3 = Mapping(stream_train_2, transpose_stream)


    return (stream_train_3, dataset_train.num_examples)

# get stream obtains data from the make_dataset.py method
def get_stream_inGPU(fName_train, sharedName='sharedData'):
    #trainXY=genSynXY()
    trainXYNP = np.load(fName_train+".npz")
    trainXY = {'x': trainXYNP['x'], 'x_mask_o': trainXYNP['x_mask'], 'y': trainXYNP['y'], 'y_mask_o': trainXYNP['y_mask'], 'lens': trainXYNP['lens']}

    train_stream, num_train_examples = get_stream(trainXY, sys.maxint)

    #convert to batch & get shapes
    train_batch = iterateShared(train_stream)
    shapes = getShapes(train_batch, num_train_examples)

    #convert to shared data
    sharedDataTrain={}
    for key in sorted(train_batch):
        sharedDataTrain[key] = shared(train_batch[key], name=sharedName+'_'+key)

    train_stream_int = IntStream(0, num_train_examples, batch_size, 'int_stream')

    return (sharedDataTrain, train_stream_int)

# get stream obtains data from the make_dataset.py method
def get_stream_inGPU_test(fName_test, sharedName, model):
    #testXY=genSynXY()
    trainXYNP = np.load(fName_test+".npz")
    trainXY = {'x': trainXYNP['x'], 'x_mask_o': trainXYNP['x_mask'], 'y': trainXYNP['y'], 'y_mask_o': trainXYNP['y_mask'], 'lens':trainXYNP['lens']}
    stream_list = get_split_streams(trainXY)
    retList = []
    for i, stream in enumerate(stream_list):
        test_stream, num_test_examples = get_stream(stream, sys.maxint)
        #convert to batch & get shapes
        test_batch = iterateShared(test_stream)
        shapes = getShapes(test_batch, num_test_examples)

        #if small, big batch size, if big, smaller batch size
        if i==0:
            test_stream_int = IntStream(0, num_test_examples, batch_size, 'int_stream')
        else:
            test_stream_int = IntStream(0, num_test_examples, batch_size/10, 'int_stream')

        retList.append((test_batch, test_stream_int))

    return retList

def switchStream(test_batch, model, sharedName):
    #replace shared variable with our data
    for i, key in enumerate(sorted(test_batch)):

        #search for var
        for var in model.variables:
            if var.name == sharedName+"_"+key:
                var.set_value(test_batch[key], borrow=False)
                break   


# get stream obtains data from the make_dataset.py method
def get_split_streams(trainXY, threshold=20):
    #trainXY=genSynXY()
    trainXYSeqLen = {'x': {}, 'x_mask_o': {}, 'y': {}, 'y_mask_o': {}, 'lens':{}}
    trainXYSmallSeq = {'x': [], 'x_mask_o': [], 'y': [], 'y_mask_o': [], 'lens': []}
    trainXYRestSeq = {}

    num_examples = trainXY['x'].shape[0]
    for i in range(0, num_examples):
        for key in trainXY:
            seqLen = trainXY[key][i].shape[0]
            seq = trainXY[key][i]
            if seqLen not in trainXYSeqLen[key]:
                trainXYSeqLen[key][seqLen] = [seq]
            else:
                trainXYSeqLen[key][seqLen].append(seq)
    
    #print out and instantiate rest object
    for seqLen in sorted(trainXYSeqLen['x']):
        seq = trainXYSeqLen['x'][seqLen]
        if seqLen >= threshold:
            trainXYRestSeq[seqLen] = {'x': [], 'x_mask_o': [], 'y': [], 'y_mask_o': [], 'lens':[]}
        print(str(seqLen) + ': '+ str(len(seq)))

    #separate into 1 small and big portions
    for key in trainXYSeqLen:
        for seqLen in trainXYSeqLen[key]:
            seq = trainXYSeqLen[key][seqLen]
            if seqLen < threshold:
                trainXYSmallSeq[key] = trainXYSmallSeq[key] + seq
            else:
                trainXYRestSeq[seqLen][key] = trainXYRestSeq[seqLen][key] + seq

    #now package into a list
    retList = [trainXYSmallSeq]
    for seqLen in trainXYRestSeq:
        retList.append(trainXYRestSeq[seqLen])

    #convert back to numpy array
    for stream in retList:
        for key in stream:
            stream[key] = np.array(stream[key])

    return retList

#helper function to return which axis contains the examples
def getShapes(train_all, num_examples):
    shapesRet = []
    for key in sorted(train_all):
        shapes = train_all[key].shape
        for i in range(0, len(shapes)):
            if shapes[i]==num_examples:
                shapesRet.append(i)
                break
    return shapesRet

def iterateShared(stream):
    epoch_iterator = (stream.get_epoch_iterator(as_dict=True))
    tShared = {}
    batch = next(epoch_iterator)
    for key in batch:
        tShared[key]=batch[key]
    return tShared



#generate synthetic data for testing purposes
def genSynXY():
    x=[]
    y=[]
    x_mask=[]
    y_mask=[]

    """x=np.array([[2,3,4], [1,2,3]], dtype='uint8')
    x_mask=np.array([[1,1,0], [1,1,1]], dtype='float')
    y=np.array([[3,4,5], [2,3,4]], dtype='uint8')
    y_mask=np.array([[1,0,0], [1,1,0]], dtype=theano.config.floatX)"""

    x.append(np.array([2,3,4], dtype='uint8'))
    x.append(np.array([1,2], dtype='uint8'))

    x_mask.append(np.array([1,1,0], dtype=theano.config.floatX))
    x_mask.append(np.array([1,1], dtype=theano.config.floatX))

    y.append(np.array([3,4,5], dtype='uint8'))
    y.append(np.array([2,3], dtype='uint8'))

    y_mask.append(np.array([1,0,0], dtype=theano.config.floatX))
    y_mask.append(np.array([1,1], dtype=theano.config.floatX))

    x=np.array(x)
    x_mask=np.array(x_mask)
    y=np.array(y)
    y_mask=np.array(y_mask)

    return {'x': x, 'y': y, 'x_mask_o': x_mask, 'y_mask_o': y_mask}

#training for testing purposes
def train():
    hidden_size=10
    num_layers=1
    rnn_type='rnn'
    train_stream = get_stream2("data/four_sq_trial_11_train_size_0.8_transitions_10000", 10000)
    #train_stream = get_stream("data/four_sq_trial_11_train_size_0.9_transitions_10000.hdf5", 'train', 10000)
    ix_to_char, char_to_ix, vocab_size = get_metadata("data/four_sq_trial_11_size_0.8_transitions_10000")
    x = tensor.matrix('x', dtype='uint8')
    x_mask = tensor.matrix('x_mask', dtype=theano.config.floatX)
    x_mask_o = tensor.matrix('x_mask_o', dtype=theano.config.floatX)
    x_mask_o_mask = tensor.matrix('x_mask_o_mask', dtype=theano.config.floatX)

    y = tensor.matrix('y', dtype='uint8')
    y_mask = tensor.matrix('y_mask', dtype=theano.config.floatX)
    y_mask_o = tensor.matrix('y_mask_o', dtype=theano.config.floatX)
    y_mask_o_mask = tensor.matrix('y_mask_o_mask', dtype=theano.config.floatX)

    x_mask_final=x_mask*x_mask_o*x_mask_o_mask
    y_mask_final=y_mask*y_mask_o*y_mask_o_mask

    y_hat, cost = nn_fprop(x, x_mask_final, y, y_mask_final, vocab_size, hidden_size, num_layers, rnn_type)

    f = function(inputs=[x, x_mask, x_mask_o, x_mask_o_mask, y, y_mask, y_mask_o, y_mask_o_mask], outputs=[y_hat, cost])
    epoch_iterator = (train_stream.get_epoch_iterator(as_dict=True))
    batch = next(epoch_iterator)
    output = f(batch['x'], batch['x_mask'], batch['x_mask_o'], batch['x_mask_o_mask'], batch['y'], batch['y_mask'], batch['y_mask_o'], batch['y_mask_o_mask'])
    """
    debug = function(inputs=[x, x_mask, x_mask_o, x_mask_o_mask, y, y_mask, y_mask_o, y_mask_o_mask], outputs=[y_hat, rrcost[0], rrcost[1], rrcost[2], rrcost[3], rrcost[4], rrcost[5], rrcost[6], rrcost[7], cost])
    epoch_iterator = train_stream.get_epoch_iterator(as_dict=True)
    while True:
        try:
            batch = next(epoch_iterator)
            output = debug(batch['x'], batch['x_mask'], batch['x_mask_o'], batch['x_mask_o_mask'], batch['y'], batch['y_mask'], batch['y_mask_o'], batch['y_mask_o_mask'])
            preds = output[0]
            s0 = output[1]
            s1 = output[2]
            s2 = output[3]
            s3 = output[4]
            s4 = output[5]
            rr = output[6]
            cost_a = output[7]
            cost = output[8]
            print('dummy')
        except StopIteration:
            break"""

    #output = f(batch['features'], batch['features_mask'], batch['targets'], batch['targets_mask'])
    for out in output:
        print(out)

if __name__ == '__main__':
    """a=genSynXY()

    streama=get_stream2("", "", 1)
    print(streama)
    print(streama.next_epoch())"""
    #ab = get_metadata("data/four_sq_trial_11_size_0.9_transitions_10000")
    train()
