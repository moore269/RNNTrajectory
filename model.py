"""
    model.py
    architecture decisions
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

from blocks import initialization
from blocks.bricks import Linear, Rectifier, NDimensionalSoftmax
#from NDimensionalSoftmax import *

from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.bricks.lookup import LookupTable
import theano.tensor as T
import theano.printing as printing
import numpy as np
import theano
import sys

#this file defines the functions that contain theano statements to build the neural network computation graph

#computes MRR for a given input stream
def RR_cost(y, y_hat, y_mask, constant1):
    # does required indexing into y_hat
    i0 = T.repeat(T.arange(y.shape[0]), y.shape[1]).flatten()
    i1 = T.tile(T.arange(y.shape[1]), y.shape[0]).flatten()
    i2 = y.flatten()

    # obtain unnormalized probability values for our class label
    y_hat_probs = T.reshape(y_hat[i0,i1,i2], y.shape)

    #grab all probabilities greater than classes probability
    s1 = T.ge(y_hat, y_hat_probs[:, :, np.newaxis])

    #Calculate Ranks by summing everything greater than class probability
    s4 = s1.sum(axis=-1)
    
    #obtain reciprocal ranks
    rr = constant1/T.cast(s4, theano.config.floatX)

    #now compute MRR and count y_mask in calculation
    cost_a=rr* y_mask 
    ymasksum = y_mask.sum()
    cost_int = cost_a.sum()

    return (cost_int, ymasksum)

#initialize theano bricks
def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()

#create softmax layer for probabilities
#this layer takes in masks for the input
def softmax_layer(h, y, x_mask, y_mask, lens, vocab_size, hidden_size, boosting):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size,
                              output_dim=vocab_size)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    softmax = NDimensionalSoftmax()

    #y_hat = softmax.apply(linear_output, extra_ndim=1)
    #y_hat.name = 'y_hat'
    cost_a = softmax.categorical_cross_entropy(
        y, linear_output, extra_ndim=1)
    #produces correct average
    cost_a=cost_a * y_mask 

    if boosting:
        #boosting step, must divide by length here
        lensMat = T.tile(lens, (y.shape[0], 1))
        cost_a = cost_a / lensMat

    #only count cost of correctly masked entries
    cost = cost_a.sum() / y_mask.sum()


    cost.name = 'cost'

    return (linear_output, cost)

#vanilla rnn layer
def rnn_layer(dim, h, n, x_mask, first, **kwargs):
    linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n))
    rnn = SimpleRecurrent(dim=dim, activation=Rectifier(), name='rnn' + str(n))
    initialize([linear, rnn])
    applyLin=linear.apply(h)
    if first:
        rnnApply = rnn.apply(applyLin, mask=x_mask, **kwargs)
    else:
        rnnApply = rnn.apply(applyLin, **kwargs)

    return rnnApply

#gru layer
def gru_layer(dim, h, n, x_mask, first, **kwargs):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    if first:
        gruApply = gru.apply(linear, gates, mask=x_mask, **kwargs)
    else:
        gruApply = gru.apply(linear, gates, **kwargs)
    return gruApply

#lstm layer
def lstm_layer(dim, h, n, x_mask, first, **kwargs):
    linear = Linear(input_dim=dim,  output_dim=dim * 4, name='linear' + str(n))
    lstm = LSTM(dim=dim, activation=Rectifier(), name='lstm' + str(n))
    initialize([linear, lstm])
    applyLin=linear.apply(h)

    if first:
        lstmApply = lstm.apply(applyLin, mask=x_mask, **kwargs)[0]
    else:
        lstmApply = lstm.apply(applyLin, **kwargs)[0]
    return lstmApply

#puts all layers together
def nn_fprop(x, x_mask,  y, y_mask, lens, vocab_size, hidden_size, num_layers, model, boosting=False, **kwargs):
    lookup = LookupTable(length=vocab_size, dim=hidden_size)
    initialize([lookup])
    h = lookup.apply(x)
    first=True
    for i in range(num_layers):
        if model == 'rnn':
            h = rnn_layer(hidden_size, h, i, x_mask=x_mask, first=first, **kwargs)
        elif model == 'gru':
            h = gru_layer(hidden_size, h, i, x_mask=x_mask, first=first, **kwargs)
        elif model == 'lstm':
            h = lstm_layer(hidden_size, h, i, x_mask=x_mask, first=first, **kwargs)
        else:
            print("models must either be rnn or lstm")
            sys.exit(0)
        first=False

    return softmax_layer(h, y, x_mask, y_mask, lens, vocab_size, hidden_size, boosting)
