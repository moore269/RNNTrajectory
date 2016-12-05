"""
    make_dataset.py
    Change data inputs
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

import numpy as np
import codecs
import h5py
import yaml
from fuel.datasets import H5PYDataset
from config import config
from readData import *
from random import shuffle
import sys
import cPickle as pickle

# Load config parameters
locals().update(config)

# inOutList consists of
# x, x_mask, y, y_mask, all, all_mask
def convertNPY(seqs, seqs_mask, seqSplits, char_to_ix):
    inOutList=[]
    for key in seqs:
        seq = seqs[key]
        seq = [char_to_ix[seqEl] for seqEl in seq]
        seq_mask = seqs_mask[key]
        lenSeq=len(seq)   

        for i in xrange(0, lenSeq, seqSplits):
            lenTo=min(i+seqSplits, lenSeq-1)
            inSeq=seq[i:lenTo]
            inSeqMask = seq_mask[i:lenTo]
            outSeq=seq[i+1:lenTo+1]
            outSeqMask = seq_mask[i+1:lenTo+1]

            inOutList.append([np.array(inSeq, dtype='uint8'), np.array(inSeqMask, dtype=theano.config.floatX), np.array(outSeq, dtype='uint8'), 
                np.array(outSeqMask, dtype=theano.config.floatX), np.array(seq, dtype='uint8'), np.array(seq_mask, dtype=theano.config.floatX), [np.array(len(seq), dtype='uint8')]])
    #to insure randomness
    shuffle(inOutList)
    x=[]
    x_mask=[]
    y=[]
    y_mask=[]
    seq_all=[]
    seq_all_mask=[]
    lens=[]
    for element in inOutList:
        x.append(element[0])
        x_mask.append(element[1])
        y.append(element[2])
        y_mask.append(element[3])
        seq_all.append(element[4])
        seq_all_mask.append(element[5])
        lens.append(element[6])

    return (x, x_mask, y, y_mask, seq_all, seq_all_mask, lens)

# save the dataset files according to name (train, valid, or test)
def saveNPY(seqs, seqs_mask, fName, name, seqSplits, char_to_ix, ix_to_char, vocab_size, trial, train_size, numTransitions):
    x, x_mask, y, y_mask, seq_all, seq_all_mask, lens= convertNPY(seqs, seqs_mask, seqSplits, char_to_ix)
    fName1 = "data/"+fName+"_trial_"+str(trial)+"_"+name+"_size_"+str(train_size)+"_transitions_"+str(numTransitions)+".npz"
    print(fName1)
    np.savez(open(fName1, 'w'), x = x, x_mask = x_mask, y=y, y_mask = y_mask, seq_all = seq_all, seq_all_mask=seq_all_mask, lens=lens)

    #save meta data
    fName2 = "data/"+fName+"_trial_"+str(trial)+"_size_"+str(train_size)+"_transitions_"+str(numTransitions)
    data={'char_to_ix': char_to_ix, 'ix_to_char': ix_to_char, 'vocab_size': vocab_size}
    pickle.dump( data, open( fName2+".p", "wb" ) )

#prune out 0s (masked) entries and then return the index where the sequence actually starts
#this helps with memory issues
def prune0s(seq, seq_mask):
    lIndex=0
    newSeq = []
    for i, entry in enumerate(seq_mask):
        if entry!=0:
            lIndex=i
            break
    j=len(seq_mask)
    for entry in reversed(seq_mask):
        j+= -1
        if entry!=0:
            rIndex=j+1
            break
    #if there are all 0s
    if j < i:
        return (0, [], [])
    newSeq = seq[lIndex:rIndex]
    new_seq_mask = seq_mask[lIndex:rIndex]
    return (lIndex, newSeq, new_seq_mask)


#split into chunks according to maximum allowable size in GPU memory
#you can set this in config with maxSeqLen
def splitChunks(seqs, seqs_mask):
    new_seqs = {}
    new_seqs_mask = {}
    for key in seqs:
        seq=seqs[key]
        seq_mask = seqs_mask[key]
        lenSeq=len(seq)
        for i in xrange(0, lenSeq, maxSeqLen):
            lenTo=min(i+maxSeqLen, lenSeq)
            #get pruned candidate sequences
            lIndexRel, newSeq, new_seq_mask = prune0s(seq[i:lenTo], seq_mask[i:lenTo])
            lIndex = i + lIndexRel
            #if we essentially get no sequence we skip it
            if len(newSeq)==0:
                continue
            #get the recent entries to provide context
            seqPrefix = seq[max(lIndex-cutThreshold, 0):lIndex]
            new_seqs[key+"_"+str(i)] = seqPrefix + newSeq
            new_seqs_mask[key+"_"+str(i)] = [0]*len(seqPrefix) + new_seq_mask

    return (new_seqs, new_seqs_mask)
            
def testSplitChunks():
    seqs = {}
    seqs_mask = {}
    seqs['1'] = [str(x) for x in xrange(0, 100)]
    seqs_mask['1'] = [1]*100
    seqs, seqs_mask = splitChunks(seqs, seqs_mask)
    print(len(seqs['1_75'])==len(seqs_mask['1_75']))
     
# preprocesses data to include first numTransitions
# modData has the following options
# m0 is normal
# m1 is split transitions of size at most 10
# m2 is pick the largest sequence and train only on that to analyze exploding gradients
# seqSplits=max value means we take the whole sequence each time instead of partitioning into intervals of length seqSplits
def make_data(trial, fName, train_size, numTransitions, modData="m0", seqSplits=sys.maxint, cutoff=sys.maxint):
    np.random.seed(0)
    if modData=="m1":
        cutoff=cutThreshold
        fName2 = fName+"_m1"
    elif modData=="m0":
        fName2 = fName
    else:
        fName2 = fName+"_"+modData

    fileLoc = 'data/'+fName+'.dat'
    train, train_mask, valid, valid_mask, test, test_mask, all_seq = readSequencesRawFirstX(fileLoc,numTransitions=numTransitions, train_size=train_size, cutoff=cutoff)

    if modData=="m3":
        train, train_mask = splitChunks(train, train_mask)
        valid, valid_mask = splitChunks(valid, valid_mask)
        test, test_mask = splitChunks(test, test_mask)


    # maps sequence alphabet to integers and vice versa
    # vocab size refers to how many are in your sequence alphabet
    char_to_ix, ix_to_char = mapStrToNum(all_seq)
    vocab_size = len(char_to_ix)
    saveNPY(train, train_mask, fName2, 'train', seqSplits, char_to_ix, ix_to_char, vocab_size, trial, train_size, numTransitions)
    saveNPY(valid, valid_mask, fName2, 'valid', seqSplits, char_to_ix, ix_to_char, vocab_size, trial, train_size, numTransitions)
    saveNPY(test, test_mask, fName2, 'test', seqSplits, char_to_ix, ix_to_char, vocab_size, trial, train_size, numTransitions)

# create all data for transitions and for different train sizes
# 11 is default trial number that corresponds to the data ordered
def create_All_Data(fName, modData="m0"):
    train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    numTransitions=[10000, 20000, 30000, 40000]
    for train_size in train_sizes:
        print("on train size: "+str(train_size))
        for numTransition in numTransitions:
            print("on transition: "+str(numTransition))
            make_data(11, fName, train_size, numTransition, modData=modData)

def create_BIG_cutoff(fName, numTransitions, modData='m1'):
    make_data(11, fName, train_size, numTransitions, modData=modData)

# creating a sample dataset for testing purposes
def create_sample_data(fName):
    train_size=0.9
    numTransitions=10000
    make_data(11, fName, train_size, numTransitions, modData='m0')

if __name__ == '__main__':
    #testSplitChunks()
    name = sys.argv[1]
    #create_All_Data(name, "m0")
    create_All_Data(name, "m3")
    create_BIG_cutoff(name, 15000000, modData='m3')
    #create_BIG_cutoff(name, sys.maxint, modData='m1')
    #create_BIG_cutoff(name, sys.maxint, modData='m0')



