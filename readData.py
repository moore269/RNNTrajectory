"""
    readData.py
    read given data
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
from collections import defaultdict
from operator import itemgetter
import random
import numpy as np
from operator import add
import theano
from random import shuffle
import copy

#read four square data
#data format: <time interval from last transition> <user ID> <from> <to>
def readLineFS(line, numTransCount, sequencesHash, cutoff, splitter="\t", idIndex=1, catIndex1=2, catIndex2=3):
    vals=line.replace("\n", "").split(splitter)
    sesID = vals[idIndex]

    #From: 2, To: 3
    cat1 = vals[catIndex1]
    cat2 = vals[catIndex2]
    
    # if it's greater than a cutoff point
    # don't include it
    if sesID in sequencesHash:
        if sequencesHash[sesID]>=cutoff:
            return (sesID, [], sequencesHash, numTransCount)

    # we count a transition no matter what since the data format is in the form of transitions
    numTransCount+=1
    #want to return 2 if it is the first we've seen
    if sesID in sequencesHash:
        sequencesHash[sesID]+=1
        return (sesID, [cat2], sequencesHash, numTransCount)
    else:
        sequencesHash[sesID]+=2
        return (sesID, [cat1, cat2], sequencesHash, numTransCount)


#function for reading a line of the yoochoose dataset file
def readLineYC(line, numTransCount, sequencesHash, cutoff, splitter=",", idIndex=0, catIndex1=2, catIndex2=-1):
    vals=line.replace("\n", "").split(splitter)
    sesID=vals[idIndex]
    category=vals[catIndex1]

    # if it's greater than a cutoff point
    # don't include it
    if sesID in sequencesHash:
        if sequencesHash[sesID]>=cutoff:
            return (sesID, [], sequencesHash, numTransCount)

    # if the last ID is equivalent, we up numTransCount
    # we can only iterate numTransCount if we've seen it in the past in this data format
    if sesID in sequencesHash:
        numTransCount+=1

    sequencesHash[sesID]+=1
    return (sesID, [category], sequencesHash, numTransCount)


#we want to prune out sequences of length 1
def pruneTransitions(sequences):
    keys = sequences.keys()
    for key in keys:
        if len(sequences[key])<=1:
            sequences.pop(key, 'no_pop')
    return sequences

def countTransitions(fileName, cutoff):
    numTransCount=0
    seqTemp = defaultdict(lambda: 0)
    fprev = open(fileName, 'r')
    for line in fprev:
        sesID, categories, seqTemp, numTransCount = readLineFS(line, numTransCount, seqTemp, cutoff)
    fprev.close()
    return numTransCount


#read in YC dataset
#sesID, timestamp, Item ID, Category
#return train, trainmask, valid, validmask, test, testmask, (all sequences)
def readSequencesRawFirstX(fileName, numTransitions=10000, train_size=0.8, cutoff=sys.maxint):
    numTransCount=0
    #if we want all data, go through whole file first to count transitions
    maxTransitions = countTransitions(fileName, sys.maxint)
    if numTransitions==sys.maxint or numTransitions > maxTransitions:
        numTransitions = maxTransitions
        print(numTransitions)

    #record all sequences and where valid starts and test starts
    sequences=defaultdict(lambda: {'sequence': [], 'validStart': -1, 'testStart': -1})

    # sequencesHash is used to record whether or not a sesID has been in it before
    # it is also used to count how many transitions per sesID
    # we can do a cutoff and only allow users up to cutoff transitions
    sequencesHash = defaultdict(lambda: 0)
    f = open(fileName, 'r')
    testStart = int(0.7*numTransitions)
    validStart= int(0.7*train_size*numTransitions)
    numTransCount = 0
    numTransEveryCount=0
    for line in f:
        numTransEveryCount+=1
        before=numTransCount
        #making this into a function gives us flexibility on reading in different datasets
        sesID, categories, sequencesHash, numTransCount = readLineFS(line, numTransCount, sequencesHash, cutoff)

        #continue if we throw away
        if before==numTransCount:
            continue

        for category in categories:

            #validation set
            if numTransEveryCount >= validStart and numTransEveryCount < testStart and sequences[sesID]['validStart'] == -1:
                sequences[sesID]['validStart'] = len(sequences[sesID]['sequence'])
            #testing set
            elif numTransEveryCount >= testStart and sequences[sesID]['testStart'] == -1:
                sequences[sesID]['testStart'] = len(sequences[sesID]['sequence'])

            #append to sequence
            sequences[sesID]['sequence'].append(category)

        #break early if we are satisfied with number of transitions
        if numTransCount>=numTransitions:
            break

    sequencesTrain=defaultdict(lambda:[])
    sequencesTrainMask=defaultdict(lambda:[])
    sequencesValid=defaultdict(lambda:[])
    sequencesValidMask=defaultdict(lambda:[])
    sequencesTest=defaultdict(lambda:[])
    sequencesTestMask=defaultdict(lambda:[])
    sequencesAll = defaultdict(lambda:[])

    # now iterate through sequence and produce train, valid, test and each of their masks
    # making sure only to include what's needed (train does not need everything)
    for sesID in sequences:
        sequencesAll[sesID] = sequences[sesID]['sequence']

        vStrt=sequences[sesID]['validStart']
        tStrt=sequences[sesID]['testStart']
        seq = sequences[sesID]['sequence']
        # if both validStart and tStart don't exist
        # only training set
        if vStrt==-1 and tStrt==-1:
            sequencesTrain[sesID] = seq

        # if only validStart exist
        # we have training and valid now
        elif vStrt!=-1 and tStrt==-1:
            for i, element in enumerate(seq):
                sequencesValid[sesID].append(element)
                if i < vStrt:
                    sequencesTrain[sesID].append(element)
                    sequencesValidMask[sesID].append(0)
                else:
                    sequencesValidMask[sesID].append(1)
        # if only testStart exists
        # we have training and test now
        elif vStrt==-1 and tStrt!=-1:
            for i, element in enumerate(seq):
                sequencesTest[sesID].append(element)
                if i < tStrt:
                    sequencesTrain[sesID].append(element)
                    sequencesTestMask[sesID].append(0)
                else:
                    sequencesTestMask[sesID].append(1)
        # if both exist, we have the traditional train, valid, test split
        # we have train, valid, test
        elif vStrt!=-1 and tStrt!=-1:

            sequencesTestMask[sesID] = [0]*tStrt + [1]*(len(seq)-tStrt)
            sequencesValidMask[sesID] = [0]*vStrt + [1]*(tStrt-vStrt)
            for i, element in enumerate(seq):
                # figure out when to add sequence elements
                # test is always added, valid is added only if <testStart
                # train only added if < trainStart
                sequencesTest[sesID].append(element)
                if i < tStrt:
                    sequencesValid[sesID].append(element)
                elif i < vStrt:
                    sequencesTrain[sesID].append(element)

        # train mask is always static
        if sesID in sequencesTrain:
            sequencesTrainMask[sesID] = [1]*len(sequencesTrain[sesID])


    #next stage is pruning stuff

    sequencesTrain = pruneTransitions(sequencesTrain)
    sequencesTrainMask = pruneTransitions(sequencesTrainMask)
    sequencesValid = pruneTransitions(sequencesValid)
    sequencesValidMask = pruneTransitions(sequencesValidMask)
    sequencesTest = pruneTransitions(sequencesTest)
    sequencesTestMask = pruneTransitions(sequencesTestMask)

    return (sequencesTrain, sequencesTrainMask, sequencesValid, sequencesValidMask, sequencesTest, sequencesTestMask, sequencesAll)


#take places visited and convert to nums
def mapStrToNum(sequences):
    strMap={}
    #0 is missing value
    strMap['0']=0
    count=0
    #iterate through all values of sequences
    for key in sequences:
        for val in sequences[key]:
            if val not in strMap:
                count+=1
                strMap[val]=count
    mapStr = {}
    for key in strMap:
        val = strMap[key]
        mapStr[val]=key
    return (strMap, mapStr)


#test set consists of
#[seq, mask, start]
def getTesting(numTransitions, train_size, trainvalidtest="test"):
    if trainvalidtest=="test":
        prefix="testing"
    else:
        prefix=trainvalidtest

    testing = np.load("data/"+prefix+"_transitions_"+str(numTransitions)+"_trainsize_"+str(train_size)+".npy")
    testSet = []
    for i in range(0, len(testing)):
        seq = np.array(testing[i][0], dtype='uint8')
        seqLen = seq.shape[0]
        seq = seq.reshape(seqLen, 1)
        mask = np.ones((seqLen, 1), dtype=theano.config.floatX)
        start = testing[i][1][0]
        testSet.append([seq, mask, start])
    return testSet

#print the sequence distribution give a list of sequences
def printSequenceDistribution(sequences):
    seqHash=defaultdict(lambda:0)
    for key in sequences:
        seqHash[len(sequences[key])]+=1

    for key in sorted(seqHash.keys()):
        print(str(key)+", "+str(seqHash[key]))
    return seqHash



if __name__ == "__main__":
    #train, test, All = readSequencesRawFirstX("data/yoochoose-clicks.dat", 10)
    #sequencesTrainPruned, sequencesTrainNPPruned, sequencesTestPruned, sequencesValidPruned, sequencesAll = readSequencesRawFirstX("data/yoochoose-clicks.dat", 23754215)
    #sequencesTrainPruned, sequencesTrainNPPruned, sequencesTestPruned, sequencesValidPruned, sequencesAll = readSequencesRawFirstX("data/four_sq.dat", 453429)
    sequencesTrainPruned, sequencesTrainNPPruned, sequencesTestPruned, sequencesValidPruned, sequencesAll = readSequencesRawFirstX("data/four_sq.dat", 10000)
    char_to_ix, ix_to_char = mapStrToNum(sequencesAll)
    vocab_size = len(char_to_ix)
    print("vocab_size: "+str(vocab_size))
    #printSequenceDistribution(sequencesAll)
    #strMap = mapStrToNum(All)
