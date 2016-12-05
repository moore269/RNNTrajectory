"""
    plotting.py
    plot after training
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
import theano
from blocks.serialization import load
from blocks.bricks import NDimensionalSoftmax
from utils import get_metadata, get_stream
from config import config
import argparse
import sys
from readData import getTesting
import os
from pylab import plot, legend, xlabel, ylabel, title, savefig, loglog
from readData import *
from predict import *

# Load config parameters
locals().update(config)


#read custom times file for times for each of the experiments
#id rnn_type    hidden_size num_layers  dropout train_size  transitions time
def readTimes(timesFile):
    timesDict = {}
    f = open(timesFile, 'r')
    keys=[]
    for i, line in enumerate(f):
        if i==0:
            keys=line.replace("\n", "").split("\t")
        else:
            vals=line.replace("\n", "").split("\t")
            timeVal = vals[len(vals)-1]
            valsStr=keys[1]+"_"+vals[1]
            for i, val in enumerate(vals[2:len(vals)-1]):
                valsStr+="_"+keys[i+2]+"_"+val
            timesDict[valsStr]=float(timeVal)

    return timesDict

def saveResults(models_folder, mod="", vary="hidden_size", isTime=False, timesFile="lstmTimes1.txt", varyRange=xrange(200, 300), trial=11, rnnType='lstm', hiddenSize=300, numLayers=1, dropout=0.0, train_size=0.8, transitions=10000):
    if len(mod)>0:
        mod=mod+"_"
    #strify them
    trial=str(trial)
    hiddenSize=str(hiddenSize)
    numLayers=str(numLayers)
    dropout=str(dropout)
    train_size=str(train_size)
    transitions=str(transitions)
    testResults = []
    if vary=="hidden_size" and isTime:
        timesOb = readTimes(timesFile)
        for hiddenSize in varyRange:
            hiddenSize=str(hiddenSize)
            if len(mod)>0:
                key="rnn_type_"+rnnType+"c_hidden_size_"+hiddenSize+"_num_layers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions
            else:
                key = "rnn_type_"+rnnType+"_hidden_size_"+hiddenSize+"_num_layers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions
            result = timesOb[key]
            testResults.append((result, result, result))
        hiddenSize=str(varyRange[0])

    elif vary=="transition_size" and isTime:
        timesOb = readTimes(timesFile)
        for transitions in varyRange:
            transitions = str(transitions)
            if len(mod)>0:
                key = "rnn_type_"+rnnType+"c_hidden_size_"+hiddenSize+"_num_layers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions
            else:
                "rnn_type_"+rnnType+"_hidden_size_"+hiddenSize+"_num_layers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions
            result = timesOb[key]
            testResults.append((result, result, result))
        transitions = str(varyRange[0])

    elif vary=="hidden_size":
        for hiddenSize in varyRange:
            hiddenSize=str(hiddenSize)
            cFile = models_folder+mod+"rnn_type_"+rnnType+"_trial_"+trial+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions+"_gru_best.pkl"
            evals = evaluation(cFile)
            testResults.append(evals)
        hiddenSize=str(varyRange[0])
    elif vary=="transition_size":
        for transitions in varyRange:
            transitions=str(transitions)
            cFile = models_folder+mod+"rnn_type_"+rnnType+"_trial_"+trial+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions+"_gru_best.pkl"
            evals = evaluation(cFile)
            testResults.append(evals)
        transitions=str(varyRange[0])
    elif vary=="train_size":
        for train_size in varyRange:
            train_size=str(train_size)
            cFile = models_folder+mod+"rnn_type_"+rnnType+"_trial_"+trial+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions+"_gru_best.pkl"
            evals=evaluation(cFile)
            testResults.append(evals)
        train_size=str(varyRange[0])
    elif vary=="dropout":
        for dropout in varyRange:
            dropout=str(dropout)
            cFile = models_folder+mod+"rnn_type_"+rnnType+"_trial_"+trial+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions+"_gru_best.pkl"
            evals=evaluation(cFile)
            testResults.append(evals)
        dropout=str(varyRange[0])        


    npResults=np.array(testResults)
    npRange = np.array([varyRange]).T
    results=np.concatenate((npRange, npResults), axis=1)
    np.save("output/time_"+str(isTime)+mod+"_rnn_type_"+rnnType+"_trial_"+trial+"_vary_"+vary+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions, results)

def plotResults(models_folder, mod="", crossComparison=False, isTime=False, titleName="default", vary="hidden_size", trial=11, rnnType='lstm', hiddenSize=300, numLayers=1, dropout=0.0, train_size=0.8, transitions=10000):
    if len(mod)>0:
        mod=mod+"_"
    trial=str(trial)
    hiddenSize=str(hiddenSize)
    numLayers=str(numLayers)
    dropout=str(dropout)
    train_size=str(train_size)
    transitions=str(transitions)
    setNames=["train", "valid", "test"]
    data=np.load("output/time_"+str(isTime)+mod+"_rnn_type_"+rnnType+"_trial_"+trial+"_vary_"+vary+"_hiddenSize_"+hiddenSize+"_numLayers_"+numLayers+"_dropout_"+dropout+"_train_size_"+train_size+"_transitions_"+transitions+".npy")

    if not crossComparison:
        for i in range(1, data.shape[1]):
            x = data[:,0]
            y=data[:,i]
            plot(x, y, label=setNames[i-1])
            if not isTime:
                legend(loc=3, ncol=2)
    else:
        for i in range(1,2):
            x = data[:,0]
            y=data[:,i]
            if len(mod)>0:
                plot(x,y, label=rnnType+"c")
            else:
                plot(x,y, label=rnnType)
            legend(loc=3, ncol=2)
    xlabel(vary)
    if isTime:
        ylabel('Training Time')
    else:
        ylabel('MRR')
    title(titleName)
    legend(loc=3, ncol=2)
    savefig("output/time_"+str(isTime)+"_"+titleName+"_"+vary+".png")


def plotAcc(accuracies, name, t):
    for key in accuracies:
        if not "prop" in key or "Best" in key:
            s = np.array(accuracies[key])
            plot(t, s, label=key)
            legend(loc=3, ncol=2)
            xlabel(name)
            ylabel('Accuracy')
            grid(True)
    title(name)
    savefig("output/"+name+".png")

def varyHiddenSize():
    varyRange=xrange(120, 320, 20)
    #saveResults(modelFolder, isTime=True, vary="hidden_size", transitions=40000, numLayers=1, varyRange=varyRange, rnnType='rnn')
    transitions=10000
    plotResults(modelFolder, crossComparison=True, mod="mod", isTime=True, numLayers=1, titleName="RNNC, "+str(transitions)+" transitions", vary="hidden_size", hiddenSize=varyRange[0], transitions=transitions, rnnType='rnn')
    plotResults(modelFolder, crossComparison=True, isTime=True, numLayers=1, titleName="RNNC, "+str(transitions)+" transitions", vary="hidden_size", hiddenSize=varyRange[0], transitions=transitions, rnnType='rnn')
    plotResults(modelFolder, crossComparison=True, isTime=True, numLayers=1, titleName="RNNC, "+str(transitions)+" transitions", vary="hidden_size", hiddenSize=varyRange[0], transitions=transitions, rnnType='lstm')

def varyTransitions():
    varyRange=xrange(10000, 50000, 10000)
    #saveResults(modelFolder, mod="mod", isTime=True, vary="transition_size", varyRange=varyRange, hiddenSize=300, numLayers=1, rnnType='rnn')
    plotResults(modelFolder, mod="mod", isTime=True, numLayers=1, hiddenSize = 300, rnnType='rnn', titleName="RNNC, vary transitions", vary="transition_size", transitions=varyRange[0])

def varyTrainSize():
    varyRange=xrange(1, 10, 1)
    varyRange = [x * 0.1 for x in varyRange] 
    #saveResults(modelFolder, isTime=False, vary="train_size", varyRange=varyRange, hiddenSize=300, numLayers=1, rnnType='lstm')
    #plotResults(modelFolder, isTime=False, numLayers=1, titleName="LSTM, 10000 transitions, vary training vs validation", vary="train_size", train_size=varyRange[0], transitions=10000, rnnType='lstm')

def varyDropout():
    varyRange=xrange(1, 10, 1)
    varyRange = [x * 0.1 for x in varyRange] 
    saveResults(modelFolder, isTime=False, vary="dropout", varyRange=varyRange, hiddenSize=500, numLayers=2, rnnType='lstm', transitions=0000)   
    #plotResults(modelFolder, isTime=False, numLayers=2, hiddenSize=500, titleName="LSTM, 10000 transitions, vary dropout", vary="dropout", dropout=varyRange[0], transitions=10000, rnnType='lstm')

#plot the distribution of sequence lengths
def plotSeqLens(dataSet, titleName, name=None, transitions=None):
    lenvals = []
    numlenVals = []
    #read from file if no transition number provided
    #we assume we want all of the datasets transition lengths, which we saved
    if not name==None:
        f=open(dataSet+"_"+name, 'r')
        for line in f:
            vals = line.replace("\n", "").split(", ")
            lenvals.append(vals[0])
            numlenVals.append(vals[1])
        saveName = "output/seqlen_"+name.replace(".txt", "")
        labName = "All"
    else:
        sequencesTrainPruned, sequencesTrainNPPruned, sequencesTestPruned, sequencesValidPruned, sequencesAll = readSequencesRawFirstX("data/"+dataSet+".dat", transitions)
        seqHash=printSequenceDistribution(sequencesAll)
        for key in sorted(seqHash):
            lenvals.append(key)
            numlenVals.append(seqHash[key])
        saveName = "output/"+dataSet+"seqlen_transitions_"+str(transitions)
        labName = str(transitions)+" transitions"

    loglog(np.array(lenvals), np.array(numlenVals), label=labName)
    xlabel("Sequence Lengths")
    legend(loc=1, ncol=1)
    title(titleName)
    savefig(saveName)

  
if __name__ == '__main__':
    scratch=os.environ.get('RCAC_SCRATCH')
    if not scratch ==None:
        modelFolder=scratch+"/lstm2models/yoochoose-clicks/"
    else:
        modelFolder="models/yoochoose-clicks/"
    #plotSeqLens(name="seqDistAll.txt", titleName="Sequence Length Distribution for All Transitions")
    plotSeqLens(dataSet="four_sq", titleName="Sequence Length Distribution for 10000 transitions", transitions=10000)
    plotSeqLens(dataSet="four_sq", titleName="Sequence Length Distribution for 20000 transitions", transitions=20000)
    plotSeqLens(dataSet="four_sq", titleName="Sequence Length Distribution for 30000 transitions", transitions=30000)
    plotSeqLens(dataSet="four_sq", titleName="Sequence Length Distribution", transitions=40000)
    plotSeqLens(dataSet="four_sq", titleName="Sequence Length Distribution", transitions=453429)
    #models_folder="models/yoochoose-clicksOLD/yoochoose-clicks/")
    #pickBestModel(11, modelFolder, numTransitions=numTransitions, rnnType='lstm')
    #evaluation(models_folder+"yoochoose-clicks/mod_rnn_type_rnn_trial_11_hiddenSize_120_numLayers_1_dropout_0.0_train_size_0.8_transitions_40000_gru_best.pkl")
    #evaluation(models_folder+"yoochoose-clicks/rnn_type_rnn_trial_11_hiddenSize_40_numLayers_1_dropout_0.0_train_size_0.8_transitions_10000_gru_best.pkl")
    #varyHiddenSize()
    #varyTransitions()
    #varyTrainSize()
    #varyDropout()
    #r = readTimes("lstmTimes1.txt")
    #print(r)

