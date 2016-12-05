"""
    baselines.py
    simple baselines
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
from readData import *
from collections import defaultdict
from readData import *

class GravityModel:
    """
    Simple predictive model for forecasting items
    """
    def __init__(self, trainSet, vocabSize):
        self.createTransitionProbs(trainSet, vocabSize)

    #unnormalized, array indices match numbers
    def createTransitionProbs(self, trainSet, vocabSize):
        transitionHash=defaultdict(lambda:[0]*vocabSize)
        for seq in trainSet:
            for i in range(1, len(seq)):
                strFrom=seq[i-1]
                strTo=seq[i]
                transitionHash[strFrom][strTo]+=1

        self.transitionHash=transitionHash
        #create sorted list of transitions
        tHash2=defaultdict(lambda:[-1])
        for key in transitionHash:
            tList = transitionHash[key]

            #gather non-zeros
            nonzeros={}
            for i, item in enumerate(tList):
                if item !=0:
                    nonzeros[i]=item

            sortedTList=[]
            for item in sorted(nonzeros.items(), key=lambda x: x[1], reverse=True):
                sortedTList.append(item[0])

            tHash2[key] = sortedTList

        self.tHashOrdered = tHash2

    def predictRec(self, sequence, start):
        recRank = 0.0
        totalPreds=len(sequence) - start
        for i in range(start, len(sequence)):
            actualVal = sequence[i][0]
            prevVal= sequence[i-1][0]
            for j, val in enumerate(self.tHashOrdered[prevVal]):
                if j>500:
                    break
                if val == actualVal:
                    recRank+= (1.0/(j+1))
                    break
        return (recRank, totalPreds)


    def evaluatePreds(self, testSet):
        recRank=0.0
        totalPreds=0
        for seqOb in testSet:
            seq=seqOb[0]
            mask = seqOb[1]
            start = seqOb[2]
            recRankSeq, totalPredsSeq = self.predictRec(seq, start)
            recRank+=recRankSeq
            totalPreds+=totalPredsSeq
        print("UnNormRecRank: "+str(recRank))
        print("totalPreds: "+str(totalPreds))
        print("recRank: "+str(recRank/totalPreds))






if __name__ == "__main__":
    #seq5=readSequences("data/four_sq.dat", 10)
    #G = GravityModel(seq5, 4)
    #accuracies = G.evaluatePreds(seq5, 2, 4)
    numTransitions=10000
    train_size=0.8
    testSet=getTesting(numTransitions, train_size, trainvalidtest="test")
    trainSet = getTesting(numTransitions, train_size, trainvalidtest="train")
    vocabSize=np.load("data/vocab_size_transition_size_"+str(numTransitions)+".npy").tolist()
    G = GravityModel(trainSet, vocabSize)
    G.evaluatePreds(testSet)
    


