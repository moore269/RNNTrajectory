import os
import sys

"""
    rnnExperiments.py
    Main experimental file
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

device=sys.argv[1]

def experiments(rmpsprops, hiddenSizes, layers, boosting, dataSets, device, transition=sys.maxint):
    count=0
    for rmsprop in rmsprops:
        for size in hiddenSizes:
            for layer in layers:
                for boost in boosting:
                    for data in dataSets:
                        count+=1
                        submitStr = "time python train.py lstm " + str(size) + " " + str(layer) + " 0.0 0.8 " + str(transition) + " m3 1 " + str(data) + " 20 " + str(boost) + " " + str(rmsprop)
                        print(submitStr)
                        os.putenv("THEANO_FLAGS", "device="+str(device))
                        os.system(submitStr)
    return count

#48
globalCount=0
rmsprops = [0.01, 0.003, 0.001, 0.0003]
hiddenSizes = [200, 300]
layers = [1]
boosting = [1, 0]
dataSets = ["four_sq", "yoochoose-clicks", "brightkite"]

globalCount+=experiments(rmsprops, hiddenSizes, layers, boosting, dataSets, device, transition=1000000)

#24
#do layers help?
hiddenSizes=[8]
layers=[2]

#globalCount+=experiments(rmsprops, hiddenSizes, layers, boosting, dataSets, device, transition=1000000)

#24
hiddenSizes=[6]
layers=[3]
#globalCount+=experiments(rmsprops, hiddenSizes, layers, boosting, dataSets, device, transition=1000000)

#32
#now see if boosting has an effect on music dataSets
dataSets = ["last_fm_groups", "lastfm_1k"]
hiddenSizes=[10]
layers=[1]
#globalCount+=experiments(rmsprops, hiddenSizes, layers, boosting, dataSets, device, transition=1000000)

print("total experiments: "+str(globalCount))