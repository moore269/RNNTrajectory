"""
    config.py
    Config file for common parameters settings
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
config = {}

#set config model according to machine
scratchFolder = os.environ.get('RCAC_SCRATCH')
if scratchFolder!=None:
	config['models_folder'] = scratchFolder + '/lstm2models/'#models folder
else:
	config['models_folder'] = 'models/' #models folder


config['batch_size'] = 100 #50  # number of samples taken per each update
config['hidden_size'] = 128  # number of hidden units per layer
config['num_layers'] = 2
config['learning_rate'] = 0.002
config['learning_rate_decay'] = 0.97 # set to 0 to not decay learning rate
config['decay_rate'] = 0.95  # decay rate for rmsprop
config['step_clipping'] = 5.0  # clip norm of gradients at this value
config['dropout'] = 0

config['epsilon'] = 0.00000000001

config['model'] = 'rnn'  # 'rnn', 'gru' or 'lstm'
config['nepochs'] = 100  # max epochs through training data
config['maxIterations'] = 100

config['num_epochs'] = 10  # until no improvement
config['seq_length'] = 50  # number of chars in the sequence
#config['data_name'] = 'yoochoose-clicks' #data input name
#config['data_name'] = 'four_sq'
#config['hdf5_file'] = 'data/'+config['data_name']  # hdf5 file with Fuel format
#config['text_file'] = 'data/'+config['data_name']+'.dat'  # raw input file
config['train_size'] = 0.8  # fraction of data that goes into train set
config['save_path'] = '_best.pkl' # path to best model file
config['load_path'] = '_saved.pkl'  # start from a saved model file
config['last_path'] = '_last.pkl' # path to save the model of the last iteration
config['plots_output'] = 'outputPlots/'
config['numProcesses'] = 4
config['saveEveryXIteration'] = 10

#have at most last cutThreshold observations
config['cutThreshold'] = 20
config['maxSeqLen'] = 200

config['onlyPlots'] = True