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
config['models_folder'] = 'models/'  # Models folder
config['batch_size'] = 100  # Number of samples taken per each update
config['learning_rate_decay'] = 0.97 # Set to 0 to not decay learning rate
config['decay_rate'] = 0.95  # Decay rate for rmsprop
config['step_clipping'] = 5.0  # Clip norm of gradients at this value

config['epsilon'] = 0.00000000001 # Must be in epsilon improvement for ealry stopping

config['model'] = 'rnn'  # 'rnn', 'gru' or 'lstm'
config['nepochs'] = 100  # Max epochs through training data
config['maxIterations'] = 100
config['num_epochs'] = 10  # Until no improvement

config['save_path'] = '_best.pkl' # Path to best model file
config['load_path'] = '_saved.pkl'  # Start from a saved model file
config['last_path'] = '_last.pkl' # Path to save the model of the last iteration
config['plots_output'] = 'outputPlots/' # Path to plots output
config['numProcesses'] = 4 # For use in python multi-processing
config['saveEveryXIteration'] = 10 # Save after every X iteration

#have at most last cutThreshold observations
config['cutThreshold'] = 20 # Cut threshold experiments
config['maxSeqLen'] = 200 # Max number of elements in the sequence

config['onlyPlots'] = True # Only plot if necessary