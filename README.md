RNN Trajectory
This repository provides a reference implementation of RNNs for Trajectory prediction described in the paper:<br>
> Using Recurrent Neural Networks in Trajectory Prediction<br>
https://www.cs.purdue.edu/homes/moore269/docs/rnntrajectory.pdf

#### Prerequisites
Blocks 0.1.1
Theano
Numpy

You can use
pip install git+git://github.com/mila-udem/blocks.git@v0.1.1 \
  -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

### Basic Usage

### Example
To train an LSTM, execute the following command from the project home directory:<br/>
	``python train.py --rnn-type lstm``

To predict using the saved model, execute:<br/>
    ``python predict.py --rnn-type lstm``

#### Options
You can check out the other options available using:<br/>
	``python train.py --help``
	<br/>and<br/>
	``python predict.py --help``

#### Configuration
To tune more configuration parameters, simply edit config.py

