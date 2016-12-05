"""
    sample.py
    sample code
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

import numpy
import theano
from blocks.serialization import load
from blocks.bricks import NDimensionalSoftmax
from utils import get_metadata, get_stream
from config import config
import argparse
import sys


def sample(nchars, x_curr, x_curr_mask, predict, ix_to_char, temperature=1.0, seed=None):
    if seed:
        numpy.random.seed(seed)
    sample_string = ''
    for _ in range(nchars):
        probs = predict(x_curr, x_curr_mask).squeeze()[-1].astype('float64')
        if numpy.random.binomial(1, temperature) == 1:
            probs = probs / probs.sum()
            sample = numpy.random.multinomial(1, probs).nonzero()[0][0]
        else:
            sample = probs.argmax()
        sys.stdout.write(ix_to_char[sample])
        sample_string += ix_to_char[sample]
        x_curr = numpy.roll(x_curr, -1)
        x_curr[-1] = sample
    sys.stdout.write('\n')

    return sample_string

if __name__ == '__main__':
    # Load config parameters
    locals().update(config)

    parser = argparse.ArgumentParser(
        description='Sample from a character-level language model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', default=save_path,
                        help='model checkpoint to use for sampling')
    parser.add_argument('-primetext', default=None,
                        help='used as a prompt to "seed" the state of the RNN using a given sequence, before we sample.')
    parser.add_argument('-length', default=1000,
                        type=int, help='number of characters to sample')
    parser.add_argument('-seed', default=None,
                        type=int, help='seed for random number generator')
    parser.add_argument('-temperature', type=float,
                        default=1.0, help='temperature of sampling')
    args = parser.parse_args()

    # Define primetext
    ix_to_char, char_to_ix, vocab_size = get_metadata(hdf5_file)
    if args.primetext and len(args.primetext) > 0:
        primetext = ''.join(
            [ch for ch in args.primetext if ch in char_to_ix.keys()])
        x_curr = numpy.expand_dims(
            numpy.array([char_to_ix[ch] for ch in primetext], dtype='uint8'), axis=1)
    else:
        dev_stream = get_stream(hdf5_file, 'dev', batch_size)
        x_curr, x_curr_mask, y_curr, y_curr_mask  = dev_stream.get_epoch_iterator().next()
        cur_seq_length = x_curr.shape[0]
        x_curr = x_curr[:, -1].reshape(cur_seq_length, 1)
        x_curr_mask = x_curr_mask[:, -1].reshape(cur_seq_length, 1)

    print 'Loading model from {0}...'.format(args.model)
    main_loop = load(args.model)
    print 'Model loaded. Building prediction function...'
    model = main_loop.model
    y_mask, y, x, x_mask = model.inputs
    softmax = NDimensionalSoftmax()
    linear_output = [
        v for v in model.variables if v.name == 'linear_output'][0]
    y_hat = softmax.apply(linear_output, extra_ndim=1)
    predict = theano.function([x, x_mask], y_hat)

    print 'Starting sampling'
    sample_string = sample(args.length, x_curr, x_curr_mask, predict, ix_to_char,
                           seed=args.seed, temperature=args.temperature)
