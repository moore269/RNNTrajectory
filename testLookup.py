"""
    testLookup.py
    Quickly test lookup functionality
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
from numpy.testing import assert_equal

import theano
from theano import tensor

from blocks.bricks.lookup import LookupTable


def test_lookup_table():
    lt = LookupTable(5, 3)
    lt.allocate()

    lt.W.set_value(numpy.arange(15).reshape(5, 3).astype(theano.config.floatX))

    x = tensor.lmatrix("x")
    y = lt.apply(x)
    f = theano.function([x], [y])

    x_val = [[1, 2], [0, 0]]
    desired = numpy.array([[[3, 4, 5], [6, 7, 8]], [[0, 1, 2], [9, 10, 11]]],
                          dtype=theano.config.floatX)
    assert_equal(f(x_val)[0], desired)

test_lookup_table()