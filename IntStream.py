"""
    IntStream.py
    Stream of Integers
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
from fuel.streams import AbstractDataStream
from fuel.datasets import IterableDataset, IndexableDataset
from fuel.iterator import DataIterator
from fuel.schemes import ShuffledScheme


class IntStream(AbstractDataStream):
    """A stream of data from integers all the way up to maxBatch.
    Parameters
    ----------
    maxBatch : maximum integer to be reached
    """
    def __init__(self, startIndex, numExamples, batchSize, name, **kwargs):
        super(IntStream, self).__init__(**kwargs)
        self.startIndex = startIndex
        self.numExamples = numExamples+startIndex
        self.batchSize = batchSize
        self.count=self.startIndex-self.batchSize
        self._sources=[name+"_From", name+"_To"]

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.reset()

    def reset(self):
        self.count=self.startIndex-self.batchSize

    def next_epoch(self):
        self.count+=self.batchSize
        if self.count>=self.numExamples:
            self.reset()
            raise StopIteration

    def get_data(self, request=None):
        """Get data from the dataset."""
        ret = self.count
        try:
            self.next_epoch()
        except StopIteration:
            raise StopIteration
        return [self.count, min(self.count+self.batchSize, self.numExamples)]

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the data stream."""
        return super(IntStream, self).get_epoch_iterator(**kwargs)

if __name__ == "__main__":
    startIndex = 957
    numExamples = 50
    batchSize = 100
    stream = IntStream(startIndex, numExamples, batchSize, 'train')
    print(stream.sources)
    siter = stream.get_epoch_iterator()
    while True:
        try:
            batch = next(siter)
            print(batch)
        except StopIteration:
            break
    print('here')