"""
    myCheckpoint.py
    Extensions for saving and loading the state of a training process.
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
import os.path
import logging

from six.moves import cPickle
import sys

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import (
    secure_dump, load, DEFAULT_PROTOCOL)
from theano import shared
import theano
import numpy as np

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class myCheckpoint(SimpleExtension):
    def __init__(self, path, tempShared, save_separately=None, use_cpickle=False,
                 **kwargs):
        kwargs.setdefault("after_training", True)
        super(myCheckpoint, self).__init__(**kwargs)
        if not save_separately:
            save_separately = []
        self.path = path
        self.save_separately = save_separately
        self.use_cpickle = use_cpickle

        #initiate temporary shared vars and temp indices for them
        self.tempShared = tempShared
        self.tempIndices = {}
        self.updateIndices()


    def updateIndices(self):
        for key in self.tempShared:
            self.tempIndices[key]=-1

    def save_separately_filenames(self, path):
        """Compute paths for separately saved attributes.
        Parameters
        ----------
        path : str
            Path to which the main checkpoint file is being saved.
        Returns
        -------
        paths : dict
            A dictionary mapping attribute names to derived paths
            based on the `path` passed in as an argument.
        """
        root, ext = os.path.splitext(path)
        return {attribute: root + "_" + attribute + ext
                for attribute in self.save_separately}

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.
        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.
        """
        sharedData={}
        _, from_user = self.parse_args(callback_name, args)
        try:

            #shared data manipulation

            #add default values to temp share
            for var in self.main_loop.model.variables:
                if var.name!=None and "sharedData" in var.name:
                    sharedData[var.name] = var.get_value(borrow=True)
                    if theano.config.floatX in str(var.type):
                        self.tempIndices[theano.config.floatX]+=1
                        temp = self.tempShared[theano.config.floatX][self.tempIndices[theano.config.floatX]]   
                    elif 'uint8' in str(var.type):
                        self.tempIndices['uint8']+=1
                        temp = self.tempShared['uint8'][self.tempIndices['uint8']]
                    else:
                        print('you have not accounted for all shared variables')
                        sys.exit(0)
                    var.set_value(temp.get_value(borrow=True), borrow=True)

            path = self.path
            if from_user:
                path, = from_user
            secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump,
                            protocol=DEFAULT_PROTOCOL)
        except Exception:
            path = None
            raise
        finally:
            self.updateIndices()
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
            # Change the shared variable back. 
            for var in self.main_loop.model.variables:
                if var.name!=None and "sharedData" in var.name and var.name in sharedData:
                    var.set_value(sharedData[var.name], borrow=True)