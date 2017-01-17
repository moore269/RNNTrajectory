"""
    mysaveload.py
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

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import secure_dump, load, load_parameter_values

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class myCheckpoint(Checkpoint):
    """Saves a pickled version of the main loop to the disk.

    The pickled main loop can be later reloaded and training can be
    resumed.

    Makes a `SAVED_TO` record in the log with the serialization destination
    in the case of success and ``None`` in the case of failure. The
    value of the record is a tuple of paths to which saving was done
    (there can be more than one if the user added a condition
    with an argument, see :meth:`do` docs).

    Parameters
    ----------
    path : str
        The destination path for pickling.
    save_separately : list of str, optional
        The list of the main loop's attributes to be pickled separately
        to their own files. The paths will be formed by adding
        the attribute name preceded by an underscore before the
        `path` extension. The whole main loop will still be pickled
        as usual.
    use_cpickle : bool
        See documentation of :func:`~blocks.serialization.dump`.

    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.


    """
    def __init__(self, path, save_separately=None, use_cpickle=False,
                 **kwargs):
        super(myCheckpoint, self).__init__(path, save_separately, use_cpickle, **kwargs)

    def do(self, which_callback, *args): 
        sharedData={}
        try: 
             for var in self.main_loop:
                if "sharedData" in var.name:
                    sharedData[var.name] = var.get_value()
                    var.set_value(0)
             super().do(self, which_callback, *args) 
        finally: 
             # Change the shared variable back. 
             for var in self.main_loop:
                if "sharedData" in var.name and var.name in sharedData:
                    var.set_value(sharedData[key])
