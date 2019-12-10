""" Buffer equivalents of memory data models

To create from an existing model in the buffer, make some changes, and then sync to buffer::

    my_buffer_skymodel = BufferSkyModel(conf["buffer"], conf["inputs"]["skymodel"])
    my_memory_skymodel = my_buffer_skymodel.memory_data_model
    ... do some stuff
    my_memory_skymodel.sync()
    
To create a new buffer for a memory_data_model::

    my_buffer_skymodel = BufferSkyModel(conf["buffer"], conf["inputs"]["skymodel"], my_memory_skymodel)
    my_memory_skymodel.sync()

An explicit sync is required in both cases

"""

__all__ = ['BufferGainTable',
           'BufferPointingTable',
           'BufferImage',
           'BufferGridData',
           'BufferConvolutionFunction',
           'BufferSkyModel',
           'BufferBlockVisibility']


import collections
import logging

from arl.data_models.data_model_helpers import buffer_data_model_to_memory, memory_data_model_to_buffer
from arl.data_models.memory_data_models import Image, BlockVisibility, SkyModel, GainTable, GridData, \
    ConvolutionFunction, PointingTable

log = logging.getLogger(__name__)


class BufferDataModel():
    """Buffer version of data model
    
    To create from an existing model in the buffer, make some changes, and then sync to buffer::

        my_buffer_skymodel = BufferSkyModel(conf["buffer"], conf["inputs"]["skymodel"])
        my_memory_skymodel = my_buffer_skymodel.memory_data_model()
        ... do some stuff
        my_memory_skymodel.sync()
    
To create a new buffer for a memory_data_model::

        my_buffer_skymodel = BufferSkyModel(conf["buffer"], conf["inputs"]["skymodel"], my_memory_skymodel)
        my_memory_skymodel.sync()

An explicit sync is required in both cases.
"""
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """
        
        :param json_buffer: Description of buffer in JSON
        :param json_model: Description of model in buffer in JSON
        :param mdm: memory data model (can be a list or a single data model)
        """
        self.json_buffer = json_buffer
        self.json_model = json_model
        if mdm is not None:
            self._memory_data_model = mdm
        else:
            self._memory_data_model = buffer_data_model_to_memory(self.json_buffer, self.json_model)

    @property
    def memory_data_model(self):
        return self._memory_data_model
    
    @property
    def type(self):
        return type(self._memory_data_model)
    
    def assert_type(self, memory_data_model_type):
        if isinstance(self._memory_data_model, collections.Iterable):
            for m in self._memory_data_model:
                assert isinstance(m, memory_data_model_type), "Expected %s, actual %s" % (memory_data_model_type,
                                                                                          type(m))
        else:
            assert isinstance(self._memory_data_model, memory_data_model_type), \
                "Expected %s, actual %s" % (memory_data_model_type, type(self._memory_data_model))
    
    def sync(self):
        """ Save to buffer
        
        :return:
        """
        memory_data_model_to_buffer(self._memory_data_model, self.json_buffer, self.json_model)


class BufferImage(BufferDataModel):
    """Buffer version of memory data model Image

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(Image)


class BufferGridData(BufferDataModel):
    """Buffer version of memory data model GridData

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(GridData)


class BufferConvolutionFunction(BufferDataModel):
    """Buffer version of memory data model ConvolutionFunction

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(ConvolutionFunction)


class BufferBlockVisibility(BufferDataModel):
    """Buffer version of memory data model BlockVisibility

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(BlockVisibility)


class BufferSkyModel(BufferDataModel):
    """Buffer version of memory data model SkyModel

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(SkyModel)


class BufferGainTable(BufferDataModel):
    """Buffer version of memory data model GainTable

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(GainTable)


class BufferPointingTable(BufferDataModel):
    """Buffer version of memory data model GainTable

    """
    
    def __init__(self, json_buffer, json_model, mdm=None):
        """

        :param json_buffer: JSON description of buffer
        :param json_model: JSON descriptiomn of model
        :return: Image
        """
        BufferDataModel.__init__(self, json_buffer, json_model, mdm)
        self.assert_type(PointingTable)
