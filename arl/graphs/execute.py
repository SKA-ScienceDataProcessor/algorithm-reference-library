""" Execute wraps execution services like dask

"""

from dask import delayed

class ARLExecuteBase():
    
    def __init__(self, use_dask=True):
        self.use_dask = use_dask
        
    def execute(self, func, *args, **kwargs):
        """ Wrap for immediate or deferred execution
        
        :param args:
        :param kwargs:
        :return:
        """
        if self.use_dask:
            from dask import delayed
            return delayed(func, *args, **kwargs)
        else:
            return func
        
    def type(self):
        """ Get the name of the execution system
        
        :return:
        """
        if self.use_dask:
            return 'dask'
        else:
            return 'function'
        
    def get(self, value):
        """
        
        :param value:
        :return:
        """
        if self.use_dask:
            return value.compute()
        else:
            return value

arlexecute = ARLExecuteBase(use_dask=True)