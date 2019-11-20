""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

from arlexecutebase import ARLExecuteBase
arlexecute = ARLExecuteBase(use_dask=True)