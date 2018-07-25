""" Generic functions converted to arlexecute components. arlexecute is a wrapper for`Dask <http://dask.pydata.org/>`_
is a python-based flexible parallel computing library for analytic computing. Dask.delayed can be used to wrap
functions for deferred execution thus allowing construction of components.
"""

from ..execution_support.arlexecute import arlexecute


def generic_blockvisibility_arlexecute(visfunction, vis_list, additive=True, *args, **kwargs):
    """ Definition of interface for create_generic_blockvisibility_graph_visfunction.

    :func visfunction: Function to be applied
    :param vis_list: List of vis_graphs
    :param additive: Add to existing visibility? (True)
    :param args:
    :param kwargs: Parameters for functions in components
    :return: List of components
    """
    
    def accumulate_results(results, **kwargs):
        for i, result in enumerate(results):
            if additive:
                vis_list[i].data['vis'] += result.data['vis']
            else:
                vis_list[i].data['vis'] = result.data['vis']
        return vis_list
    
    results = list()
    for vis_graph in vis_list:
        results.append(arlexecute.execute(visfunction, pure=True)(vis_graph, *args, **kwargs))
    return [arlexecute.execute(accumulate_results, pure=True)(results, **kwargs)]
