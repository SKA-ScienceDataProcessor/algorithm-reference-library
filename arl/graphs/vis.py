"""

Simpler visualization of dask graphs

From J Crist gist
<script src="https://gist.github.com/jcrist/dc5b7cedfddff123f2177e5238e566e5.js"></script>
"""

import os
import graphviz

from dask.optimize import key_split
from dask.dot import _get_display_cls
from dask.core import get_dependencies
from dask.callbacks import Callback
from dask.dot import dot_graph



def node_key(s):
    if isinstance(s, tuple):
        return s[0]
    return str(s)


def simple_vis(x, filename='simple', format=None, **kwargs):
    if hasattr(x, 'dask'):
        dsk = x._optimize(x.dask, x._keys())
    else:
        dsk = x

    deps = {k: get_dependencies(dsk, k) for k in dsk}

    g = graphviz.Digraph(graph_attr={'rankdir': 'LR'})

    nodes = set()
    edges = set()
    for k in dsk:
        key = node_key(k)
        if key not in nodes:
            g.node(key, label=key_split(k), shape='rectangle')
            nodes.add(key)
        for dep in deps[k]:
            dep_key = node_key(dep)
            if dep_key not in nodes:
                g.node(dep_key, label=key_split(dep), shape='rectangle')
                nodes.add(dep_key)
            # Avoid circular references
            if dep_key != key and (dep_key, key) not in edges:
                g.edge(dep_key, key)
                edges.add((dep_key, key))

    fmts = ['.png', '.pdf', '.dot', '.svg', '.jpeg', '.jpg']
    if format is None and any(filename.lower().endswith(fmt) for fmt in fmts):
        filename, format = os.path.splitext(filename)
        format = format[1:].lower()

    if format is None:
        format = 'png'

    data = g.pipe(format=format)
    if not data:
        raise RuntimeError("Graphviz failed to properly produce an image. "
                           "This probably means your installation of graphviz "
                           "is missing png support. See: "
                           "https://github.com/ContinuumIO/anaconda-issues/"
                           "issues/485 for more information.")

    display_cls = _get_display_cls(format)

    if not filename:
        return display_cls(data=data)

    full_filename = '.'.join([filename, format])
    with open(full_filename, 'wb') as f:
        f.write(data)

    return display_cls(filename=full_filename)

# Scheduler plugin to produce animated graphs
# From https://gist.github.com/jcrist/0c28f632513aa13d4edea3d482bf47d1

class Track(Callback):
    def __init__(self, path='dasks', save_every=1):
        self.path = path
        self.save_every = save_every
        self.n = 0
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    def _plot(self, dsk, state):
        data = {}
        func = {}
        for key in state['released']:
            data[key] = {'color': 'blue'}
        for key in state['cache']:
            data[key] = {'color': 'red'}
        for key in state['finished']:
            func[key] = {'color': 'blue'}
        for key in state['running']:
            func[key] = {'color': 'red'}

        filename = os.path.join(self.path, 'part_{:0>4d}'.format(self.n))

        dot_graph(dsk, filename=filename, format='png',
                  data_attributes=data,
                  function_attributes=func)

    def _pretask(self, key, dsk, state):
        if self.n % self.save_every == 0:
            self._plot(dsk, state)
        self.n += 1

    def finish(self, dsk, state, errored):
        self._plot(dsk, state)
        self.n += 1