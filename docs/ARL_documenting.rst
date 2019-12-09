
Documenting ARL
***************

* The primary documentation is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ (rst).
* We use `Sphinx <http://www.sphinx-doc.org>`_ to extract code documentation.
* We use the package `sphinx_automodapi <https://sphinx-automodapi.readthedocs.io/>`_ to build the API informatiom.
* For this to work, all of the code must be loadable into python. To facilitate this, we make use of the dreaded
  ``from somewhere import *``. This means that modules must use ``__all__`` to only export those names that are
  delivered by that module, as oopposed to the other names used in the module.
