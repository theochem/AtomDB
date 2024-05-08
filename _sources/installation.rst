.. _usr_installation:

Installation
############

Downloading Code
================

The latest code can be obtained through `theochem <https://github.com/theochem/AtomDB/>`_ in Github,

.. code-block:: bash

   git clone https://github.com/theochem/AtomDB.git

.. _usr_py_depend:


Installation
============

Installation via pip can be done by the following command:


.. code-block:: bash

    pip install git+https://github.com/theochem/AtomDB.git


AtomDB can also be installed by cloning via git,


.. code-block:: bash

   git clone https://github.com/theochem/AtomDB.git


Then installation via pip can be done by going into the directory where AtomDB was downloaded to
and running,


.. code-block:: bash

    cd AtomDB
    pip install .


or to install AtomDB as an editable package, run,

.. code-block:: bash

    pip install -e .


Successful installation can be checked by running the tests,

.. code-block:: bash

    pytest --pyargs atomdb


Dependencies
============

Basic dependencies:
-------------------

The following dependencies will be necessary for AtomDB to build properly,

* Python >= 3.7: http://www.python.org/
* NumPy >= 1.16.0: http://www.numpy.org/
* SciPy >= 1.4.0: http://www.scipy.org/
* msgpack >= 1.0.0: https://msgpack.org/
* msgpack-numpy >= 0.4.8: https://github.com/lebedov/msgpack-numpy
* h5py >= 3.6.0: https://www.h5py.org/
* importlib_resources >=3.0.0: https://github.com/python/importlib_resources

These will be installed automatically when installing AtomDB via pip.

Optional dependencies for generating documentation:
---------------------------------------------------

The following dependencies are optional and are only necessary if you want to generate the
documentation locally,

* ipython : https://ipython.org/
* numpydoc : https://numpydoc.readthedocs.io/en/latest/
* sphinx_copybutton : https://sphinx-copybutton.readthedocs.io/en/latest/
* sphinx-autoapi : https://sphinx-autoapi.readthedocs.io/en/latest/
* nbsphinx : https://nbsphinx.readthedocs.io/en/latest/
* nbconvert : https://nbconvert.readthedocs.io/en/latest/
* sphinx_rtd_theme : https://sphinx-rtd-theme.readthedocs.io/en/latest/
* sphinx_autodoc_typehints : https://sphinx-autodoc-typehints.readthedocs.io/en/latest/
* docutils == 0.16 : https://docutils.sourceforge.io/
* nbsphinx-link : https://nbsphinx-link.readthedocs.io/en/latest/

These can be installed via pip by running,

.. code-block:: bash

    pip install .[doc]

Optional dependencies for development:
--------------------------------------

AtomDB also provides a toolbox for extending its capabilities by modifying the
(or adding new) databases. The following dependencies are optional and are
only necessary if you intend to do so.

* pytest >= 2.6 : https://docs.pytest.org/en/stable/
* pyscf >= 1.7.0 : https://pyscf.org/
* qc-gbasis : https://gbasis.qcdevs.org/
* qc-grid : https://grid.qcdevs.org/
* qc-iodata : https://iodata.qcdevs.org/

These can be installed via pip by running,

.. code-block:: bash

    pip install .[dev]


Building Documentation
======================

The documentation can be built locally by running the following commands in the doc directory,


.. code-block:: bash

    make html


Other formats can be built by replacing html with the desired format. For a list of available
formats, run,


.. code-block:: bash

    make