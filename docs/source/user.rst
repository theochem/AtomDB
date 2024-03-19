Installation
============

AtomDB can be installed with ``pip``. In a virtual environment, simply run the
following command to install AtomDB and its dependencies:

.. code:: bash

   pip install .


If you intend to build the `Sphinx documentation`__, you can also install AtomDB
with the optional dependencies required to do so by appending the ``[doc]`` tag
to the install target:

.. _Sphinx: https://www.sphinx-doc.org/

__ Sphinx_

.. code:: bash

   pip install .[doc]

Here is a full example installation:

[**NOTE**: At this stage in development you must use option C]

.. code:: bash

    # Create a virtual environment in `~/example`.
    # Feel free to change the path.
    python3 -m venv ~/example

    # Activate the virtual environment.
    source ~/example/bin/activate

    # Then run one of the following:

    # A) Install the stable release in the venv `example`.
    pip3 install atomdb

    # B) For developers, install an alpha or beta pre-release
    # (only do this if you understand the implications.)
    pip install --pre atomdb

    # C) Install the latest git revision
    # (only do this if you understand the implications.)
    pip install -e .

Setup
=====

If AtomDB is installed in a directory for which you do not have write
permissions, or if you'd rather store your AtomDB database in another directory
(or you already have one), then you can set the ``ATOMDB_DATAPATH`` environment
variable to the directory of your choice; either a non-existant directory, or an
existing AtomDB database directory.

On release, AtomDB should come with compiled database files pre-installed in the
default ``ATOMDB_DATAPATH``. Raw data files containing the output of electronic
structure computations will be hosted in another Git repo (AtomDB-Data) using
LFS. If you are a developer working on compiling new database entries or writing
new datasets, then you can set ``ATOMDB_DATAPATH`` to the location of the
AtomDB-Data repo on your computer.

Tutorial
========

.. code-block:: bash

    ~/git/atomdb $ python -m atomdb -h
    usage: python -m atomdb [-h] [-c] [-q] [-e E] dataset elem basis charge mult

    Compile and/or query an AtomDB entry

    positional arguments:
      dataset     name of dataset
      elem        element symbol
      basis       basis set
      charge      charge
      mult        multiplicity

    optional arguments:
      -h, --help  show this help message and exit
      -c          compile the specified entry
      -q          query the specified entry
      -e E        excitation level

TODO
