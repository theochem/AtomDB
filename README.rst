..
    : This file is part of AtomDB.
    :
    : AtomDB is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : AtomDB is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with AtomDB. If not, see <http://www.gnu.org/licenses/>.

AtomDB
======
|Python3.9| |Github|

About
-----

AtomDB is a database of atomic and ionic properties.

Installation
------------

.. code-block::

    python -m pip install -e .

Usage
-----

.. code-block::

    % python -m atomdb -h
    usage: python -m atomdb [-h] [-c] [-q] [--exc EXC] dataset elem basis charge mult

    Compile and/or query an AtomDB entry

    positional arguments:
    dataset     Dataset
    elem        Element symbol
    basis       Basis set
    charge      Charge
    mult        Multiplicity

    optional arguments:
    -h, --help  show this help message and exit
    -c          Compile the specified entry
    -q          Query the specified entry
    --exc EXC   Excitation level

If AtomDB is installed in a directory for which you do not have write permissions, or if you'd
rather store your AtomDB database in another directory (or you already have one), then you
can set the ``ATOMDB_DATAPATH`` environment variable to the directory of your choice; either
a non-existing directory, or an existing AtomDB database.

On release, AtomDB should come with compiled database files pre-installed in the default
``ATOMDB_DATAPATH``. Raw data files containing the output of electronic structure computations
will be hosted in another Git repo (AtomDB-Data) via LFS. If you are a developer working on compiling new
database entries or writing new datasets, then you can set ``ATOMDB_DATAPATH`` to the location
of the AtomDB-Data repo on your computer.

Send Michael your SSH public key and ask him for the current URL for the AtomDB-Data repo.

Contributing
------------

You can help by writing features, properties, and datasets, or by running computations! 🙂

TODO
~~~~
- Add functions for dealing with multiple database entries at once
- Add more properties (C-DFT, SP-CDFT, etc.)
- Add more datasets (Hartree-Fock, some useful DFT calcs...?)
- Get the NIST dataset, and add those dataset-independent fields to the DB entries
- Make all field names consistent with IOData
- See Issues tab

Citations
---------

2021 QuantumElephant 🐘 AtomDB ⚛

.. |Python3.9| image:: http://img.shields.io/badge/python-3.9-blue.svg
   :target: https://docs.python.org/3/
.. |Github| image:: https://img.shields.io/badge/quantumelephant-black.svg?logo=GitHub
   :target: https://github.com/quantumelephant/atomdb/
