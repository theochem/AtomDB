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
|Python| |Github|

About
-----

AtomDB is a database of atomic and ionic properties. ⚛

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

Contributing
------------

You can help by coding stuff or running computations! 🙂

Citations
---------

*todo*

.. |Python| image:: http://img.shields.io/badge/python-3-blue.svg
   :target: https://docs.python.org/3/
.. |Github| image:: https://img.shields.io/badge/quantumelephant-black.svg?logo=GitHub
   :target: https://github.com/quantumelephant/atomdb/
