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

|Documentation| |Github|

AtomDB
======

About
-----

AtomDB is a versatile, free and open-source Python library for accessing and managing atomic and
promolecular properties. It serves as an extended database or periodic table, of neutral and charged
atomic properties offering accurate experimental and computational data for various atomic
charge/multiplicity states.

AtomDB is a `QC-Devs <https://qcdevs.org/>`_ project.

Documentation
-------------

AtomDB's documentation, including installation and usage instructions, as well as API documentation,
is available at `atomdb.qcdevs.org <https://atomdb.qcdevs.org/>`_.

Functionality
~~~~~~~~~~~~~

* **Atomic scalar properties**
    AtomDB provides a wide range of atomic properties for neutral and charged atoms, including:
    **Atomic number**, **Atomic symbol**, **Atomic mass**, **Atomic radius**, **van der Waals radius**,
    **Covalent radius**, **Ionization potential**, **Electron affinity**, **Electronegativity**,
    **Atomic polarizability**

* **Point dependent properties**
    AtomDB provides functions to calculate point-dependent properties, such as:
    **Electron density** :math:`\rho(r)`, **Electron density gradient** :math:`\nabla \rho(r)`,
    **Electron density Laplacian** :math:`\nabla^2 \rho(r)`, **Electron density Hessian** :math:`\nabla^2 \rho(r)`
    (for these properties, only the radial part is provided), and **Kinetic energy density** :math:`ked(r)`

    The computation of contributions per orbital, set of orbitals or spin to these properties is also supported.

* **promolecular properties**
    AtomDB provides the capabilities to create promolecular models, and then estimate molecular properties
    from the atomic properties.

* **Dumping and loading**
    AtomDB provides the capability to dump and load atomic properties to and from json files.

For a complete list of available properties, see
`this table <https://atomdb.qcdevs.org/api/index.html#properties>`_.

Contributing
------------

We welcome any contributions to the AtomDB library in accordance with our
`Code of Conduct <https://qcdevs.org/guidelines/qcdevs_code_of_conduct/>`_; please see our
`Contributing Guidelines <https://qcdevs.org/guidelines/>`_.
Please report any issues you encounter while using AtomDB on GitHub Issues.

For further information and inquiries please contact us at
`qcdevs@gmail.com <mailto:qcdevs@gmail.com>`_.

Citing AtomDB
-------------

Please use the following citation in any publication using AtomDB:

.. code::bibtex

    @Article{atomdb,
        author  = {S{\'a}nchez D{\'\i}az, Gabriela and Richer, Michelle and
                   Mart{\'\i}nez Gonz{\'a}lez, Marco and {v}an Zyl, Maximilian and
                   Pujal, Leila and Tehrani, Alireza and Bianchi, Julianna and
                   Ayers, Paul W. and Heidar-Zadeh, Farnaz},
        title   = {{AtomDB: A Python Library for Atomic and Promolecular Properties}},
        journal = {-},
        year    = {2024},
        url     = {https://atomdb.qcdevs.org/},
    }

License
-------

AtomDB's source code is hosted on GitHub and is released under the GNU General Public License v3.0.

.. |Documentation| image:: https://img.shields.io/badge/documentation-blue.svg?logo=Sphinx
   :target: https://atomdb.qcdevs.org/

.. |Github| image:: https://img.shields.io/badge/theochem-black.svg?logo=GitHub
   :target: https://github.com/theochem/AtomDB/
