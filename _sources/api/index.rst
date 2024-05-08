.. AtomDB documentation master file, created by
   sphinx-quickstart on Fri Feb  2 15:55:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AtomDB's documentation!
================================

AtomDB is a versatile, free and open-source Python library for accessing and managing atomic and
promolecular properties. It serves as an extended database or periodic table, of neutral and charged
atomic properties offering accurate experimental and computational data for various atomic
charge/multiplicity states.

Please use the following citation in any publication using AtomDB:

.. bibliography::
    :all:
    :list: bullet

.. literalinclude:: atomdb_cite.bib
    :language: bibtex
    :tab-width: 100

`AtomDB source code <https://github.com/theochem/AtomDB/>`_ is hosted on GitHub and is released under the
GNU General Public License v3.0. We welcome any contributions to the AtomDB library in accordance with our
Code of Conduct; please see our Contributing Guidelines. Please report any issues you encounter while
using AtomDB on `GitHub Issues <https://github.com/theochem/AtomDB/issues/new>`_. For further
information and inquiries please contact us at qcdevs@gmail.com.

Functionality
=============
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

Properties
==========

The table below lists the atomic properties available for the datasets in AtomDB:

.. csv-table::
    :file: ./table_datasets.csv
    :widths: 20, 16, 16, 16, 16, 16
    :delim: ;
    :header-rows: 1
    :align: center
