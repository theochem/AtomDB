# -*- coding: utf-8 -*-
# AtomDB is an extended periodic table database containing experimental
# and/or computational information on stable ground state
# and/or excited states of neutral and charged atomic species.
#
# Copyright (C) 2014-2015 The AtomDB Development Team
#
# This file is part of AtomDB.
#
# AtomDB is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AtomDB is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


from atomdb.units import angstrom, amu
from atomdb.base import Species, SpeciesTable


def load_basic_periodic_table():
    """Load data from elements.csv file into a `SpeciesTable`.

    The following attributes are present for some elements. When a parameter
    is not known for a given element, the attribute is set to `None`.

    group
        The group of element in periodic table e(not for actinides and lanthanides).

    period
        The row of element in the periodic table.

    mass
        The IUPAC atomic masses (wieghts) of 2013.
        T.B. Coplen, W.A. Brand, J. Meija, M. Gröning, N.E. Holden, M.
        Berglund, P. De Bièvre, R.D. Loss, T. Prohaska, and T. Walczyk.
        http://ciaaw.org, http://www.ciaaw.org/pubs/TSAW2013_xls.xls,
        When ranges are provided, the middle of the range is used.

    cov_radius_cordero
        Covalent radius. B. Cordero, V. Gomez, A. E. Platero-Prats, M.
        Reves, J. Echeverria, E. Cremades, F. Barragan, and S. Alvarez,
        Dalton Trans. pp. 2832--2838 (2008), URL
        http://dx.doi.org/10.1039/b801115j

    cov_radius_bragg
        Covalent radius. W. L. Bragg, Phil. Mag. 40, 169 (1920), URL
        http://dx.doi.org/10.1080/14786440808636111

    cov_radius_slater
        Covalent radius. J. C. Slater, J. Chem. Phys. 41, 3199 (1964), URL
        http://dx.doi.org/10.1063/1.1725697

    vdw_radius_bondi
        van der Waals radius. A. Bondi, J. Phys. Chem. 68, 441 (1964), URL
        http://dx.doi.org/10.1021/j100785a001

    vdw_radius_truhlar
        van der Waals radius. M. Mantina A. C. Chamberlin R. Valero C. J.
        Cramer D. G. Truhlar J. Phys. Chem. A 113 5806 (2009), URL
        http://dx.doi.org/10.1021/jp8111556

    vdw_radius_rt
        van der Waals radius. R. S. Rowland and R. Taylor, J. Phys. Chem.
        100, 7384 (1996), URL http://dx.doi.org/10.1021/jp953141+

    vdw_radius_batsanov
        van der Waals radius. S. S. Batsanov Inorganic Materials 37 871
        (2001), URL http://dx.doi.org/10.1023/a%3a1011625728803

    vdw_radius_dreiding
        van der Waals radius. Stephen L. Mayo, Barry D. Olafson, and William
        A. Goddard III J. Phys. Chem. 94 8897 (1990), URL
        http://dx.doi.org/10.1021/j100389a010

    vdw_radius_uff
        van der Waals radius. A. K. Rappi, C. J. Casewit, K. S. Colwell, W.
        A. Goddard III, and W. M. Skid J. Am. Chem. Soc. 114 10024 (1992),
        URL http://dx.doi.org/10.1021/ja00051a040

    vdw_radius_mm3
        van der Waals radius. N. L. Allinger, X. Zhou, and J. Bergsma,
        Journal of Molecular Structure: THEOCHEM 312, 69 (1994),
        http://dx.doi.org/10.1016/s0166-1280(09)80008-0

    wc_radius
        Waber-Cromer radius of the outermost orbital maximum. J. T. Waber
        and D. T. Cromer, J. Chem. Phys. 42, 4116 (1965), URL
        http://dx.doi.org/10.1063/1.1695904

    cr_radius
        Clementi-Raimondi radius. E. Clementi, D. L. Raimondi, W. P.
        Reinhardt, J. Chem. Phys. 47, 1300 (1967), URL
        http://dx.doi.org/10.1063/1.1712084

    pold_crc
        Isolated atom dipole polarizability. CRC Handbook of Chemistry and
        Physics (CRC, Boca Raton, FL, 2003). If multiple values were present
        in the CRC book, the value used in Erin's postg code is taken.

    pold_chu
        Isolated atom dipole polarizability. X. Chu & A. Dalgarno, J. Chem.
        Phys., 121(9), 4083--4088 (2004), URL
        http://dx.doi.org/10.1063/1.1779576 Theoretical value for hydrogen
        from this paper: A.D. Buckingham, K.L. Clarke; Chem. Phys. Lett.
        57(3), 321--325 (1978), URL
        http://dx.doi.org/10.1016/0009-2614(78)85517-1

    c6_chu
        Isolated atom C_6 dispersion coefficient. X. Chu & A. Dalgarno, J. Chem.
        Phys., 121(9), 4083--4088 (2004), URL
        http://dx.doi.org/10.1063/1.1779576 Theoretical value for hydrogen
        from this paper: K. T. Tang, J. M. Norbeck and P. R. Certain; J.
        Chem. Phys. 64, 3063 (1976), URL #
        http://dx.doi.org/10.1063/1.432569
    """

    import csv

    convertor_types = {
        'int': (lambda s: int(s)),
        'float': (lambda s : float(s)),
        'au': (lambda s : float(s)),    # just for clarity, atomic units
        'str': (lambda s: s.strip()),
        'angstrom': (lambda s: float(s)*angstrom),
        '2angstrom': (lambda s: float(s)*angstrom/2),
        'angstrom**3': (lambda s: float(s)*angstrom**3),
        'amu': (lambda s: float(s)*amu),
    }

    with open('atomdb/data/elements.csv', 'r') as f:
        r = csv.reader(f)
        # go to the actual data
        for row in r:
            if len(row[1]) > 0:
                break
        # parse the first two header rows
        names = row
        convertors = [convertor_types[key] for key in r.next()]
        elements = []

        species = []
        for row in r:
            if len(row) == 0:
                break
            kwargs = {}
            for i in xrange(len(row)):
                cell = row[i]
                if len(cell) > 0:
                    kwargs[names[i]] = convertors[i](cell)
                else:
                    kwargs[names[i]] = None
            number = kwargs.pop("number")
            keys = sorted(kwargs.keys())
            species.append(Species(number, number, **kwargs))

    return SpeciesTable(species)


table_Periodic = load_basic_periodic_table()
