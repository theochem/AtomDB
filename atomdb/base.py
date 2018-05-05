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
"""Base class for extended periodic table of properties."""


class Species(object):
    """Neural or Charged Atomic Species Class."""

    def __init__(self, number, nelectron, **kwargs):
        """
        Parameters
        ----------
        number : int
            Atomic number.
        nelectron : int
            Number of electrons.
        kwargs : dict
            Additional attributes.
        """
        # check atomic number and number of electrons
        if not isinstance(number, int) or not number > 0:
            raise ValueError("The atomic number argument should be a positive integer.")
        if number > 118:
            raise ValueError("The atomic number should be less than or equal to 118.")
        if not isinstance(nelectron, int) or not nelectron >= 0:
            raise ValueError("The nelectron argument should be a positive integer.")

        self._number = number
        self._nelectron = nelectron

        # add additional attributes
        self.update(**kwargs)

    @property
    def number(self):
        """Atomic number."""
        return self._number

    @property
    def nelectron(self):
        """Number of electrons."""
        return self._nelectron

    def update(self, **kwargs):
        """Update the attributes."""
        for key, value in kwargs.iteritems():
            if key in dir(self):
                raise ValueError("The {0} attribute already exists!".format(key))
            setattr(self, key, value)

    def __getitem__(self, attr):
        """Return attribute value."""
        return getattr(self, attr)

    def __repr__(self):
        """Prints a list of attributes."""
        attrs = [atr for atr in dir(self) if not callable(atr) and not atr.startswith('_')]
        content = "Available attributes:" + "\n" + "\n".join(sorted(attrs)) + "\n"
        return content


class SpeciesTable(object):
    """Table of Neutral or Charged Atomic Species."""

    def __init__(self, species):
        """
        Parameters
        ----------
        species : instances of `Species`
        """
        self.__species = species
        self.__lookup = {}
        for sp in self.__species:
            self.__lookup[(sp.number, sp.nelectron)] = sp

    def __len__(self):
        """Return the number of species in the table."""
        return len(self.__species)

    def __getitem__(self, (number, nelectron)):
        """Return the `Species` object for the specified atomic number and charge."""
        return self.__lookup[(number, nelectron)]

    def available_species(self, number):
        """Return the available `Species` in the table for the given atomic number."""
        sps = [key[1] for key in self.__lookup.keys() if key[0] == number]
        return sps
