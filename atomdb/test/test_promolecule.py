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

import numpy as np
import numpy.testing as npt
import pytest

from atomdb.promolecule import make_promolecule

TEST_CASES_MAKE_PROMOLECULE = [
    pytest.param(
        {
            "atnums": [1, 1],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            "dataset": "slater",
        },
        id="H2 default charge/mult",
    ),
    pytest.param(
        {
            "atnums": [6, 6],
            "charges": [1, -1],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            "dataset": "slater",
        },
        id="C2 +/-1 charge, default mult",
    ),
    pytest.param(
        {
            "atnums": [6, 6],
            "charges": [0.5, -0.5],
            "mults": [1, -1],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            "dataset": "slater",
        },
        id="C2 +/-0.5 charge",
    ),
    pytest.param(
        {
            "atnums": [4],
            "charges": [1.2],
            "mults": [1.2],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "dataset": "uhf_augccpvdz",
        },
        id="Be floating point charge/floating point mult",
    ),
    pytest.param(
        {
            "atnums": [4],
            "charges": [1.2],
            "mults": [1],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "dataset": "uhf_augccpvdz",
        },
        id="Be floating point charge/integer mult",
    ),
    pytest.param(
        {
            "atnums": [4],
            "charges": [1.2],
            "mults": [-1.2],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "dataset": "uhf_augccpvdz",
        },
        id="Be floating point charge/floating point mult (neg)",
    ),
    pytest.param(
        {
            "atnums": [4],
            "charges": [1.2],
            "mults": [-1],
            "coords": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            "dataset": "uhf_augccpvdz",
        },
        id="Be floating point charge/integer mult (neg)",
    ),
]


@pytest.mark.parametrize("case", TEST_CASES_MAKE_PROMOLECULE)
def test_make_promolecule(case):
    r"""
    Test ``make_promolecule()`` function.

    """
    atnums = case.get("atnums")
    coords = case.get("coords")
    charges = case.get("charges", None)
    mults = case.get("mults", None)
    units = case.get("units", None)
    dataset = case.get("dataset", None)

    promol = make_promolecule(
        atnums,
        coords,
        charges=charges,
        mults=mults,
        units=units,
        dataset=dataset,
    )

    # Check that coefficients add up to (# centers)
    npt.assert_allclose(sum(promol.coeffs), len(atnums))

    # Check that electron number and charge are recovered
    if charges is not None:
        npt.assert_allclose(sum(atnums) - sum(charges), promol.nelec())
        npt.assert_allclose(sum(charges), promol.charge())

    # Check that spin number and multiplicity are recovered
    if mults is not None:
        npt.assert_allclose(sum(np.sign(m) * (abs(m) - 1) for m in mults), promol.nspin())
        npt.assert_allclose(abs(sum(np.sign(m) * (abs(m) - 1) for m in mults)) + 1, promol.mult())
