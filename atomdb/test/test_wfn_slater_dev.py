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
#

import pytest

from atomdb.datasets.slater import get_cs_occupations

from importlib_resources import files
import os

# mark all tests in this file as development tests
pytestmark = pytest.mark.dev

# get test data path
TEST_DATAPATH = files("atomdb.test.data")
TEST_DATAPATH = os.fspath(TEST_DATAPATH._paths[0])


def test_get_cs_occupations():
    configurations = ["1S(2)2S(2)2P(1)", "K(2)L(8)3S(2)3P(3)"]
    results = [
        [{"1S": 1, "2S": 1, "2P": 1}, {"1S": 1, "2S": 1, "2P": 0}],
        [
            {"1S": 1, "2S": 1, "2P": 3, "3S": 1, "3P": 3},
            {"1S": 1, "2S": 1, "2P": 3, "3S": 1, "3P": 0},
        ],
    ]
    for configuration, result in zip(configurations, results):
        alpha_occ, beta_occ, _ = get_cs_occupations(configuration)
        for key, a_occ in alpha_occ.items():
            assert a_occ == result[0][key]
            assert beta_occ[key] == result[1][key]
