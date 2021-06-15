# This file is part of AtomDB.
#
# AtomDB is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# AtomDB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with AtomDB. If not, see <http://www.gnu.org/licenses/>.

r"""AtomDB console script."""


from argparse import ArgumentParser

from sys import stderr

import atomdb


# Initialize command line argument parser
parser = ArgumentParser(prog="python -m atomdb", description="Compile and/or query an AtomDB entry")


# Specify positional arguments and options
parser.add_argument("-c", action="store_true", default=False, help="compile the specified entry")
parser.add_argument("-q", action="store_true", default=False, help="query the specified entry")
parser.add_argument("dataset", type=str, help="name of dataset")
parser.add_argument("elem", type=str, help="element symbol")
parser.add_argument("basis", type=str, default=None, help="basis set")
parser.add_argument("charge", type=int, help="charge")
parser.add_argument("mult", type=int, help="multiplicity")
parser.add_argument("-e", type=int, default=0, help="excitation level")


if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Exit if no commands are specified
    if not (args.c or args.q):
        print("No command specified. Exiting...", file=stderr)
        exit(1)

    # Run specified command(s)
    if args.c:
        atomdb.compile(args.elem, args.charge, args.mult, args.e, args.basis, args.dataset)
    if args.q:
        print(atomdb.load(args.elem, args.charge, args.mult, args.e, args.basis, args.dataset).to_json())

    # Exit successfully
    exit(0)
