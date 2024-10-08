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

from atomdb import compile_species, load


# Initialize command line argument parser
#
parser = ArgumentParser(prog="atomdb", description="Compile or query an AtomDB entry.")

# Define subcommands
#
command = parser.add_argument_group("commands")

command_group = command.add_mutually_exclusive_group(required=True)

command_group.add_argument(
    "-c", "--compile_species", action="store_true", help="compile_species a species into the database"
)

command_group.add_argument(
    "-q", "--query", action="store_true", help="query a species from the database"
)


# Add arguments
#
arg_group = parser.add_argument_group("arguments")

arg_group.add_argument("dataset", type=str, help="name of dataset")

arg_group.add_argument("elem", type=str, help="element symbol")

arg_group.add_argument("charge", type=int, help="charge")

arg_group.add_argument("mult", type=int, help="multiplicity")

arg_group.add_argument("-e", "--exc", type=int, default=0, help="excitation level")


if __name__ == "__main__":

    args = parser.parse_args()

    if args.compile:

        compile_species(args.elem, args.charge, args.mult, args.exc, args.dataset)

    elif args.query:

        species = load(args.elem, args.charge, args.mult, args.exc, args.dataset)

        print(species.to_json())
