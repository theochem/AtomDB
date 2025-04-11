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

"""AtomDB console script.

This script provides a command-line interface to compile or query entries in the AtomDB database.
"""

import sys
from argparse import ArgumentParser, ArgumentTypeError
from atomdb import compile_species, load


def positive_int(value):
    """Validate if the argument is a positive integer."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ArgumentTypeError(f"{value} is not a positive integer.")
        return ivalue
    except ValueError as e:
        raise ArgumentTypeError(f"Invalid integer value: {value}") from e


def main():
    """Main function for the AtomDB CLI."""
    parser = ArgumentParser(
        prog="atomdb",
        description="Command-line tool to compile or query an AtomDB entry.",
    )

    # Define mutually exclusive commands
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument(
        "-c", "--compile_species", action="store_true", help="Compile a species into the database."
    )
    command_group.add_argument(
        "-q", "--query", action="store_true", help="Query a species from the database."
    )

    # Add arguments
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument("elem", type=str, help="Element symbol (e.g., H, He, Li).")
    parser.add_argument("charge", type=positive_int, help="Charge of the species (positive integer).")
    parser.add_argument("mult", type=positive_int, help="Multiplicity of the species (positive integer).")
    parser.add_argument(
        "-e", "--exc", type=int, default=0, help="Excitation level (default: 0). Must be non-negative."
    )

    # Parse arguments
    args = parser.parse_args()

    # Input validation
    if args.exc < 0:
        print("Error: Excitation level (-e) must be a non-negative integer.", file=sys.stderr)
        sys.exit(1)

    try:
        # Handle commands
        if args.compile_species:
            print(f"Compiling species: {args.elem}, Charge: {args.charge}, Multiplicity: {args.mult}, Excitation: {args.exc}")
            compile_species(args.elem, args.charge, args.mult, args.exc, args.dataset)
            print("Compilation successful.")
        elif args.query:
            print(f"Querying species: {args.elem}, Charge: {args.charge}, Multiplicity: {args.mult}, Excitation: {args.exc}")
            species = load(args.elem, args.charge, args.mult, args.exc, args.dataset)
            print("Query result:", species.to_json())
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

