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

from atomdb import *


parser = ArgumentParser(prog='python -m atomdb', description='Run an AtomDB command.')


subparsers = parser.add_subparsers(dest='cmd', help='Command to run.')


parser_generate = subparsers.add_parser('generate', help='Generate raw data files.')
parser_generate.add_argument('dataset', type=str, help='Dataset for which to generate raw data.')
parser_generate.add_argument('element', type=str, help='Element symbol of species.')
parser_generate.add_argument('charge', type=int, help='Charge of species.')
parser_generate.add_argument('mult', type=int, help='Multiplicity of species.')
parser_generate.add_argument('--exc', type=int, dest='nexc', default=0, help='Excitation level of species.')


parser_compile = subparsers.add_parser('compile', help='Compile DB files.')
parser_compile.add_argument('dataset', type=str, help='Dataset for which to compile DB files.')
parser_compile.add_argument('element', type=str, help='Element symbol of species.')
parser_compile.add_argument('charge', type=int, help='Charge of species.')
parser_compile.add_argument('mult', type=int, help='Multiplicity of species.')
parser_compile.add_argument('--exc', type=int, dest='nexc', default=0, help='Excitation level of species.')


parser_query = subparsers.add_parser('query', help='Query and print a DB entry as JSON.')
parser_query.add_argument('dataset', type=str, help='Dataset to query.')
parser_query.add_argument('element', type=str, help='Element symbol of species.')
parser_query.add_argument('charge', type=int, help='Charge of species.')
parser_query.add_argument('mult', type=int, help='Multiplicity of species.')
parser_query.add_argument('--exc', type=int, dest='nexc', default=0, help='Excitation level of species.')


if __name__ == '__main__':

    # Parse arguments
    args = parser.parse_args()

    # Run specified command
    if args.cmd == 'generate':
        #try:
        generate_species(args.element, args.charge, args.mult, args.nexc, dataset=args.dataset)
        # TODO: handle all keywords and more specific errors
        #except Exception:
            #print("An error occured. Exiting...", file=stderr)
            #exit(1)
    elif args.cmd == 'compile':
        #try:
        dump_species(
            compile_species(args.element, args.charge, args.mult, args.nexc, dataset=args.dataset),
        )
        # TODO: handle all keywords and more specific errors
        #except Exception:
            #print("An error occured. Exiting...", file=stderr)
            #exit(1)
    elif args.cmd == 'query':
        #try:
        print_species(
            load_species(args.element, args.charge, args.mult, args.nexc, dataset=args.dataset)
        )
        # TODO: handle all keywords and more specific errors
        #except Exception:
            #print("An error occured. Exiting...", file=stderr)
            #exit(1)
    exit(0)
