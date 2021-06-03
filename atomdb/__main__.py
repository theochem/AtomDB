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


parser = ArgumentParser(prog='python -m atomdb', description='Run an AtomDB command.')


subparsers = parser.add_subparsers(dest='cmd', help='Command to run.')


parser_generate = subparsers.add_parser('generate', help='Generate raw data files.')
parser_generate.add_argument('dataset', type=str, help='Dataset for which to generate raw data.')
parser_generate.add_argument('element', type=str, help='Element symbol of species.')
parser_generate.add_argument('charge', type=int, help='Charge of species.')
parser_generate.add_argument('mult', type=int, help='Multiplicity of species.')
parser_generate.add_argument('nexc', type=int, help='Excitation level of species.')


parser_compile = subparsers.add_parser('compile', help='Compile DB files.')
parser_compile.add_argument('dataset', type=str, help='Dataset for which to compile DB files.')
parser_compile.add_argument('element', type=str, help='Element symbol of species.')
parser_compile.add_argument('charge', type=int, help='Charge of species.')
parser_compile.add_argument('mult', type=int, help='Multiplicity of species.')
parser_compile.add_argument('nexc', type=int, help='Excitation level of species.')


parser_compile = subparsers.add_parser('query', help='Query and print a DB entry as JSON.')
parser_compile.add_argument('dataset', type=str, help='Dataset to query.')
parser_compile.add_argument('element', type=str, help='Element symbol of species.')
parser_compile.add_argument('charge', type=int, help='Charge of species.')
parser_compile.add_argument('mult', type=int, help='Multiplicity of species.')
parser_compile.add_argument('nexc', type=int, help='Excitation level of species.')


if __name__ == '__main__':

    # Parse arguments
    args = parser.parse_args()

    # Run specified command
    if args.cmd == 'generate':
        # TODO: Run generate function -- exit(0) if successful or exit(1) if unsuccessful
        # Generate raw data
        print(args)
    elif args.cmd == 'compile':
        # TODO: Run compile function -- exit(0) if successful or exit(1) if unsuccessful
        # Compile raw data to .msg
        print(args)
    elif args.cmd == 'query':
        # TODO: Run query function -- exit(0) if successful or exit(1) if unsuccessful
        # Print JSON file to stdout
        print(args)
    else:
        # Execution should never reach here because parser.parse_args handles this case
        exit(127)
