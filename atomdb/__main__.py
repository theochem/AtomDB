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
from sys import argv


parser = ArgumentParser(prog='python -m atomdb', description='Run an AtomDB command.')


subparsers = parser.add_subparsers(help='command help text')


parser_generate = subparsers.add_parser('generate', help='subcommand help text')
parser_generate.add_argument('dataset', type=str, help='argument help text')
parser_generate.add_argument('element', type=str, help='argument help text')
parser_generate.add_argument('charge', type=str, help='argument help text')
parser_generate.add_argument('multiplicity', type=str, help='argument help text')
parser_generate.add_argument('excitation_level', type=str, help='argument help text')


parser_compile = subparsers.add_parser('compile', help='subcommand help text')
parser_compile.add_argument('dataset', type=str, help='argument help text')
parser_compile.add_argument('element', type=str, help='argument help text')
parser_compile.add_argument('charge', type=str, help='argument help text')
parser_compile.add_argument('multiplicity', type=str, help='argument help text')
parser_compile.add_argument('excitation_level', type=str, help='argument help text')


if __name__ == '__main__':

    namespace = parser.parse_args(argv[1:])
