import csv
import numpy as np

from importlib.resources import files

__all__ = ["sym2num", "name2num", "num2sym", "num2name", "Atom"]

if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use 'from atomdb.periodic import Atom' instead.")
    raise SystemExit

else:
    # auxiliar functions
    def _read_csv(file):
        """Reads a csv file and returns a list of lists with the data.

        Parameters
        ----------
        file : file
            The file to read.

        Returns
        -------
        list of lists
            The data in the file.
        """
        data = []
        for row in csv.reader(file):
            # ignore comments and empty lines
            if row and not row[0].startswith("#") and not all(c in ", \n" for c in row):
                # Replace \n with new line in each element
                data.append([element.replace("\\n", "\n") for element in row])
        return data

    def _indent_lines(input_string, indent):
        """Indent each line of a string by a given number of spaces."""
        lines = input_string.splitlines()
        indented_lines = [(indent * " ") + line for line in lines]
        return "\n".join(indented_lines)

    def _gendoc(info_file):
        """Generate a docstring for the Element class.

        Parameters
        ----------
        info_file : str
            The name of the csv file with the data for constructing the docstring.
        """
        # create a dictionary to store the metadata of the properties
        _data_src = _read_csv(open(info_file))
        _prop2name = {}
        _prop2desc = {}
        _prop2unit = {}
        _prop2source = {}
        _prop2url = {}
        _prop2notes = {}
        for prop, prop_name, key, unit, description, source, url, notes in _data_src:
            _prop2name[prop] = prop_name
            _prop2desc.setdefault(prop, {})[key] = description
            _prop2unit.setdefault(prop, {})[key] = unit
            _prop2source.setdefault(prop, {})[key] = source
            _prop2url.setdefault(prop, {})[key] = url
            _prop2notes.setdefault(prop, {})[key] = notes

        docstring = """Class to store the properties of the elements.

    Attributes
    ----------
    atnum : int
        Atomic number of the element.
    symbol : str
        Chemical symbol of the element.
    name : str
        Name of the element.
    charge : int
        Charge of the atom.
    multiplicity : int
        Multiplicity of the atom.
    """

        # autocomplete the docstring with data from the csv files
        # for each property
        for i in _prop2name:
            # if only one source is available
            if len(_prop2desc[i]) == 1:
                # add line to docstring with property name, type, description, unit, source, and url
                docstring += f"{i} : float\n    {_prop2name[i]} of the element.\n"
                if list(_prop2desc[i].values())[0] != "":
                    docstring += _indent_lines(f"{list(_prop2desc[i].values())[0]}", 4) + "\n"
                if list(_prop2unit[i].values())[0] != "":
                    docstring += (
                        _indent_lines(f"Units: {list(_prop2unit[i].values())[0]}", 4) + "\n"
                    )
                if list(_prop2source[i].values())[0] != "":
                    docstring += _indent_lines(f"{list(_prop2source[i].values())[0]}", 4) + "\n"
                if list(_prop2url[i].values())[0] != "":
                    docstring += _indent_lines(f"{list(_prop2url[i].values())[0]}", 4) + "\n"
                if list(_prop2notes[i].values())[0] != "":
                    docstring += (
                        _indent_lines(f"Notes:\n{list(_prop2notes[i].values())[0]}", 4) + "\n"
                    )
            # if multiple sources are available
            else:
                docstring += (
                    f"{i} : dict\n    Dictionary with the {_prop2name[i]} of the element.\n"
                )
                # add a line docstring with the description, source, and url for each source
                for j in _prop2col[i]:
                    if _prop2desc[i][j] != "":
                        docstring += _indent_lines(f"{j} : {_prop2desc[i][j]}", 4) + "\n"
                    if _prop2unit[i][j] != "":
                        docstring += _indent_lines(f"Units: {_prop2unit[i][j]}", 8) + "\n"
        return docstring

    # Code to run when the module is imported
    data_file = files("atomdb.data").joinpath("elements_data.csv")
    info_file = files("atomdb.data").joinpath("data_info.csv")

    # work with data from "elements_data.csv"
    data = _read_csv(open(data_file))

    # get properties and key sources
    properties = data.pop(0)[3:]
    key_sources = data.pop(0)[3:]
    units = data.pop(0)[3:]

    # separate atnums, symbols, and names from the rest of the data
    atnums, symbols, names = [], [], []
    for row in data:
        atnums.append(int(row[0]))
        symbols.append(row[1])
        names.append(row[2].lower())
        # convert the rest of the data to numbers
        row[:] = [int(i) if i else np.nan for i in row[3:5]] + [
            float(i) if i else np.nan for i in row[5:]
        ]

    # create utility dictionaries to convert between different element identifiers
    sym2num = dict(zip(symbols, atnums))
    name2num = dict(zip(names, atnums))
    num2sym = dict(zip(atnums, symbols))
    num2name = dict(zip(atnums, names))

    # create utility dictionary to locate the columns of the properties
    _prop2col = {}
    for column, (prop, key) in enumerate(zip(properties, key_sources)):
        _prop2col.setdefault(prop, {})[key] = {"ncol": column}

    class Atom:
        def __init__(self, id, charge=0):
            """Create an Atom object.

            Parameters
            ----------
            id : str or int
                The atomic number, symbol, or name of the element.
            charge : int, optional
                The charge of the atom, by default 0.
            """
            # check if the input is a valid element identifier and get the atomic number
            if isinstance(id, str):
                # if it is a is a name
                if len(id) > 3:
                    self.atnum = name2num[id.lower()]
                # if it is a symbol
                else:
                    self.atnum = sym2num[id]
            elif isinstance(id, int) and id in sym2num.values():
                self.atnum = id
            else:
                raise ValueError(
                    "Invalid input for element identifier, must be Z, symbol, or name of element"
                )
            # set the symbol and name of the element
            self.atsym = num2sym[self.atnum]
            self.atname = num2name[self.atnum].capitalize()

            # check if the charge is valid and set the charge
            if not isinstance(charge, int):
                raise ValueError("Charge must be an integer")
            if self.atnum - charge < 0:
                raise ValueError(
                    f"Charge cannot be greater than atomic number. Z = {self.atnum}, charge = {charge}"
                )
            if self.atnum - charge > max(atnums):
                raise ValueError(
                    f"Number of electrons {self.atnum - charge} outside of range of known elements"
                )

            self.charge = charge

            # create a dictionary to store the properties of the element
            tmp_dict = _prop2col.copy()
            for i in tmp_dict:
                # select the multiplicity of isoelectronic element
                if i == "multiplicity":
                    nelectrons = self.atnum - charge
                    # if the number of electrons is 0 get the multiplicity of He (singlet)
                    if nelectrons == 0:
                        nelectrons = 2
                    tmp_dict[i] = int(data[nelectrons - 1][tmp_dict[i][""]["ncol"]])
                elif len(tmp_dict[i]) == 1 and "" in tmp_dict[i]:
                    tmp_dict[i] = data[self.atnum - 1][tmp_dict[i][""]["ncol"]]
                else:
                    tmp_dict[i] = {
                        j: data[self.atnum - 1][tmp_dict[i][j]["ncol"]] for j in tmp_dict[i]
                    }
            # set the class attributes to the properties of the element
            self.__dict__.update(tmp_dict)

    # set the docstring of the Atom class
    Atom.__doc__ = _gendoc(info_file)
