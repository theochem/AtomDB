from importlib_resources import files  # Importing files for accessing package resources
from atomdb.utils import CONVERTOR_TYPES  # Importing conversion types from AtomDB utilities

__all__ = [
    "Element",
    "element_number",
    "element_symbol",
    "element_name",
]


def setup_element():
    """Generate the `Element` class and helper functions."""

    # Extract element data and metadata
    data, props, srcs, units, prop2col, num2str, str2num = get_data()
    prop2name, prop2desc, prop2src, prop2url, prop2note = get_info()

    # Function to retrieve atomic number from element symbol/name
    def element_number(elem):
        """
        Return the element number from a string or integer.

        Parameters:
        elem (str | int): Symbol, name, or number of an element.

        Returns:
        int: Atomic number of the element.
        """
        if isinstance(elem, str):
            return str2num[elem]  # Convert symbol or name to atomic number
        else:
            atnum = int(elem)
            if atnum not in num2str:
                raise ValueError(f"Invalid element number: {atnum}")
            return atnum

    # Function to get the symbol of an element
    def element_symbol(elem):
        """
        Return the element symbol from a string or integer.

        Parameters:
        elem (str | int): Symbol, name, or number of an element.

        Returns:
        str: Element symbol.
        """
        return num2str[element_number(elem)][0]  # Convert element to symbol

    # Function to get the full name of an element
    def element_name(elem):
        """
        Return the element name from a string or integer.

        Parameters:
        elem (str | int): Symbol, name, or number of an element.

        Returns:
        str: Element name.
        """
        return num2str[element_number(elem)][1]  # Convert element to name

    # Class constructor for `Element`
    def init(self, elem):
        """
        Initialize an `Element` instance.

        Parameters:
        elem (str | int): Symbol, name, or number of an element.
        """
        self._atnum = element_number(elem)  # Store atomic number as an instance variable

    # Define properties for Element class
    @property
    def atnum(self):
        """Returns atomic number of the element."""
        return self._atnum

    @property
    def symbol(self):
        """Returns symbol of the element."""
        return element_symbol(self._atnum)

    @property
    def name(self):
        """Returns name of the element."""
        return element_name(self._atnum)

    # Define attributes for the dynamically created `Element` class
    attrs = {
        "__init__": init,
        "atnum": atnum,
        "symbol": symbol,
        "name": name,
    }

    # Generate properties dynamically based on CSV data
    for prop, name in prop2name.items():
        short = f"{name} of the element."

        # Handle cases where there are multiple data sources
        if len(prop2col[prop]) == 1 and "" in prop2col[prop]:
            t = type(data[0][prop2col[prop][""]]).__name__
            sig = f"{prop} : {t}"
            long = ""
        else:
            t = type(data[0][next(iter(prop2col[prop].values()))]).__name__
            sig = f"{prop} : Dict[{t}]"
            long = "\nNotes\n-----\nThis property is a dictionary with source keys."

        # Generate class property dynamically
        f = make_property(data, prop, prop2col)
        f.__doc__ = f"{short}\n\nReturns\n-------\n{sig}\n{long}"

        attrs[prop] = f  # Add property to attributes dictionary

    attrs["__doc__"] = "Element properties with attributes like atomic number, symbol, and name."

    # Dynamically create `Element` class
    Element = type("Element", (object,), attrs)

    return Element, element_number, element_symbol, element_name


def read_csv(file):
    """Read a CSV file into a list of lists while ignoring empty lines and comments."""
    lines = []
    with open(file) as f:
        for row in reader(f):
            # Ignore empty lines and comment lines
            if row and not row[0].lstrip().startswith("#") and any(c not in ", \n" for c in row):
                lines.append([i.replace("\\n", "\n") for i in row])  # Replace "\n" escape sequences
    return lines


def get_data():
    """Extract the contents of `data/elements_data.csv`."""
    data = read_csv(files("atomdb.data").joinpath("elements_data.csv"))

    # Extract property headers and metadata
    props = data.pop(0)[3:]  # Skip first 3 columns (atomic number, symbol, name)
    srcs = data.pop(0)[3:]
    units = data.pop(0)[3:]

    # Create a mapping from property names to their corresponding column indices
    prop2col = {}
    for col, (prop, key) in enumerate(zip(props, srcs)):
        prop2col.setdefault(prop, {})[key] = col

    # Maps for converting symbols and names to atomic numbers
    num2str, str2num = {}, {}
    for row in data:
        atnum = int(row.pop(0))
        symbol = row.pop(0).title()
        name = row.pop(0).title()
        num2str[atnum] = (symbol, name)
        str2num[symbol] = atnum
        str2num[name] = atnum
        str2num[name.lower()] = atnum  # Support lowercase names

        # Convert string data to numerical values where applicable
        for i, (unit, val) in enumerate(zip(units, row)):
            row[i] = CONVERTOR_TYPES[unit](val) if val else None

    return data, props, srcs, units, prop2col, num2str, str2num


def get_info():
    """Extract metadata from `data/data_info.csv`."""
    info = read_csv(files("atomdb.data").joinpath("data_info.csv"))

    # Dictionaries to store additional information
    prop2name, prop2desc, prop2src, prop2url, prop2note = {}, {}, {}, {}, {}
    for prop, name, key, desc, src, url, note in info:
        prop2name[prop] = name
        prop2desc.setdefault(prop, {})[key] = desc
        prop2src.setdefault(prop, {})[key] = src
        prop2url.setdefault(prop, {})[key] = url
        prop2note.setdefault(prop, {})[key] = note

    return prop2name, prop2desc, prop2src, prop2url, prop2note


def make_property(data, prop, prop2col):
    """Dynamically generate properties for the `Element` class."""
    if len(prop2col[prop]) == 1 and "" in prop2col[prop]:
        def f(self):
            return data[self._atnum - 1][prop2col[prop][""]]
    else:
        def f(self):
            row = data[self._atnum - 1]
            return {k: row[v] for k, v in prop2col[prop].items() if row[v] is not None}
    return property(f)


# Generate the `Element` class and helper functions
Element, element_number, element_symbol, element_name = setup_element()
