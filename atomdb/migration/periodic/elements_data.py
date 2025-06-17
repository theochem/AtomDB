import csv
import tables as pt
import numpy as np
from importlib_resources import \
files
import warnings

# Suppresses NaturalNameWarning warnings from PyTables.
warnings.filterwarnings('ignore', category=pt.NaturalNameWarning)


# Set-up variables
elements_data_csv = files("atomdb.data").joinpath("elements_data.csv")
data_info_csv = files("atomdb.data").joinpath("data_info.csv")
hdf5_file = "elements_data.h5"


# Properties of each element in the HDF5 file.
PROPERTY_CONFIGS = [
    {
        'property': 'cov_radius',
        'group': 'Radius',
        'table_name': 'cov_radius',
        'description': 'Covalent Radius'
    },
    {
        'property': 'vdw_radius',
        'group': 'Radius',
        'table_name': 'vdw_radius',
        'description': 'Van der Waals Radius'
    },
    {
        'property': 'at_radius',
        'group': 'Radius',
        'table_name': 'at_radius',
        'description': 'Atomic Radius'
    },
    {
        'property': 'mass',
        'group': None,
        'table_name': 'atmass',
        'description': 'Atomic Mass'
    },
    {
        'property': 'pold',
        'group': None,
        'table_name': 'polarizability',
        'description': 'Polarizability'
    },
    {
        'property': 'c6',
        'group': None,
        'table_name': 'dispersion_c6',
        'description': 'C6 Dispersion Coefficient'
    },
    {
        'property': 'eneg',
        'group': None,
        'table_name': 'Energy',
        'description': 'Electronegativity'
    }
]


# Periodic tables data schema definitions
class ElementDescription(pt.IsDescription):
    """Schema for the basic_properties table for each element."""
    atnum = pt.Int32Col(pos=0)
    symbol = pt.StringCol(2, pos=1)
    name = pt.StringCol(25, pos=2)
    group = pt.Int32Col(pos=3)
    period = pt.Int32Col(pos=4)
    mult = pt.Int32Col(pos=5)


class PropertyValues(pt.IsDescription):
    """Schema for property value tables."""
    source = pt.StringCol(30, pos=0)
    unit = pt.StringCol(20, pos=1)
    value = pt.Float64Col(pos=2)


class ElementsDataInfo(pt.IsDescription):
    """Schema for the property_info table."""
    property_key = pt.StringCol(20, pos=0)
    property_name = pt.StringCol(50, pos=1)
    source_key = pt.StringCol(30, pos=2)
    property_description = pt.StringCol(250, pos=3)
    reference = pt.StringCol(250, pos=4)
    doi = pt.StringCol(150, pos=5)
    notes = pt.StringCol(500, pos=6)


def create_data_for_tables(hdf5_file, parent_folder, table_name, table_description, row_description, columns, row_data, sources_data, units_data):
    """
        Create a table in the HDF5 file for a specific properties.

        Args:
            hdf5_file: PyTables file object.
            parent_folder: Group where the table will be created.
            table_name (str): Name of the table.
            table_description (str): Description of the table.
            row_description: PyTables IsDescription class for the table schema.
            columns (list): List of column names from the CSV to include.
            row_data (dict): Data for the current element.
            sources_data (dict): sources of each property.
            units_data (dict): units of each property.
        """

    # Creates a new table in the HDF5 file.
    table = hdf5_file.create_table(parent_folder, table_name, row_description, table_description)

    # Iterates over the list of columns relevant to the current table.
    for col in columns:
        source = sources_data.get(col, 'unknown') # defaulting to 'unknown' if not found.
        unit = units_data.get(col, 'unknown')     # defaulting to 'unknown' if not found.
        value = np.nan

        if col in row_data and row_data[col].strip():
            try:
                value = float(row_data[col])
            except (ValueError, TypeError):
                value = np.nan

        # Creates a new row in the table.
        row = table.row
        row['source'] = source.encode('utf-8') if source else ''
        row['unit'] = unit.encode('utf-8') if unit else ''
        row['value'] = value
        row.append()

    # Flushes the table to ensure all data is written to the HDF5 file.
    table.flush()


def read_elements_data_csv(elements_data_csv):
    """
        Read the elements_data.csv file.

        Args:
            elements_data_csv: Path to the elements_data.csv file.

        Returns:
            - data: List of dictionaries containing element data.
            - unique_headers: List of unique column headers.
            - sources_data (dict): sources of each property.
            - units_data (dict): units of each property.
        """

    # Opens the csv file, filters out comment lines (starting with #) and empty lines.
    with open(elements_data_csv, 'r') as f:
        reader = csv.reader(f)
        lines = [line for line in reader if not line[0].startswith('#') and any(line)]

    headers = [header.strip() for header in lines[0]] # first row as column headers
    sources = [source.strip() for source in lines[1]] # second row as sources
    units = [unit.strip() for unit in lines[2]]       # third row as units
    data_rows = lines[3:]                             # remaining rows as data

    # Process headers to make them unique
    unique_headers = []
    header_counts = {}
    for header in headers:
        if header in header_counts:
            header_counts[header] += 1
            unique_headers.append(f"{header}.{header_counts[header]}") # creates suffix (header.1, header.2) for duplicate headers
        else:
            header_counts[header] = 0
            unique_headers.append(header)

    # Create data as list of dictionaries
    data = []
    for row in data_rows:
        data.append(dict(zip(unique_headers, row)))

    sources_data = dict(zip(unique_headers, sources))
    units_data = dict(zip(unique_headers, units))

    return data, unique_headers, sources_data, units_data



def read_data_info_csv(data_info_csv):
    """
    Read and parse the data_info.csv file containing metadata.

    Args:
        data_info_csv: Path to the data_info.csv file.

    Returns:
        data_info: List of dictionaries containing metadata for each property.
    """
    # Opens the csv file, filters out comment lines (starting with #) and empty lines.
    with open(data_info_csv, 'r') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                lines.append(stripped)

        # hardcode the headers
        data_info_headers = [
            'Property key',
            'Property name',
            'Source key',
            'Property description',
            'Reference',
            'doi',
            'Notes'
        ]

        reader = csv.reader(lines)
        data_rows = list(reader)

        data_info = []
        for row in data_rows:
            data_info.append(dict(zip(data_info_headers, row)))

    return data_info


def write_elements_data_to_hdf5(data, unique_headers, sources_data, units_data):
    """
    Write element data to an HDF5 file.

    Args:
        data: List of dictionaries containing element data.
        unique_headers: List of unique column headers.
        sources_data (dict): sources of each property.
        units_data (dict): units of each property.
    """

    # Open a file in "w"rite mode
    h5file = pt.open_file(hdf5_file, mode="w", title='Periodic Data')

    # Create the Elements group
    elements_group = h5file.create_group('/', 'Elements', 'Elements Data')

    for row in data:
        atnum = int(row['atnum']) if 'atnum' in row and row['atnum'].strip() else 0
        symbol = row['symbol'] if 'symbol' in row and row['symbol'].strip() else ''
        name = row['name'] if 'name' in row and row['name'].strip() else ''
        group = int(row['group']) if 'group' in row and row['group'].strip() else 0
        period = int(row['period']) if 'period' in row and row['period'].strip() else 0
        mult = int(row['mult']) if 'mult' in row and row['mult'].strip() else 0

        # Create a new group
        element_group_name = f"{atnum:03d}"
        element_group = h5file.create_group(elements_group, element_group_name, f'Data for {name}')

        # Create the basic properties table and fill it with data
        basic_properties_table = h5file.create_table(element_group, 'basic_properties', ElementDescription, 'Basic Properties')
        basic_properties_row = basic_properties_table.row
        basic_properties_row['atnum'] = atnum
        basic_properties_row['symbol'] = symbol.encode('utf-8') if symbol else ''
        basic_properties_row['name'] = name.encode('utf-8') if name else ''
        basic_properties_row['group'] = group
        basic_properties_row['period'] = period
        basic_properties_row['mult'] = mult
        basic_properties_row.append()
        basic_properties_table.flush()

        # flag to track whether a radius group has been created for the element.
        radius_group_created = False
        radius_group = None

        # For each property configuration, finds all columns that start with the property’s prefix.
        for config in PROPERTY_CONFIGS:
            columns = [col for col in unique_headers if col.startswith(config['property'])]

            if not columns:
                continue

            if config['group']:
                if not radius_group_created:
                    radius_group = h5file.create_group(element_group, 'Radius', 'Radius Properties')
                    radius_group_created = True
                parent = radius_group
            else:
                parent = element_group

            create_data_for_tables(h5file, parent, config['table_name'], config['description'],
                                   PropertyValues, columns, row, sources_data, units_data)

    # Close the file
    h5file.close()



def write_data_info_to_hdf5(data_info_list):
    """
        Write dara from data_info.csv to the HDF5 file.

        Args:
            data_info_list: List of dictionaries containing metadata.
        """


    # Opens the HDF5 file in append mode ("a") --> add metadata without overwriting existing data.
    with pt.open_file(hdf5_file, mode='a', title='Periodic Data') as h5file:
        data_info_group = h5file.create_group('/', 'data_info', 'Data Info')

        property_info_table = h5file.create_table(data_info_group, 'property_info', ElementsDataInfo,'Property Information')

        for row in data_info_list:
            table_row = property_info_table.row
            table_row['property_key'] = row.get('Property key', '').encode('utf-8')
            table_row['property_name'] = row.get('Property name', '').encode('utf-8')
            table_row['source_key'] = row.get('Source key', '').encode('utf-8')
            table_row['property_description'] = row.get('Property description', '').encode('utf-8')
            table_row['reference'] = row.get('Reference', '').encode('utf-8')
            table_row['doi'] = row.get('doi', '').encode('utf-8')
            table_row['notes'] = row.get('Notes', '').encode('utf-8')
            table_row.append()
        property_info_table.flush()



if __name__ == "__main__":
    # Read the elements data from the CSV file
    data, unique_headers, sources_data, units_data = read_elements_data_csv(elements_data_csv)

    # Read the provenance data from the CSV file
    data_info_df = read_data_info_csv(data_info_csv)

    # Write the periodic table data to an HDF5 file
    write_elements_data_to_hdf5(data, unique_headers, sources_data, units_data)

    # Write the provenance data to the HDF5 file
    write_data_info_to_hdf5(data_info_df)

