import csv
import tables as pt
# import pandas as pd  ## Note: consider using csv module instead of pandas
import numpy as np
from importlib_resources import \
files  ## Note: alternatively use python path library, see utils.py file

import warnings
warnings.filterwarnings('ignore', category=pt.NaturalNameWarning)

## Note: suggestion, place set-up variables at the top of the file
elements_data_csv = files("atomdb.data").joinpath("elements_data.csv")  ## Note: replace harcoded paths to make script more portable
data_info_csv = files("atomdb.data").joinpath("data_info.csv")
hdf5_file = "elements_data.h5"

property_configs = [
    {'property': 'cov_radius', 'group': 'Radius', 'table_name': 'cov_radius',
     'description': 'Covalent Radius'},

    {'property': 'vdw_radius', 'group': 'Radius', 'table_name': 'vdw_radius',
     'description': 'Van der Waals Radius'},

    {'property': 'at_radius', 'group': 'Radius', 'table_name': 'at_radius',
     'description': 'Atomic Radius'},

    {'property': 'mass', 'group': None, 'table_name': 'atmass', 'description': 'Atomic Mass'},

    {'property': 'pold', 'group': None, 'table_name': 'polarizability', 'description': 'Polarizability'},

    {'property': 'c6', 'group': None, 'table_name': 'dispersion_c6',
     'description': 'C6 Dispersion Coefficient'},

    {'property': 'eneg', 'group': None, 'table_name': 'eneg', 'description': 'Electronegativity'}
]


# Periodic table data schema definition
class ElementDescription(pt.IsDescription):
    """
    pos: pos of the property in the column
    pt.StringCol(2): max num of characters
    """
    atnum = pt.Int32Col(pos=0)
    symbol = pt.StringCol(2, pos=1)
    name = pt.StringCol(25, pos=2)
    group = pt.Int32Col(pos=3)
    period = pt.Int32Col(pos=4)
    mult = pt.Int32Col(pos=5)


class PropertyValues(pt.IsDescription):
    source = pt.StringCol(30, pos=0)
    unit = pt.StringCol(20, pos=1)
    value = pt.Float64Col(pos=2)


class ElementsDataInfo(pt.IsDescription):
    property_key = pt.StringCol(20, pos=0)
    property_name = pt.StringCol(50, pos=1)
    source_key = pt.StringCol(30, pos=2)
    property_description = pt.StringCol(250, pos=3)
    reference = pt.StringCol(250, pos=4)
    doi = pt.StringCol(150, pos=4)
    notes = pt.StringCol(600, pos=5)


def create_data_for_tables(hdf5_file, parent_folder, table_name, table_description, row_description,
                           columns, row_data, sources_data, units_data):
    table = hdf5_file.create_table(parent_folder, table_name, row_description, table_description)

    for col in columns:
        source = sources_data.get(col, 'unknown')
        unit = units_data.get(col, 'unknown')
        value = np.nan

        if col in row_data and row_data[col].strip():
            try:
                value = float(row_data[col])
            except (ValueError, TypeError):
                value = np.nan

        row = table.row
        row['source'] = source.encode('utf-8') if source else ''
        row['unit'] = unit.encode('utf-8') if unit else ''
        row['value'] = value
        row.append()

    table.flush()
    # return table   ## Note: not needed, as the table is already created in the hdf5 file


def read_elements_data_csv(elements_data_csv):
    with open(elements_data_csv, 'r') as f:
        reader = csv.reader(f)
        lines = [line for line in reader if not line[0].startswith('#') and any(line)]

    headers = [h.strip() for h in lines[0]]
    sources = [s.strip() for s in lines[1]]
    units = [u.strip() for u in lines[2]]
    data_rows = lines[3:]

    # Process headers to make them unique
    unique_headers = []
    header_counts = {}
    for header in headers:
        if header in header_counts:
            header_counts[header] += 1
            unique_headers.append(f"{header}.{header_counts[header]}")
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
    with open(data_info_csv, 'r') as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            if line and not line[0].startswith('#'):
                lines.append([item.strip() for item in line])

    # Get headers (first row)
    headers = [h.lstrip('#').strip() for h in lines[0]]
    data_rows = lines[1:]

    # Create list of dictionaries
    data_info = []
    for row in data_rows:
        data_info.append(dict(zip(headers, row)))

    return data_info



def write_elements_data_to_hdf5(data, unique_headers, sources_data, units_data):
    """
    Write the periodic table data to an HDF5 file.
    """

    # Open a file in "w"rite mode
    # with pt.open_file(hdf5_file, mode='w', title='Periodic Data') as h5file:
    h5file = pt.open_file(hdf5_file, mode="w",
                          title='Periodic Data')  ## Note: removing one indentation level

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
        element_group_name = f"{atnum:03d}"  ## Note: change to just label by atomic number, e.g. 001
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

        # flag to track if we need to create size folder or it already exists
        radius_group_created = False
        radius_group = None

        for config in property_configs:
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

