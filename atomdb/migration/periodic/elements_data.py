import tables as pt
import pandas as pd
import numpy as np


class basic_properties(pt.IsDescription):
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


class property_values(pt.IsDescription):
    source = pt.StringCol(30, pos=0)
    unit = pt.StringCol(20, pos=1)
    value = pt.Float64Col(pos=2)


class data_info(pt.IsDescription):
    property_key = pt.StringCol(20, pos=0)
    property_name = pt.StringCol(25, pos=1)
    source_key = pt.StringCol(30, pos=2)
    property_description = pt.StringCol(250, pos=3)
    reference = pt.StringCol(250, pos=4)
    doi = pt.StringCol(150, pos=4)
    notes = pt.StringCol(600, pos=5)


def create_data_for_tables(hdf5_file, parent_folder, table_name, table_description, row_description, columns, row_data, sources_data, units_data):
    table = hdf5_file.create_table(parent_folder, table_name, row_description, table_description)

    for col in columns:
        if pd.notna(row_data[col]):
            source = sources_data.get(col, 'unknown')
            unit = units_data.get(col, 'unknown')
            value = np.nan

            if pd.notna(row_data[col]) and str(row_data[col]).strip():
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
    return table


csv_file = '/home/enjy/work/Gsoc/repo/AtomDB/atomdb/data/elements_data.csv'
hdf5_file = "elements_data.h5"

property_configs = [
    {'property': 'cov_radius', 'group': 'size', 'table_name': 'cov_radius', 'description': 'Covalent Radius'},

    {'property': 'vdw_radius', 'group': 'size', 'table_name': 'vdw_radius', 'description': 'Van der Waals Radius'},

    {'property': 'at_radius', 'group': 'size', 'table_name': 'at_radius', 'description': 'Atomic Radius'},

    {'property': 'mass', 'group': None, 'table_name': 'mass', 'description': 'Atomic Mass'},

    {'property': 'pold', 'group': None, 'table_name': 'pold', 'description': 'Polarizability'},

    {'property': 'c6', 'group': None, 'table_name': 'c6', 'description': 'C6 Dispersion Coefficient'},

    {'property': 'eneg', 'group': None, 'table_name': 'eneg', 'description': 'Electronegativity'}
]



elements_data_df = pd.read_csv(csv_file, comment='#', header=None, dtype=str)
elements_data_df.dropna(how='all', inplace=True)



headers = elements_data_df.iloc[0].str.strip().to_list()
unique_headers = []
header_counts = {}
for header in headers:
    if header in header_counts:
        header_counts[header] += 1
        unique_headers.append(f"{header}.{header_counts[header]}")
    else:
        header_counts[header] = 0
        unique_headers.append(header)

sources = elements_data_df.iloc[1].str.strip().tolist()
units = elements_data_df.iloc[2].str.strip().tolist()
data = elements_data_df.iloc[3:].reset_index(drop=True)
data.columns = unique_headers


sources_data = dict(zip(unique_headers, sources))
units_data = dict(zip(unique_headers, units))




try:
    with pt.open_file(hdf5_file, mode='w', title='Periodic Data') as h5file:
        elements_group = h5file.create_group('/', 'elements', 'Elements Data' )

        for idx, row in data.iterrows():
            atnum = int(row['atnum']) if pd.notna(row['atnum']) else 0
            symbol = row['symbol'] if pd.notna(row['symbol']) else ''
            name = row['name'] if pd.notna(row['name']) else ''
            group = int(row['group']) if pd.notna(row['group']) else 0
            period = int(row['period']) if pd.notna(row['period']) else 0
            mult = int(row['mult']) if pd.notna(row['mult']) else 0

            element_group_name = f"Element_{atnum:03d}_{name}"
            element_group = h5file.create_group(elements_group, element_group_name, f'Data for {name}')

            basic_properties_table =h5file.create_table(element_group, 'basic_properties', basic_properties, 'Basic Properties')
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
            size_group_created = False
            size_group = None

            for config in property_configs:
                columns = [col for col in unique_headers if col.startswith(config['property'])]

                if not columns:
                    continue


                if config['group']:
                    if not size_group_created:
                        size_group = h5file.create_group(element_group, 'size', 'Size Properties')
                        size_group_created = True
                    parent = size_group
                else:
                    parent = element_group

                create_data_for_tables(h5file, parent, config['table_name'], config['description'], property_values, columns, row, sources_data, units_data)



except Exception as e:
    print(f"Error creating HDF5 file: {e}")
    exit(1)



















