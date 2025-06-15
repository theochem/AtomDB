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
    value = pt.StringCol(pos=2)


class data_info(pt.IsDescription):
    property_key = pt.StringCol(20, pos=0)
    property_name = pt.StringCol(25, pos=1)
    source_key = pt.StringCol(30, pos=2)
    property_description = pt.StringCol(250, pos=3)
    reference = pt.StringCol(250, pos=4)
    doi = pt.StringCol(150, pos=4)
    notes = pt.StringCol(600, pos=5)


property_configs = [
    {'category': 'cov_radius', 'group': 'size', 'table_name': 'cov_radius', 'description': 'Covalent Radius'},

    {'category': 'vdw_radius', 'group': 'size', 'table_name': 'vdw_radius', 'description': 'Van der Waals Radius'},

    {'category': 'at_radius', 'group': 'size', 'table_name': 'at_radius', 'description': 'Atomic Radius'},

    {'category': 'mass', 'group': None, 'table_name': 'mass', 'description': 'Atomic Mass'},

    {'category': 'pold', 'group': None, 'table_name': 'pold', 'description': 'Polarizability'},

    {'category': 'c6', 'group': None, 'table_name': 'c6', 'description': 'C6 Dispersion Coefficient'},

    {'category': 'eneg', 'group': None, 'table_name': 'eneg', 'description': 'Electronegativity'}
]



elements_data_df = pd.read_csv('atomdb/data/elements_data.csv', comment='#', header=None, dtype=str)
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

            row = table.row()
            row['source'] = source.encode('utf-8') if source else ''
            row['unit'] = unit.encode('utf-8') if unit else ''
            row['value'] = value
            row.append()

    table.flush()
    return table















