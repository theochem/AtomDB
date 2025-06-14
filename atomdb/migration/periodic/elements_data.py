import tables as pt

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
