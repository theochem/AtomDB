import msgpack


__all__ = [
    "dump_msg",
    "load_msg",
]


pack = msgpack.Packer(use_bin_type=True)
unpack = msgpack.Unpacker(use_list=False, raw=True, strict_map_key=True)


def dump_msg(data_dict, fn):
    r"""Serialize (dump) an object to a msgpack file."""
    with open(fn, "wb") as f:
        pack(data_dict, f)


def load_msg(fn):
    r"""Deserialize (load) an object from a msgpack file."""
    with open(fn, "rb") as f:
        data_dict = unpack(f)
    return data_dict
