import logging
import types
from enum import Enum, EnumMeta
from kaitaistruct import KaitaiStruct, ValidationNotEqualError
from dash_extensions.enrich import html

seen_set = set()


def kaitai2html(kaitai_struct, tree=None, chunk_length=None, chunk_offset=0):
    if tree is None:
        seen_set.clear()
        tree = ""
    html_str = html.Label("No entries found")
    # iterate over all attributes of the kaitai_struct
    top_level_entries = []
    for attr in dir(kaitai_struct):
        if attr.startswith("_") or attr in ["start", "end"]:
            continue
        # check if the attribute is a kaitai_struct
        try:
            ret_attr = getattr(kaitai_struct, attr)
        except EOFError:
            top_level_entries.append(
                html.Label("{}: {} ({})".format(attr, "<unable to parse!>", type(kaitai_struct).__name__),
                           className="tree"))
            continue
        except ValidationNotEqualError as err:
            top_level_entries.append(
                html.Label(
                    '{}: {} - expected: "{}" ({})'.format(attr, err.actual, err.expected, type(kaitai_struct).__name__),
                    className="tree"))
            continue
        except Exception as err:
            top_level_entries.append(html.Label("{}: {} ({})".format(attr, err, type(kaitai_struct).__name__),
                                                className="tree"))
            continue
        if isinstance(ret_attr, KaitaiStruct):
            # if yes, call kaitai2html on it
            if hasattr(ret_attr, "start") and hasattr(ret_attr, "end"):
                if chunk_length is not None:
                    start_chunk = chunk_offset + ret_attr.start // chunk_length
                    end_chunk = chunk_offset + ret_attr.end // chunk_length
                    if start_chunk == end_chunk:
                        chunks = f"Chunk {start_chunk}"
                    else:
                        chunks = f"Chunks {start_chunk} - {end_chunk}"
                else:
                    chunks = ""
                attr_str = f"{attr} (start: {ret_attr.start}, end: {ret_attr.end} - {chunks})"
                if attr_str in seen_set:
                    continue
                else:
                    seen_set.add(attr_str)
            else:
                attr_str = f"{attr}"
            next_attr = getattr(kaitai_struct, attr, chunk_length)
            if next_attr != kaitai_struct:
                top_level_entries.append(html.Div(id={'type': 'kaitai_struct', 'name': tree + "." + attr},
                                                  className="tree",
                                                  children=[html.Label(attr_str),
                                                            kaitai2html(getattr(kaitai_struct, attr, chunk_length),
                                                                        tree + "." + attr, chunk_length,
                                                                        chunk_offset)]))
        elif isinstance(ret_attr, Enum):
            continue
        elif isinstance(ret_attr, EnumMeta):
            continue
        elif isinstance(ret_attr, types.MethodType):
            continue
        elif isinstance(ret_attr, type):
            continue
        elif isinstance(ret_attr, list):
            childs = []
            for i, item in enumerate(ret_attr):
                if isinstance(item, KaitaiStruct):
                    childs.append(kaitai2html(item, tree + "." + attr + "[" + str(i) + "]", chunk_length, chunk_offset))
            top_level_entries.append(html.Div(id={'type': 'kaitai_struct', 'name': tree + "." + attr},
                                              className="tree", children=[html.Label(attr), html.Div(childs)]))
        else:
            # if no, add the attribute to the html
            top_level_entries.append(
                html.Label("{}: {} ({})".format(attr, ret_attr, type(kaitai_struct).__name__),
                           className="tree"))
    return html.Div(id={'type': 'kaitai_struct', 'index': tree + f" ({kaitai_struct.__repr__()})"},
                    className="tree", children=top_level_entries)
