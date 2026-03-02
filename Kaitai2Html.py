"""Convert Kaitai Struct objects to HTML tree representation."""

import types
from enum import Enum, EnumMeta
from typing import Any, List, Optional, Set, Tuple

from dash_extensions.enrich import html
from kaitaistruct import KaitaiStruct, ValidationNotEqualError


def _format_chunk_range(
    start: int, end: int, chunk_length: Optional[int], chunk_offset: int
) -> str:
    """Format chunk range information for display."""
    if chunk_length is None:
        return ""

    start_chunk = chunk_offset + start // chunk_length
    end_chunk = chunk_offset + end // chunk_length

    if start_chunk == end_chunk:
        return f"Chunk {start_chunk}"
    else:
        return f"Chunks {start_chunk} - {end_chunk}"


def _handle_attribute_error(attr: str, error: Exception, struct_type: type) -> html.Label:
    """Handle errors when accessing a Kaitai Struct attribute."""
    if isinstance(error, EOFError):
        return html.Label(
            f"{attr}: <unable to parse!> ({struct_type.__name__})",
            className="tree",
        )
    elif isinstance(error, ValidationNotEqualError):
        return html.Label(
            f'{attr}: {error.actual} - expected: "{error.expected}" ({struct_type.__name__})',
            className="tree",
        )
    else:
        return html.Label(
            f"{attr}: {error} ({struct_type.__name__})",
            className="tree",
        )


def _process_kaitai_attribute(
    attr: str,
    ret_attr: Any,
    kaitai_struct: KaitaiStruct,
    tree: str,
    chunk_length: Optional[int],
    chunk_offset: int,
    seen_set: Set[str],
) -> Tuple[Optional[html.Div], bool]:
    """
    Process a Kaitai Struct attribute.

    Args:
        attr: Attribute name
        ret_attr: Attribute value
        kaitai_struct: The Kaitai Struct object
        tree: Tree path string
        chunk_length: Length of each chunk
        chunk_offset: Starting offset for chunks
        seen_set: Set of already processed attribute strings

    Returns:
        Tuple of (HTML element if created, should_continue flag)
    """
    if not (hasattr(ret_attr, "start") and hasattr(ret_attr, "end")):
        attr_str = attr
    else:
        chunks = _format_chunk_range(ret_attr.start, ret_attr.end, chunk_length, chunk_offset)
        attr_str = f"{attr} (start: {ret_attr.start}, end: {ret_attr.end} - {chunks})"

        if attr_str in seen_set:
            return None, True
        seen_set.add(attr_str)

    next_attr = getattr(kaitai_struct, attr, chunk_length)
    if next_attr == kaitai_struct:
        return None, False

    return (
        html.Div(
            id={"type": "kaitai_struct", "name": tree + "." + attr},
            className="tree",
            children=[
                html.Label(attr_str),
                kaitai2html(
                    getattr(kaitai_struct, attr, chunk_length),
                    tree + "." + attr,
                    chunk_length,
                    chunk_offset,
                ),
            ],
        ),
        False,
    )


def _process_list_attribute(
    attr: str,
    ret_attr: list,
    tree: str,
    chunk_length: Optional[int],
    chunk_offset: int,
) -> html.Div:
    """Process a list attribute containing Kaitai Struct items."""
    childs = []
    for i, item in enumerate(ret_attr):
        if isinstance(item, KaitaiStruct):
            childs.append(
                kaitai2html(
                    item,
                    tree + "." + attr + "[" + str(i) + "]",
                    chunk_length,
                    chunk_offset,
                )
            )

    return html.Div(
        id={"type": "kaitai_struct", "name": tree + "." + attr},
        className="tree",
        children=[html.Label(attr), html.Div(childs)],
    )


def _should_skip_attribute(attr: str, ret_attr: Any) -> bool:
    """Check if an attribute should be skipped."""
    if attr.startswith("_") or attr in ["start", "end"]:
        return True
    if isinstance(ret_attr, (Enum, EnumMeta, types.MethodType, type)):
        return True
    return False


def kaitai2html(
    kaitai_struct: KaitaiStruct,
    tree: Optional[str] = None,
    chunk_length: Optional[int] = None,
    chunk_offset: int = 0,
) -> html.Div:
    """
    Convert a Kaitai Struct object to an HTML tree representation.

    Args:
        kaitai_struct: The Kaitai Struct object to convert
        tree: Tree path string for nested structures
        chunk_length: Length of each chunk for offset calculation
        chunk_offset: Starting offset for chunk numbering

    Returns:
        HTML Div element representing the structure tree
    """
    if tree is None:
        tree = "root"

    seen_set: Set[str] = set()
    top_level_entries: List[html.Label | html.Div] = []

    for attr in dir(kaitai_struct):
        if _should_skip_attribute(attr, getattr(type(kaitai_struct), attr, None)):
            continue

        try:
            ret_attr = getattr(kaitai_struct, attr)
        except Exception as err:
            top_level_entries.append(_handle_attribute_error(attr, err, type(kaitai_struct)))
            continue

        if isinstance(ret_attr, KaitaiStruct):
            entry, should_continue = _process_kaitai_attribute(
                attr, ret_attr, kaitai_struct, tree, chunk_length, chunk_offset, seen_set
            )
            if entry:
                top_level_entries.append(entry)
        elif isinstance(ret_attr, list):
            top_level_entries.append(
                _process_list_attribute(attr, ret_attr, tree, chunk_length, chunk_offset)
            )
        else:
            top_level_entries.append(
                html.Label(
                    f"{attr}: {ret_attr} ({type(kaitai_struct).__name__})",
                    className="tree",
                )
            )

    return html.Div(
        id={"type": "kaitai_struct", "index": tree + f" ({kaitai_struct.__repr__()})"},
        className="tree",
        children=top_level_entries,
    )
