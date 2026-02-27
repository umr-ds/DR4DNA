# -*- coding: utf-8 -*-
# Run this app using: `python app.py <file.ini>` and
# visit http://127.0.0.1:8050/ in your web browser.
"""
DR4DNA - DNA Data Reconstruction Application

This application provides a semi-automatic reconstruction toolkit for
DNA data storage experiments. It allows users to identify and repair
corrupted packets in DNA-encoded data files.

Usage:
    python app.py <config.ini>

Example:
    python app.py eval/sleeping_beauty_RU10_w_error_correction_v1.ini
"""

import argparse
import string
import typing

import dash_extensions.enrich as dash
import numpy as np
from dash import ctx
from dash_extensions.enrich import (
    ALL,
    MATCH,
    DashProxy,
    Input,
    MultiplexerTransform,
    Output,
    State,
    dcc,
    html,
)

from app_callbacks import callbacks
from callback_handlers import ButtonCallbackHandler, PluginCallbackHandler, RepairCallbackHandler
from constants import (
    CHUNK_TAG_INVALID,
    CHUNK_TAG_UNDECODED,
    CHUNK_TAG_VALID,
    COLOR_CORRECT_BUTTON,
    COLOR_CORRECT_COLORBLIND,
    COLOR_INCORRECT_BUTTON,
    COLOR_INCORRECT_COLORBLIND,
    COLOR_LIGHT_RED_BUTTON,
    COLOR_WHITE_BUTTON,
    COLOR_YELLOW_BUTTON,
    EXTERNAL_STYLESHEETS,
    LAST_CHUNK_LEN_FORMAT,
    META_TAGS,
)
from layout import gen_app_layout
from logger import get_logger
from NOREC4DNA.ConfigWorker import ConfigReadAndExecute
from repair_algorithms import *  # NOSONAR - Required to load/register all plugins  # noqa: F403, F401
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from repair_algorithms.PluginManager import PluginManager  # noqa: F401 - Used implicitly
from semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit
from state import AppState, get_app_state, initialize_app_state  # noqa: F401 - Used implicitly

logger = get_logger(__name__)

# Global callback handler instances (will be initialized after globals are set up)
plugin_handler: typing.Optional[PluginCallbackHandler] = None
button_handler: typing.Optional[ButtonCallbackHandler] = None
repair_handler: typing.Optional[RepairCallbackHandler] = None


def update_point(trace: typing.Any, points: typing.Any, selector: typing.Any) -> typing.Any:
    """Handle canvas point selection."""
    logger.debug(f"Point selected: trace={trace}, points={points}, selector={selector}")
    return points.point_inds


# Button style constants
BUTTON_STYLE_WHITE = COLOR_WHITE_BUTTON
BUTTON_STYLE_RED = COLOR_INCORRECT_BUTTON
BUTTON_STYLE_GREEN = COLOR_CORRECT_BUTTON
BUTTON_STYLE_LIGHT_RED = COLOR_LIGHT_RED_BUTTON
BUTTON_STYLE_YELLOW = COLOR_YELLOW_BUTTON

# Current button styles (can be changed by colorblind mode)
correct_button_style = COLOR_CORRECT_BUTTON
incorrect_button_style = COLOR_INCORRECT_BUTTON

colorblind_correct = COLOR_CORRECT_COLORBLIND
colorblind_incorrect = COLOR_INCORRECT_COLORBLIND

# Module-level UI state (Dash components - not serialized in AppState)
canvas_list = []
child = []
force_load_plugins = []
all_plugins_childs = []

app = DashProxy(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    meta_tags=META_TAGS,
    prevent_initial_callbacks=True,
    transforms=[MultiplexerTransform()],
    suppress_callback_exceptions=True,
)

input_callback_handler = [
    Input("repair-button", "n_clicks"),
    Input("repair-reorder-button-possible", "n_clicks"),
    Input("repair-id-input-box", "value"),
    Input("hex-repair-input", "value"),
    Input("txt-repair-input", "value"),
    Input("repair-chunks-button", "n_clicks"),
    Input("analyze-button", "n_clicks"),
    Input("repair-exclusion-button", "n_clicks"),
    Input("repair-reorder-button", "n_clicks"),
    Input("reset-chunk-tag-button", "n_clicks"),
    Input("calculate-rank-button", "n_clicks"),
    Input("save-button", "n_clicks"),
    Input("packet-tag-chunk-invalid-button", "n_clicks"),
    Input("packet-tag-chunk-valid-button", "n_clicks"),
    Input("mode-switch", "value"),
    Input("colorblind-switch", "value"),
    Input({"type": "forceload-plugin-button", "index": ALL}, "n_clicks"),
    Input({"type": "plugin_io_upload-data", "index": ALL}, "contents"),
    Input({"type": "plugin_io_btn", "index": ALL}, "n_clicks"),
    Input({"type": "plugin_io_value", "index": ALL}, "value"),
    Input({"type": "plugin_io_upload-data", "index": ALL}, "contents"),
]


def init_globals(solver):
    """
    Initialize global application state.

    Args:
        solver: SemiAutomaticReconstructionToolkit instance
    """
    global chunk_tag, column_tag, child, force_load_plugins, all_plugins_childs, canvas_list
    state = get_app_state()
    plugin_manager = state.get_plugin_manager()

    # Reset UI state
    child = []
    force_load_plugins = []
    all_plugins_childs = []
    canvas_list = []

    chunk_tag = [0 for _ in range(len(solver.decoder.GEPP.b))]
    column_tag = [0 for _ in range(solver.decoder.GEPP.b.shape[1])]

    # Get file type for plugin compatibility checking
    file_type = solver.predict_file_type()
    logger.info(f"Initializing plugins for file type: {file_type}")

    for plugin in plugin_manager.get_plugins():
        plugin_instance: FileSpecificRepair = plugin(solver, chunk_tag=get_chunk_tag())
        plugin_manager.plugin_instances.append(plugin_instance)
        plugin_instance.get_ui_elements()
        force_load_plugins.append(
            html.Button(
                plugin_instance.__class__.__name__,
                id={"type": "forceload-plugin-button", "index": plugin_instance.__class__.__name__},
            )
        )

        # Check if plugin is compatible with the file type
        try:
            compatible = plugin_instance.is_compatible(file_type)
            logger.debug(
                f"Plugin {plugin_instance.__class__.__name__} compatible with {file_type}: {compatible}"
            )

            if compatible:
                plugin_childs = plugin_manager.load_plugin(plugin_instance)
                if len(plugin_childs) > 0:
                    div = html.Div(
                        id="plugin_" + plugin_instance.__class__.__name__.lower(),
                        className="box",
                        children=plugin_childs,
                    )
                    all_plugins_childs.append(div)
                    logger.info(f"Auto-loaded plugin: {plugin_instance.__class__.__name__}")
        except Exception as e:
            logger.warning(
                f"Error checking compatibility for {plugin_instance.__class__.__name__}: {e}"
            )

    child.append(calculate_column_correctness_view())
    for i, x in enumerate(
        solver.view_file_with_chunkborders(
            False, False, LAST_CHUNK_LEN_FORMAT, checksum_len_format=state.checksum_len_format
        )
    ):
        child.append(
            html.Div(
                [html.H4(f"{str(i).zfill(8)}", id={"type": "e_row_h", "index": i}), x],
                id={"type": "e_row", "index": i},
                className="entry_row",
            )
        )

    app.layout = gen_app_layout(
        solver, get_chunk_tag(), force_load_plugins, all_plugins_childs, state.show_canvas, child
    )


def get_column_tag():
    """Get current column tags from application state."""
    return get_app_state().get_column_tag()


def update_column_tag(tag):
    """
    Update column tags in application state.

    Args:
        tag: List of column tag values
    """
    get_app_state().update_column_tag(tag)


def reset_column_tag():
    """Reset all column tags to zero."""
    update_column_tag([0 for _ in range(len(get_column_tag()))])


def get_chunk_tag():
    """Get current chunk tags from application state."""
    return get_app_state().get_chunk_tag()


def update_chunk_tag(tag):
    """
    Update chunk tags in application state.

    Args:
        tag: List of chunk tag values
    """
    get_app_state().update_chunk_tag(tag)


def update_single_element_chunk_tag(key, value):
    """
    Update a single chunk tag element.

    Args:
        key: Index of chunk to update
        value: New tag value for the chunk
    """
    tag = get_chunk_tag()
    tag[key] = value
    update_chunk_tag(tag)


def reset_chunk_tag():
    """Reset all chunk tags to zero."""
    update_chunk_tag([0 for _ in range(len(get_chunk_tag()))])


@app.callback(
    Output({"type": "plugin_io_download-data", "index": MATCH}, "data"),
    Input({"type": "plugin_io_download", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def download_data(n_clicks):
    """
    Handle plugin data download request.

    Args:
        n_clicks: Number of button clicks (trigger)

    Returns:
        Dash download component with plugin data bytes
    """
    state = get_app_state()
    plugin_manager = state.get_plugin_manager()

    c_ctx = dash.callback_context
    if not isinstance(c_ctx.triggered_id, str) and c_ctx.triggered_id["type"].startswith(
        "plugin_io"
    ):
        trigger_id = c_ctx.triggered_id["index"]
        for _plugin in plugin_manager.plugin_instances:
            if not _plugin.active:
                continue
            ui: typing.Dict[
                str, typing.Dict[str, typing.Union[str, bool, typing.Callable]]
            ] = _plugin.get_ui_elements()
            for key, value in ui.items():
                if trigger_id == key:
                    res = value["callback"](c_ctx=c_ctx)
                    download_dat = None
                    filename = "data"
                    for k, res_value in res.items():
                        if k == "download":
                            download_dat = res_value
                        elif k == "filename":
                            filename = res_value
                    if download_dat is not None:
                        return dcc.send_bytes(download_dat, filename)


def create_notification(text, color):
    """
    Create a notification UI element.

    Args:
        text: Notification message text
        color: Color style for the notification

    Returns:
        HTML div component with notification styling
    """
    return html.Div(
        [
            html.Button(id="close-notify-btn", className="delete"),
            html.Strong(text, style={"color": color}),
        ],
        className="notification is-primary",
    )


def calculate_column_correctness_view():
    """
    Generate column correctness visualization view.

    Creates colored div elements for each column based on correctness values.
    Uses red color intensity to indicate correctness level.

    Returns:
        HTML div containing column indicators
    """
    tag = get_column_tag()
    res = []
    multiplicator = 5 if 1.0 * max(tag) > 5 * np.mean(tag) else 1
    for _i, val in enumerate(tag):
        # calculate color according to value & add to res:
        val = min(val, 255)
        use_val = min(multiplicator * val, 255)
        res.append(
            html.Div(
                className="colum-div",
                id=f"column-div-{_i}",
                children=f"{str(val).zfill(2)}",
                style={"background-color": f"#FF0000{hex(use_val).replace('0x', '').zfill(2)}"},
            )
        )
        res.append(" ")
    res.append(" |")
    for _i, val in enumerate(tag):
        val = min(val, 255)
        use_val = min(multiplicator * val, 255)
        res.append(
            html.Div(
                className="colum-div",
                id=f"column-div-right-{_i}",
                children="+",
                style={"background-color": f"#FF0000{hex(use_val).replace('0x', '').zfill(2)}"},
            )
        )
    res.append("|")
    return html.Div(
        [
            html.H4(f"{''.join(['-'] * 8)}", id="column_indicator_h"),
            html.Div(res, style={"display": "inline-block"}),
        ],
        id="column_indicator",
        className="column_entry_row",
        style={"margin-bottom": "10px"},
    )


def propagate_chunk_tag_update():
    """Propagate chunk tag updates to all compatible plugins."""
    state = get_app_state()
    solver = state.get_solver()
    plugin_manager = state.get_plugin_manager()
    for _plugin in plugin_manager.plugin_instances:
        if _plugin.is_compatible(solver.predict_file_type()):
            _plugin.update_chunk_tag(get_chunk_tag())


def propagate_gepp_update():
    """
    Propagate GEPP updates to all plugins.

    Invalidates old chunk tags and propagates new GEPP to all active plugins.
    """
    state = get_app_state()
    solver = state.get_solver()
    plugin_manager = state.get_plugin_manager()
    # invalidate old chunkTags and propagate new GEPP to all plugins
    reset_chunk_tag()
    state.mark_content_updated()
    for _plugin in plugin_manager.plugin_instances:
        if not _plugin.active:
            continue
        if _plugin.is_compatible(solver.predict_file_type()):
            res = _plugin.update_gepp(solver.decoder.GEPP)
            if res is not None and "chunk_tag" in res:
                update_chunk_tag(res["chunk_tag"])
                propagate_chunk_tag_update()


def repair_chunks(repair_id, hex_value):
    """
    Repair a specific chunk with provided hex data.

    Args:
        repair_id: ID of chunk to repair
        hex_value: Hex string data to use for repair

    Returns:
        Tuple of (notification, canvas_update, recalculate_view_result)
    """
    state = get_app_state()
    solver = state.get_solver()

    if sum(state.common_packets) != 1 and not solver.multi_error_packets_mode:
        return html.Div("More than one packet still possible!"), dash.no_update, dash.no_update
    # use only the common packets that influence the selected chunk
    # an additional problem seems to be that we can tag chunks as invalid, others as valid and they yield to a deadlock
    # e.g. packet 1 was used for chunk 1 and 2, packet 2 was used for chunk 3 and 4 and packet 3 was user for chunk 5 and 4
    # if we tag chunk 1 as valid, chunk 2 as invalid, then multi-mode will yield packet 2 and packet 3 as possible invalid packets
    # BUT if we then try to repair chunk 2, there is no packet that might have created chunk 2 without also invalidating chunk 1

    common_packets_for_chunk = np.zeros(len(state.common_packets), dtype=bool)
    for packet_id, is_in in enumerate(state.common_packets):
        if is_in and solver.decoder.GEPP.get_common_packets([repair_id])[packet_id]:
            common_packets_for_chunk[packet_id] = True
    solver.manual_repair(
        repair_id,
        np.argmax(common_packets_for_chunk),
        bytearray.fromhex(hex_value.replace(" ", "")),
    )
    propagate_gepp_update()
    return recalculate_view()


@app.callback(
    Output({"type": "e_row", "index": MATCH}, "style"),
    [
        Input({"type": "e_row", "index": MATCH}, "n_clicks"),
        Input({"index": MATCH, "type": "e_row"}, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def change_button_style(n_clicks: int, n_clicks2: int) -> typing.Dict:
    """
    Handle chunk row button style changes.

    Cycles through states: unknown -> invalid -> valid -> unknown

    Args:
        n_clicks: Click count from first input
        n_clicks2: Click count from second input

    Returns:
        Button style dictionary
    """
    clicked_line = ctx.triggered_id["index"]
    current_tag = get_chunk_tag()[clicked_line]

    if current_tag == CHUNK_TAG_UNDECODED:
        # Undecoded chunk - return yellow
        return BUTTON_STYLE_YELLOW

    # Cycle to next state
    new_tag = (current_tag + 1) % 3
    update_single_element_chunk_tag(clicked_line, new_tag)

    if new_tag == CHUNK_TAG_INVALID:
        return incorrect_button_style
    elif new_tag == CHUNK_TAG_VALID:
        return correct_button_style
    else:
        return BUTTON_STYLE_WHITE


def _bytes_to_text_display(data):
    """Convert bytes to text display string (printable chars or '.')."""
    return "".join([chr(_i) if 32 <= _i <= 127 else "." for _i in data])


def _bytes_to_hex_display(data):
    """Convert bytes to hex display string with spaces."""
    return " ".join([f"{_i:02x}" for _i in data])


def _sync_hex_txt_inputs(hex_value, txt_value):
    """
    Synchronize hex and text input values.

    Args:
        hex_value: Current hex string value
        txt_value: Current text string value

    Returns:
        Tuple of (synced_hex, synced_txt)
    """
    hex_vals = hex_value.split(" ")
    if len(txt_value) == len(hex_vals):
        # Sync text changes to hex
        for _i, val in enumerate(txt_value):
            # Skip if non-printable in hex and '.' in txt
            if not (32 <= int(hex_vals[_i], 16) <= 127) and val == ".":
                logger.warning(
                    "Non-printable character in hex-view and '.' in txt-view. Skipping..."
                )
            else:
                hex_vals[_i] = f"{ord(val):02x}"
        return " ".join(hex_vals), txt_value
    return hex_value, txt_value


def _get_repair_display_values(trigger_id, id_value, hex_value, txt_value, solver):
    """
    Get display values for repair inputs based on trigger.

    Args:
        trigger_id: ID of triggered element
        id_value: Chunk ID
        hex_value: Hex string value
        txt_value: Text string value
        solver: Solver instance

    Returns:
        Tuple of (hex_display, txt_display)
    """
    if trigger_id == "repair-button":
        res = solver.decoder.GEPP.b[id_value]
        return _bytes_to_hex_display(res), _bytes_to_text_display(res)
    elif trigger_id == "hex-repair-input":
        res_bytes = bytes.fromhex(hex_value)
        return hex_value, _bytes_to_text_display(res_bytes)
    elif trigger_id == "txt-repair-input":
        return _sync_hex_txt_inputs(hex_value, txt_value)
    return "", ""


def repair_callback(trigger_id, input_value, id_value, hex_value, txt_value):
    """
    Handle repair input callbacks for hex and text repair fields.

    Synchronizes hex and text repair inputs, validating and converting between formats.

    Args:
        trigger_id: ID of the element that triggered the callback
        input_value: Current input value
        id_value: Chunk ID to repair
        hex_value: Hex string representation of data
        txt_value: Text string representation of data

    Returns:
        Tuple of (notification, hidden_flag, disabled_flag, hex_value, hex_style, txt_value, txt_style)
    """
    state = get_app_state()
    solver = state.get_solver()

    if hex_value is None:
        hex_value = ""
    if txt_value is None:
        txt_value = ""

    # Check if repair is allowed
    if trigger_id == "repair-button" and (
        id_value is None
        or get_chunk_tag()[id_value] == 2
        or get_chunk_tag()[id_value] == 0
        or (sum(state.common_packets) > 1 and not solver.multi_error_packets_mode)
    ):
        return (
            html.Div(
                "Repair only possible for rows tagged as invalid. Additionally, a single corrupt packet should be identified."
            ),
            True,
            False,
            "",
            {},
            "",
            {},
        )

    # Validate and get display values
    if all(c in string.hexdigits + " " for c in "" + hex_value):
        res_hex, res_str = _get_repair_display_values(
            trigger_id, id_value, hex_value, txt_value, solver
        )
    else:
        res_hex, res_str = "", ""

    hex_style = {"width": f"{len(res_hex) * 10}px"}
    str_style = {"width": f"{len(res_str) * 10}px"}
    return (
        html.Div(
            "Repair only possible for rows tagged as invalid. Additionally, a single corrupt packet should be identified."
        ),
        False if input_value % 2 == 1 else True,
        False if input_value % 2 == 0 else True,
        res_hex,
        hex_style,
        res_str,
        str_style,
    )


@app.callback(
    Output("analytics-output", "children"),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)
def update_analytics(n):
    """
    Update analytics display on interval.

    Args:
        n: Interval count (unused, just for triggering)

    Returns:
        File type prediction HTML or no_update
    """
    state = get_app_state()
    if state.is_content_updated():
        state.reset_content_updated()
        return html.H3(state.get_solver().predict_file_type())
    else:
        return dash.no_update


@app.callback(
    Output("analyze-count-output", "children"),
    Output("row_view", "children"),
    Output("ls-loading-output-2", "children"),
    Output("dashCanvas", "json_data"),
    Input("dashCanvas", "json_data"),
    prevent_initial_call=True,
)
def update_canvas_data(json_data):
    """
    Handle canvas data updates from plugins.

    Args:
        json_data: Canvas JSON data from Dash canvas component

    Returns:
        Tuple of recalculate_view results and new JSON data
    """
    state = get_app_state()
    plugin_manager = state.get_plugin_manager()

    updates_b = False
    new_json_data = dash.no_update
    if json_data is None:
        return dash.no_update
    for _plugin in plugin_manager.plugin_instances:
        if not _plugin.active:
            continue
        res = _plugin.update_canvas(json_data)
        if res is not None:
            if "updates_b" in res:
                updates_b = res["updates_b"]
            if "chunk_tag" in res:
                update_chunk_tag(res["chunk_tag"])
        if updates_b:
            propagate_gepp_update()
    return recalculate_view() + (new_json_data,)


def init_callback_handlers():
    """Initialize callback handler instances after globals are set up."""
    global plugin_handler, button_handler, repair_handler
    state = get_app_state()

    plugin_handler = PluginCallbackHandler(
        state,
        state.get_plugin_manager(),
        get_chunk_tag,
        update_chunk_tag,
        update_column_tag,
        recalculate_view,
        propagate_gepp_update,
    )

    button_handler = ButtonCallbackHandler(
        state,
        recalculate_view,
        propagate_gepp_update,
        get_chunk_tag,
        update_chunk_tag,
        reset_chunk_tag,
        reset_column_tag,
        propagate_chunk_tag_update,
        repair_chunks,
    )

    repair_handler = RepairCallbackHandler(repair_callback, repair_chunks)


@app.callback(
    Output("analytics-input", "children"),
    Output("repair-input", "hidden"),
    Output("repair-id-input-box", "disabled"),
    Output("hex-repair-input", "value"),
    Output("hex-repair-input", "style"),
    Output("txt-repair-input", "value"),
    Output("txt-repair-input", "style"),
    # Output for recalculate_view:
    Output("analyze-count-output", "children"),
    Output("row_view", "children"),
    Output("ls-loading-output-2", "children"),
    # Output for plugins:
    Output("plugin_view", "children"),
    # Canvas style:
    Output("canvas", "style"),
    # Canvas data:
    Output("dashCanvas", "image_content"),
    Output("kaitai_view", "children"),
    State("packet-tag-chunk-input", "value"),
    State("dashCanvas", "json_data"),
    State("dashCanvas", "image_content"),
    State("dashCanvas", "width"),
    State("dashCanvas", "height"),
    # Repair-View Inputs:
    input_callback_handler,
    prevent_initial_call=True,
)
def callback_handler(*args, **kwargs):
    """Refactored callback handler that delegates to specialized handler classes."""
    c_ctx = dash.callback_context
    trigger_id = c_ctx.triggered[0]["prop_id"].split(".")[0]
    packet_tag_chunk_input = c_ctx.states.get("packet-tag-chunk-input.value")

    # Handle plugin I/O callbacks
    if not isinstance(c_ctx.triggered_id, str) and c_ctx.triggered_id["type"].startswith(
        "plugin_io"
    ):
        return plugin_handler.handle_plugin_io(c_ctx.triggered_id["index"], c_ctx, *args, **kwargs)

    # Handle repair-related callbacks
    if trigger_id in ["repair-button", "hex-repair-input", "txt-repair-input"]:
        return repair_handler.handle_repair_inputs(trigger_id, c_ctx)
    elif trigger_id == "repair-chunks-button":
        return repair_handler.handle_repair_chunks_button(c_ctx)

    # Handle button callbacks
    elif trigger_id == "analyze-button":
        return button_handler.handle_analyze_button()
    elif trigger_id == "repair-exclusion-button":
        return button_handler.handle_repair_exclusion_button()
    elif trigger_id == "calculate-rank-button":
        return button_handler.handle_calculate_rank_button()
    elif trigger_id == "reset-chunk-tag-button":
        return button_handler.handle_reset_chunk_tag_button()
    elif trigger_id == "save-button":
        return button_handler.handle_save_button()
    elif trigger_id in ["packet-tag-chunk-invalid-button", "packet-tag-chunk-valid-button"]:
        return button_handler.handle_packet_tag_buttons(trigger_id, packet_tag_chunk_input)
    elif trigger_id == "mode-switch":
        return button_handler.handle_mode_switch(c_ctx.inputs.get("mode-switch.value"))
    elif trigger_id == "colorblind-switch":
        return button_handler.handle_colorblind_switch(c_ctx.triggered[0]["value"])
    elif trigger_id in ["repair-reorder-button", "repair-reorder-button-possible"]:
        return button_handler.handle_repair_reorder_buttons(trigger_id)
    elif (
        c_ctx.triggered_id is not None
        and not isinstance(c_ctx.triggered_id, str)
        and c_ctx.triggered_id["type"] == "forceload-plugin-button"
    ):
        return button_handler.handle_forceload_plugin_button(c_ctx)

    # Default case
    return dash.no_update


def fast_most_common_matrix(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the most common value at each position across multiple matrices.

    Takes a 3D array of matrices and returns the matrix of the most common value
    at each position (i,j), along with a boolean matrix indicating positions
    where only a single unique value exists.

    Args:
        matrices: 3D numpy array of matrices with shape (num_matrices, rows, columns)

    Returns:
        Tuple of (output_matrix, has_single_val) where:
            - output_matrix: Matrix of most common values at each position
            - has_single_val: Boolean matrix indicating positions with single unique value
    """
    # takes a 3d array of matrices and returns the matrix of the most common value of each position (i,j,_)
    # get the dimensions of the first matrix in the list
    num_rows = matrices.shape[0]
    num_columns = matrices.shape[1]

    # initialize an empty matrix to hold the output
    output_matrix = np.zeros((num_rows, num_columns), dtype=matrices[0].dtype)
    has_single_val = np.zeros((num_rows, num_columns), dtype=bool)

    # loop through all the positions [i,j] in the output matrix
    for _i in range(num_rows):
        for _j in range(num_columns):
            b_count = np.bincount(matrices[_i, _j, :])
            most_common_element = b_count.argmax()
            output_matrix[_i, _j] = most_common_element
            has_single_val[_i, _j] = len(np.nonzero(b_count)[0]) == 1
    return output_matrix, has_single_val


def recalculate_view():
    """
    Recalculate and regenerate the file view based on current state.

    Analyzes chunk tags, identifies invalid/valid rows, calculates common packets,
    and generates the HTML view with appropriate styling for each row.

    Returns:
        Tuple containing view components and state information for Dash callback
    """
    state = get_app_state()
    solver = state.get_solver()

    child_view = []
    invalid_rows = [_i for _i, _x in enumerate(get_chunk_tag()) if _x == 1]
    valid_rows = [_i for _i, _x in enumerate(get_chunk_tag()) if _x == 2]
    state.common_packets = solver.decoder.GEPP.get_common_packets(
        invalid_rows, valid_rows, solver.multi_error_packets_mode
    )  # [:semi_automatic_solver.decoder.GEPP.m]
    not_used_packets = solver.calculate_unused_packets()
    state.common_packets = [
        (not not_used_packets[_i]) and state.common_packets[_i]
        for _i, _x in enumerate(state.common_packets)
    ]
    unused_packet_ids = [_i for _i, j in enumerate(not_used_packets) if j]
    logger.info(f"The following packets were not used for the reconstruction: {unused_packet_ids}")
    common_packets_str = " ".join("1" if x else "0" for x in state.common_packets)
    logger.info(f"Potentially invalid packets: {common_packets_str}")
    rem_possible_chunks = solver.get_possible_invalid_chunks_from_common_packets(
        state.common_packets
    )
    # add an indicator for the column correctness:
    child_view.append(calculate_column_correctness_view())
    for _i, _x in enumerate(
        solver.view_file_with_chunkborders(
            False, False, LAST_CHUNK_LEN_FORMAT, checksum_len_format=state.checksum_len_format
        )
    ):
        if _i in invalid_rows:
            child_view.append(
                html.Div(
                    [html.H4(f"{str(_i).zfill(8)}", id={"type": "e_row_h", "index": _i}), _x],
                    id={"type": "e_row", "index": _i},
                    className="entry_row",
                    style=incorrect_button_style,
                )
            )
        elif get_chunk_tag()[_i] == CHUNK_TAG_UNDECODED:
            child_view.append(
                html.Div(
                    [html.H4(f"{str(_i).zfill(8)}", id={"type": "e_row_h", "index": _i}), _x],
                    id={"type": "e_row", "index": _i},
                    className="entry_row",
                    style=BUTTON_STYLE_YELLOW,
                )
            )
        elif _i in valid_rows:
            child_view.append(
                html.Div(
                    [html.H4(f"{str(_i).zfill(8)}", id={"type": "e_row_h", "index": _i}), _x],
                    id={"type": "e_row", "index": _i},
                    className="entry_row",
                    style=correct_button_style,
                )
            )
        elif rem_possible_chunks[_i]:
            child_view.append(
                html.Div(
                    [html.H4(f"{str(_i).zfill(8)}", id={"type": "e_row_h", "index": _i}), _x],
                    id={"type": "e_row", "index": _i},
                    className="entry_row",
                    style=BUTTON_STYLE_LIGHT_RED,
                )
            )
        else:
            child_view.append(
                html.Div(
                    [html.H4(f"{str(_i).zfill(8)}", id={"type": "e_row_h", "index": _i}), _x],
                    id={"type": "e_row", "index": _i},
                    className="entry_row",
                )
            )
    poss_packet_str = f"Possible invalid packets: {sum(state.common_packets)} | {','.join(['(#' + str(i) + ' / output:' + str(solver.decoder.GEPP.packet_mapping[i]) + '), ' for i, x in enumerate(state.common_packets) if x])}"
    return html.Div(poss_packet_str), html.Div(child_view), html.Div("")


def _main_entry():
    """Main entry point - initializes application state and starts the server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("ini", metavar="ini", type=str, help="config file (ini)")
    parsed_args = parser.parse_args()
    ini_file = parsed_args.ini

    cfg_worker = ConfigReadAndExecute(ini_file)
    x = cfg_worker.execute(return_decoder=True, skip_solve=True)[0]
    solver = SemiAutomaticReconstructionToolkit(x)
    solver.decoder.solve(partial=True)
    checksum_len_format = cfg_worker.config[cfg_worker.config.sections()[0]].get(
        "checksum_len_str", None
    )
    if checksum_len_format == "":
        checksum_len_format = None

    # Initialize the global application state
    initialize_app_state(solver, checksum_len_format)

    init_globals(solver)
    callbacks(app)
    init_callback_handlers()  # Initialize the handler instances
    app.run(threaded=True, host="0.0.0.0")


if __name__ == "__main__":
    _main_entry()
