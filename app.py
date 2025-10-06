# -*- coding: utf-8 -*-
# Run this app using: `python app.py <file.ini>` and
# visit http://127.0.0.1:8050/ in your web browser.
import argparse
import string
import typing

import numpy as np
import dash_extensions.enrich as dash
from dash_extensions.enrich import html, dcc, State, MATCH, ALL, Output, DashProxy, Input, MultiplexerTransform
from dash import ctx

from app_callbacks import callbacks
from callback_handlers import PluginCallbackHandler, ButtonCallbackHandler, RepairCallbackHandler
from layout import gen_app_layout
from repair_algorithms import *  # NOSONAR
from repair_algorithms.PluginManager import PluginManager
from repair_algorithms.FileSpecificRepair import FileSpecificRepair

from NOREC4DNA.ConfigWorker import ConfigReadAndExecute
from semi_automatic_reconstruction_toolkit import SemiAutomaticReconstructionToolkit

# Global callback handler instances (will be initialized after globals are set up)
plugin_handler: typing.Optional[PluginCallbackHandler] = None
button_handler: typing.Optional[ButtonCallbackHandler] = None
repair_handler: typing.Optional[RepairCallbackHandler] = None


def update_point(trace, points, selector):
    print(trace, points, selector)
    return points.point_inds


LAST_CHUNK_LEN_FORMAT = "I"
EXTERNAL_STYLESHEETS = ["https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css"]
META_TAGS = [{"name": "viewport", "content": "width=device-width, initial-scale=1"}]

white_button_style = {'backgroundColor': 'white'}
red_button_style = {'backgroundColor': 'red'}
green_button_style = {'backgroundColor': 'green'}
light_red_button_style = {'backgroundColor': 'lightcoral'}
yellow_button_style = {'backgroundColor': 'yellow'}

correct_button_style = green_button_style
incorrect_button_style = red_button_style

colorblind_correct = {'backgroundColor': '#84CE73'}
colorblind_incorrect = {'backgroundColor': 'brown'}

common_packets = []
child = []
content_updated = False
show_canvas = False
# 0 = no information about the correctness of the chunk
# 1 = chunk is correct
# 2 = chunk is incorrect
chunk_tag = []
column_tag = []

canvas_list = []
plugin_manager = PluginManager()
plugin_manager.plugin_instances.clear()
force_load_plugins = []
all_plugins_childs = []

app = DashProxy(__name__, external_stylesheets=EXTERNAL_STYLESHEETS, meta_tags=META_TAGS,
                prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

input_callback_handler = [Input('repair-button', 'n_clicks'),
                          Input('repair-reorder-button-possible', 'n_clicks'),
                          Input('repair-id-input-box', 'value'),
                          Input('hex-repair-input', 'value'),
                          Input('txt-repair-input', 'value'),
                          Input('repair-chunks-button', 'n_clicks'),
                          Input('analyze-button', 'n_clicks'),
                          Input('repair-exclusion-button', 'n_clicks'),
                          Input('repair-reorder-button', 'n_clicks'),
                          Input('reset-chunk-tag-button', 'n_clicks'),
                          Input('calculate-rank-button', 'n_clicks'),
                          Input('save-button', 'n_clicks'),
                          Input('packet-tag-chunk-invalid-button', 'n_clicks'),
                          Input('packet-tag-chunk-valid-button', 'n_clicks'),
                          Input('mode-switch', 'value'),
                          Input('colorblind-switch', 'value'),
                          Input({'type': 'forceload-plugin-button', 'index': ALL}, 'n_clicks'),
                          Input({'type': 'plugin_io_upload-data', 'index': ALL}, 'contents'),
                          Input({'type': 'plugin_io_btn', 'index': ALL}, 'n_clicks'),
                          Input({'type': 'plugin_io_value', 'index': ALL}, 'value'),
                          Input({'type': 'plugin_io_upload-data', "index": ALL}, 'contents')]


def init_globals(semi_automatic_solver):
    global chunk_tag, column_tag
    chunk_tag = [0 for _ in range(len(semi_automatic_solver.decoder.GEPP.b))]
    column_tag = [0 for _ in range(semi_automatic_solver.decoder.GEPP.b.shape[1])]
    for plugin in plugin_manager.get_plugins():
        plugin_instance: FileSpecificRepair = plugin(semi_automatic_solver,
                                                     chunk_tag=get_chunk_tag())
        plugin_manager.plugin_instances.append(plugin_instance)
        plugin_instance.get_ui_elements()
        force_load_plugins.append(html.Button(plugin_instance.__class__.__name__, id={"type": "forceload-plugin-button",
                                                                                      "index": plugin_instance.__class__.__name__}))
        if plugin_instance.is_compatible(semi_automatic_solver.predict_file_type()):
            plugin_childs = plugin_manager.load_plugin(plugin_instance)
            if len(plugin_childs) > 0:
                div = html.Div(id="plugin_" + plugin_instance.__class__.__name__.lower(), className="box",
                               children=plugin_childs)
                all_plugins_childs.append(div)
    child.append(calculate_column_correctness_view())
    for i, x in enumerate(semi_automatic_solver.view_file_with_chunkborders(False, False, LAST_CHUNK_LEN_FORMAT,
                                                                            checksum_len_format=CHECKSUM_LEN_FORMAT)):
        child.append(html.Div([html.H4(f"{str(i).zfill(8)}", id={'type': 'e_row_h', 'index': i}), x],
                              id={'type': 'e_row', 'index': i}, className="entry_row"))

    app.layout = gen_app_layout(semi_automatic_solver, get_chunk_tag(), force_load_plugins, all_plugins_childs,
                                show_canvas, child)


def get_column_tag():
    global column_tag
    return column_tag


def update_column_tag(tag):
    global column_tag
    column_tag = tag


def reset_column_tag():
    update_column_tag([0 for _ in range(len(get_column_tag()))])


def get_chunk_tag():
    global chunk_tag
    return chunk_tag


def update_chunk_tag(tag):
    global chunk_tag
    chunk_tag = tag


def update_single_element_chunk_tag(key, value):
    global chunk_tag
    chunk_tag[key] = value


def reset_chunk_tag():
    update_chunk_tag([0 for _ in range(len(get_chunk_tag()))])


@app.callback(Output({'type': 'plugin_io_download-data', 'index': MATCH}, "data"),
              Input({'type': 'plugin_io_download', 'index': MATCH}, "n_clicks"),
              prevent_initial_call=True, )
def download_data(n_clicks):
    c_ctx = dash.callback_context
    if not isinstance(c_ctx.triggered_id, str) and c_ctx.triggered_id["type"].startswith("plugin_io"):
        trigger_id = c_ctx.triggered_id["index"]
        for _plugin in plugin_manager.plugin_instances:
            if not _plugin.active:
                continue
            ui: typing.Dict[str, typing.Dict[str, typing.Union[str, bool, typing.Callable]]] = _plugin.get_ui_elements()
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
    return html.Div(
        [html.Button(id="close-notify-btn", className="delete"),
         html.Strong(text, style={"color": color})], className="notification is-primary")


def calculate_column_correctness_view():
    tag = get_column_tag()
    res = []
    multiplicator = 5 if 1.0 * max(tag) > 5 * np.mean(tag) else 1
    for _i, val in enumerate(tag):
        # calculate color according to value & add to res:
        val = min(val, 255)
        use_val = min(multiplicator * val, 255)
        res.append(html.Div(className="colum-div", id=f"column-div-{_i}", children=f"{str(val).zfill(2)}",
                            style={"background-color": f"#FF0000{hex(use_val).replace('0x', '').zfill(2)}"}))
        res.append(" ")
    res.append(" |")
    for _i, val in enumerate(tag):
        val = min(val, 255)
        use_val = min(multiplicator * val, 255)
        res.append(html.Div(className="colum-div", id=f"column-div-right-{_i}", children="+",
                            style={"background-color": f"#FF0000{hex(use_val).replace('0x', '').zfill(2)}"}))
    res.append("|")
    return html.Div([html.H4(f"{''.join(['-'] * 8)}", id="column_indicator_h"),
                     html.Div(res, style={"display": "inline-block"})],
                    id="column_indicator", className="column_entry_row", style={"margin-bottom": "10px"})


def propagete_chunk_tag_update():
    for _plugin in plugin_manager.plugin_instances:
        if _plugin.is_compatible(semi_automatic_solver.predict_file_type()):
            _plugin.update_chunk_tag(get_chunk_tag())


def propagate_gepp_update():
    global content_updated
    # invalidate old chunkTags and propagate new GEPP to all plugins
    reset_chunk_tag()
    content_updated = True
    for _plugin in plugin_manager.plugin_instances:
        if not _plugin.active:
            continue
        if _plugin.is_compatible(semi_automatic_solver.predict_file_type()):
            res = _plugin.update_gepp(semi_automatic_solver.decoder.GEPP)
            if res is not None and "chunk_tag" in res:
                update_chunk_tag(res["chunk_tag"])
                propagete_chunk_tag_update()


def repair_chunks(repair_id, hex_value):
    if sum(common_packets) != 1 and not semi_automatic_solver.multi_error_packets_mode:
        return html.Div("More than one packet still possible!"), dash.no_update, dash.no_update
    # use only the common packets that influence the selected chunk
    # an additional problem seems to be that we can tag chunks as invalid, others as valid and they yield to a deadlock
    # e.g. packet 1 was used for chunk 1 and 2, packet 2 was used for chunk 3 and 4 and packet 3 was user for chunk 5 and 4
    # if we tag chunk 1 as valid, chunk 2 as invalid, then multi-mode will yield packet 2 and packet 3 as possible invalid packets
    # BUT if we then try to repair chunk 2, there is no packet that might have created chunk 2 without also invalidating chunk 1

    common_packets_for_chunk = np.zeros(len(common_packets), dtype=bool)
    for packet_id, is_in in enumerate(common_packets):
        if is_in and semi_automatic_solver.decoder.GEPP.get_common_packets([repair_id])[packet_id]:
            common_packets_for_chunk[packet_id] = True
    semi_automatic_solver.manual_repair(repair_id, np.argmax(common_packets_for_chunk),
                                        bytearray.fromhex(hex_value.replace(" ", "")))
    propagate_gepp_update()
    return recalculate_view()


@app.callback(
    Output({'type': 'e_row', 'index': MATCH}, 'style'),
    [Input({'type': 'e_row', 'index': MATCH}, 'n_clicks'),
     Input({'index': MATCH, 'type': 'e_row'}, 'n_clicks')],
    prevent_initial_call=True)
def change_button_style(n_clicks, n_clicks2):
    clicked_line = ctx.triggered_id["index"]
    if get_chunk_tag()[clicked_line] == 3:
        # the selected chunk is not decoded yet, return yellow and don't update the state.
        return yellow_button_style
    update_single_element_chunk_tag(clicked_line, (get_chunk_tag()[clicked_line] + 1) % 3)
    if get_chunk_tag()[clicked_line] == 1:
        return incorrect_button_style
    elif get_chunk_tag()[clicked_line] == 2:
        return correct_button_style
    else:
        return white_button_style


def repair_callback(trigger_id, input_value, id_value, hex_value, txt_value):
    if hex_value is None:
        hex_value = ""
    if txt_value is None:
        txt_value = ""
    # in this case, all we have to do is propagate to all packets that are still reachable from chunk "id_value"
    # and then recalculate the view
    if trigger_id == "repair-button" and (id_value is None or get_chunk_tag()[id_value] == 2 or get_chunk_tag()[
        id_value] == 0 or (sum(common_packets) > 1 and not semi_automatic_solver.multi_error_packets_mode)):
        return html.Div(
            "Repair only possible for rows tagged as invalid. Additionally, a single corrupt packet should be identified."), True, False, "", {}, "", {}
    # set width to fit content:

    # make sure only HEX and space in hex_value
    if all(c in string.hexdigits + " " for c in "" + hex_value):
        if trigger_id == "repair-button":
            # fill in hex-repair-input and txt-repair-input
            res = semi_automatic_solver.decoder.GEPP.b[id_value]
            res_str = "".join([chr(_i) if 32 <= _i <= 127 else "." for _i in res])
            res_hex = " ".join([f"{_i:02x}" for _i in res])
        elif trigger_id == "hex-repair-input":
            # fill in txt-repair-input
            # convert hex to bytes and from bytes to string:
            res_bytes = bytes.fromhex(hex_value)
            res_hex = hex_value  # keep current value...
            res_str = "".join([chr(_i) if 32 <= _i <= 127 else "." for _i in res_bytes])
        elif trigger_id == "txt-repair-input":
            # fill in hex-repair-input
            # if len(hex) !=  len(txt), wait until both are equal again
            hex_vals = hex_value.split(" ")
            if len(txt_value) == len(hex_vals):
                # care!: we should ONLY change bytes in hex-view if the corresponding byte in txt-view is changed (is a printable character!)
                # problem: we use "." for non-printable chars in txt-view. Thus, we have to check
                # if the corresponding byte in hex-view is a printable "." or if the byte is a non-printable byte
                # iterate over all charaters in
                for _i, val in enumerate(txt_value):
                    # PROBLEM: if the user intentionally changes a byte to "." from a non-printable character,
                    # we do not propagate this change...
                    # if we have a non-printable character in hex-view and "." in txt view, skip it...
                    if not (32 <= int(hex_vals[_i], 16) <= 127) and val == ".":
                        print("Warning: non printable character in hex-view and '.' in txt-view. Skipping...")
                    else:
                        hex_vals[_i] = f"{ord(val):02x}"
                res_hex = " ".join(hex_vals)
                res_str = txt_value
            else:
                res_hex = hex_value
                res_str = txt_value
        else:
            res_hex = ""
            res_str = ""
    else:
        res_hex = ""
        res_str = ""
    hex_style = {'width': f'{len(res_hex) * 10}px'}
    str_style = {'width': f'{len(res_str) * 10}px'}
    return html.Div(
        "Repair only possible for rows tagged as invalid. Additionally, a single corrupt packet should be identified."), False if input_value % 2 == 1 else True, False if input_value % 2 == 0 else True, \
        res_hex, hex_style, res_str, str_style


@app.callback(Output("analytics-output", "children"), Input('interval-component', 'n_intervals'),
              prevent_initial_call=True)
def update_analytics(n):
    global content_updated
    if content_updated:
        content_updated = False
        return html.H3(semi_automatic_solver.predict_file_type())
    else:
        return dash.no_update


@app.callback(Output('analyze-count-output', 'children'),
              Output('row_view', 'children'),
              Output("ls-loading-output-2", "children"),
              Output("dashCanvas", "json_data"),
              Input('dashCanvas', 'json_data'))
def update_canvas_data(json_data):
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

    plugin_handler = PluginCallbackHandler(
        plugin_manager, get_chunk_tag, update_chunk_tag, update_column_tag,
        recalculate_view, propagate_gepp_update
    )

    button_handler = ButtonCallbackHandler(
        semi_automatic_solver, recalculate_view, propagate_gepp_update,
        get_chunk_tag, update_chunk_tag, reset_chunk_tag, reset_column_tag,
        propagete_chunk_tag_update, repair_chunks
    )

    repair_handler = RepairCallbackHandler(repair_callback, repair_chunks)


@app.callback(
    Output("analytics-input", "children"),
    Output("repair-input", "hidden"),
    Output("repair-id-input-box", "disabled"),
    Output('hex-repair-input', 'value'),
    Output('hex-repair-input', 'style'),
    Output('txt-repair-input', 'value'),
    Output('txt-repair-input', 'style'),
    # Output for recalculate_view:
    Output('analyze-count-output', 'children'),
    Output('row_view', 'children'),
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
    prevent_initial_call=True
)
def callback_handler(*args, **kwargs):
    """Refactored callback handler that delegates to specialized handler classes."""
    c_ctx = dash.callback_context
    trigger_id = c_ctx.triggered[0]["prop_id"].split(".")[0]
    packet_tag_chunk_input = c_ctx.states.get('packet-tag-chunk-input.value')

    # Handle plugin I/O callbacks
    if not isinstance(c_ctx.triggered_id, str) and c_ctx.triggered_id["type"].startswith("plugin_io"):
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
        return button_handler.handle_mode_switch(c_ctx.inputs.get('mode-switch.value'))
    elif trigger_id == "colorblind-switch":
        return button_handler.handle_colorblind_switch(c_ctx.triggered[0]["value"])
    elif trigger_id in ["repair-reorder-button", "repair-reorder-button-possible"]:
        return button_handler.handle_repair_reorder_buttons(trigger_id)
    elif (c_ctx.triggered_id is not None and not isinstance(c_ctx.triggered_id, str) and
          c_ctx.triggered_id["type"] == "forceload-plugin-button"):
        return button_handler.handle_forceload_plugin_button(c_ctx)

    # Default case
    return dash.no_update


def fast_most_common_matrix(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    global common_packets
    child_view = []
    invalid_rows = [_i for _i, _x in enumerate(get_chunk_tag()) if
                    _x == 1]
    valid_rows = [_i for _i, _x in enumerate(get_chunk_tag()) if _x == 2]
    common_packets = semi_automatic_solver.decoder.GEPP.get_common_packets(invalid_rows,
                                                                           valid_rows,
                                                                           semi_automatic_solver.multi_error_packets_mode)  # [:semi_automatic_solver.decoder.GEPP.m]
    not_used_packets = semi_automatic_solver.calculate_unused_packets()
    common_packets = [(not not_used_packets[_i]) and common_packets[_i] for _i, _x in enumerate(common_packets)]
    print(
        f"The following packets were not used for the reconstruction: {[_i for _i, j in enumerate(not_used_packets) if j]}")
    print("potentially invalid Packets:")
    print(" ".join(map(lambda x: "1" if x else "0", common_packets)), flush=True)
    rem_possible_chunks = semi_automatic_solver.get_possible_invalid_chunks_from_common_packets(common_packets)
    # add an indicator for the column correctness:
    child_view.append(calculate_column_correctness_view())
    for _i, _x in enumerate(semi_automatic_solver.view_file_with_chunkborders(False, False, LAST_CHUNK_LEN_FORMAT,
                                                                              checksum_len_format=CHECKSUM_LEN_FORMAT)):
        if _i in invalid_rows:
            child_view.append(html.Div([html.H4(f"{str(_i).zfill(8)}", id={'type': 'e_row_h', 'index': _i}), _x],
                                       id={'type': 'e_row', 'index': _i}, className="entry_row",
                                       style=incorrect_button_style))
        elif chunk_tag[_i] == 3:
            child_view.append(html.Div([html.H4(f"{str(_i).zfill(8)}", id={'type': 'e_row_h', 'index': _i}), _x],
                                       id={'type': 'e_row', 'index': _i}, className="entry_row",
                                       style=yellow_button_style))
        elif _i in valid_rows:
            child_view.append(html.Div([html.H4(f"{str(_i).zfill(8)}", id={'type': 'e_row_h', 'index': _i}), _x],
                                       id={'type': 'e_row', 'index': _i}, className="entry_row",
                                       style=correct_button_style))
        elif rem_possible_chunks[_i]:
            child_view.append(html.Div([html.H4(f"{str(_i).zfill(8)}", id={'type': 'e_row_h', 'index': _i}), _x],
                                       id={'type': 'e_row', 'index': _i}, className="entry_row",
                                       style=light_red_button_style))
        else:
            child_view.append(html.Div([html.H4(f"{str(_i).zfill(8)}", id={'type': 'e_row_h', 'index': _i}), _x],
                                       id={'type': 'e_row', 'index': _i}, className="entry_row"))
    poss_packet_str = f"Possible invalid packets: {sum(common_packets)} | {','.join(['(#' + str(i) + ' / output:' + str(semi_automatic_solver.decoder.GEPP.packet_mapping[i]) + '), ' for i, x in enumerate(common_packets) if x])}"
    return html.Div(poss_packet_str), html.Div(child_view), html.Div("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ini", metavar="ini", type=str, help="config file (ini)")
    parsed_args = parser.parse_args()
    ini_file = parsed_args.ini

    cfg_worker = ConfigReadAndExecute(ini_file)
    x = cfg_worker.execute(return_decoder=True, skip_solve=True)[0]
    semi_automatic_solver = SemiAutomaticReconstructionToolkit(x)
    semi_automatic_solver.decoder.solve(partial=True)
    CHECKSUM_LEN_FORMAT = cfg_worker.config[cfg_worker.config.sections()[0]].get("checksum_len_str", None)
    if CHECKSUM_LEN_FORMAT == "":
        CHECKSUM_LEN_FORMAT = None
    init_globals(semi_automatic_solver)
    callbacks(app)
    init_callback_handlers()  # Initialize the handler instances
    app.run(threaded=True, host="0.0.0.0")
    """
    # to enable debugging / dev tools:
    app.run(host="0.0.0.0", dev_tools_ui=True, dev_tools_hot_reload=True, debug=True, threaded=True,
            dev_tools_hot_reload_interval=10000, dev_tools_hot_reload_watch_interval=10000)
    """
