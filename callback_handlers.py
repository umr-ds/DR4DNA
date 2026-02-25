# -*- coding: utf-8 -*-
"""
Refactored callback handlers for the DR4DNA application.
This module breaks down the complex callback_handler function into smaller, manageable pieces.
"""

import copy
import shutil
import typing
from pathlib import Path

import numpy as np
import dash_extensions.enrich as dash
from dash_extensions.enrich import html, dcc
from dash_canvas.utils.io_utils import array_to_data_url

from state import AppState, get_app_state


class CallbackResponse:
    """Helper class to manage callback response data."""

    def __init__(self):
        self.info_str = dash.no_update
        self.canvas_image_content = dash.no_update
        self.kaitai_view = dash.no_update
        self.refresh_view = False
        self.update_gepp = False

    def create_no_update_tuple(self, size=14):
        """Create a tuple of dash.no_update values."""
        return tuple([dash.no_update] * size)

    def create_standard_response(self, recalc_view_result=None):
        """Create standard response tuple."""
        if recalc_view_result:
            return (self.info_str, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, dash.no_update) + recalc_view_result + (
                       dash.no_update, dash.no_update, self.canvas_image_content, self.kaitai_view)
        return (self.info_str,) + self.create_no_update_tuple(11) + (self.canvas_image_content, self.kaitai_view)


class PluginCallbackHandler:
    """Handles plugin-related callbacks."""

    def __init__(self, state: AppState, plugin_manager, get_chunk_tag_func, update_chunk_tag_func,
                 update_column_tag_func, recalculate_view_func, propagate_gepp_update_func):
        self.state = state
        # Store plugin_manager reference for convenience, but it's also in state
        self.plugin_manager = plugin_manager
        self.get_chunk_tag = get_chunk_tag_func
        self.update_chunk_tag = update_chunk_tag_func
        self.update_column_tag = update_column_tag_func
        self.recalculate_view = recalculate_view_func
        self.propagate_gepp_update = propagate_gepp_update_func
    
    def get_plugin_manager(self):
        """Get plugin manager from state to ensure we always have the current instance."""
        return self.state.get_plugin_manager()

    def handle_plugin_io(self, trigger_id, c_ctx, *args, **kwargs):
        """Handle plugin I/O callbacks."""
        response = CallbackResponse()
        plugin_manager = self.get_plugin_manager()

        for plugin in plugin_manager.plugin_instances:
            if not plugin.active:
                continue

            ui = plugin.get_ui_elements()
            for key, value in ui.items():
                if trigger_id == key:
                    res = value["callback"](chunk_tag=self.get_chunk_tag(), c_ctx=c_ctx, *args, **kwargs)
                    special_return = self._process_plugin_response(res, response)

                    # If _process_plugin_response returned a value (e.g., from "repair" handler), use it directly
                    if special_return is not None:
                        return special_return

                    if response.refresh_view or response.update_gepp:
                        if response.update_gepp:
                            self.propagate_gepp_update()
                        return response.create_standard_response(self.recalculate_view())
                    else:
                        return response.create_standard_response()

        # No matching plugin found - return no_update for all outputs
        return response.create_standard_response()

    def _process_plugin_response(self, res, response):
        """Process plugin callback response. Returns special value for 'repair' handler, None otherwise."""
        handlers = {
            "chunk_tag": lambda v: self.update_chunk_tag(v),
            "column_tag": lambda v: self.update_column_tag(v),
            "update_b": lambda v: setattr(response, 'update_gepp', v),
            "refresh_view": lambda v: setattr(response, 'refresh_view', v),
            "image_content": lambda v: setattr(response, 'canvas_image_content', v),
            "canvas_data": lambda v: self._handle_canvas_data(v, res, response),
            "kaitai_content": lambda v: setattr(response, 'kaitai_view', res["kaitai_content"]),
            "info": lambda v: setattr(response, 'info_str', v),
            "repair_variations": lambda v: self._handle_repair_variations(v, response),
            "repair_for_each_packet": lambda v: self._handle_repair_for_each_packet(v, response),
        }

        for key, res_value in res.items():
            # Special case for "repair" - it needs to return a complete response tuple
            if key == "repair":
                return self._handle_repair(res_value, res, response)
            elif key in handlers:
                handlers[key](res_value)
            elif key not in ["updates_canvas", "height", "width", "generate_all", "correctness_function", "repair_list"]:
                print(f"Warning: unknown key {key} in callback result")

        return None  # Normal case - no special return needed

    def _handle_canvas_data(self, res_value, res, response):
        """Handle canvas data updates."""
        if "updates_canvas" in res and res["updates_canvas"]:
            response.canvas_image_content = array_to_data_url(res_value)

    def _handle_repair_variations(self, res_value, response):
        """Handle repair variations logic."""
        solver = self.state.get_solver()
        state = self.state
        
        result = "Saved to file(s): ["
        generate_all = res_value.get("generate_all", False)
        variations = res_value["variations"]
        tmp = []

        for i, packet_to_repair in enumerate(state.common_packets):
            if packet_to_repair:
                for chunk_id in range(solver.decoder.GEPP.chunk_to_used_packets.shape[1]):
                    if (solver.decoder.GEPP.chunk_to_used_packets[chunk_id, i] and
                        chunk_id in variations):
                        tmp.append(solver.repair_and_store_by_packet(
                            chunk_id, i, variations[chunk_id], len(tmp) == 0))
                        if not generate_all:
                            break

        result += f"{', '.join(tmp)}]"
        response.info_str = result

    def _handle_repair_for_each_packet(self, res_value, response):
        """Handle repair for each packet logic."""
        solver = self.state.get_solver()

        result = "Saved to file(s): ["
        generate_all = res_value.get("generate_all", False)
        correctness_function = res_value.get("correctness_function", None)
        repair_list = res_value["repair_list"]
        tmp = []

        for possible_packet_ids, invalid_row, repaired_content_row in repair_list:
            for i, packet_to_repair in enumerate(possible_packet_ids):
                tmp.append(solver.repair_and_store_by_packet(
                    invalid_row, packet_to_repair, repaired_content_row,
                    len(tmp) == 0, correctness_function))
                if not generate_all and any(x.startswith("CORRECT_") for x in tmp):
                    break

        result += f"{', '.join(tmp)}]"
        response.info_str = result

    def _handle_repair(self, res_value, res, response):
        """Handle repair logic."""
        from app import repair_chunks, propagate_gepp_update

        if "chunk_tag" in res:
            self.update_chunk_tag(res["chunk_tag"])

        # Perform the actual repair - this calls propagate_gepp_update() and recalculate_view() internally
        repair_chunks_res = repair_chunks(
            res_value["corrected_row"],
            "".join([x.replace("0x", "").zfill(2) for x in np.vectorize(hex)(res_value["corrected_value"])])
        )

        return (response.info_str, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update) + repair_chunks_res + (
                   dash.no_update, dash.no_update, response.canvas_image_content, response.kaitai_view)


class ButtonCallbackHandler:
    """Handles button-related callbacks."""

    def __init__(self, state: AppState, recalculate_view_func, propagate_gepp_update_func,
                 get_chunk_tag_func, update_chunk_tag_func, reset_chunk_tag_func, reset_column_tag_func,
                 propagete_chunk_tag_update_func, repair_chunks_func):
        self.state = state
        self.recalculate_view = recalculate_view_func
        self.propagate_gepp_update = propagate_gepp_update_func
        self.get_chunk_tag = get_chunk_tag_func
        self.update_chunk_tag = update_chunk_tag_func
        self.reset_chunk_tag = reset_chunk_tag_func
        self.reset_column_tag = reset_column_tag_func
        self.propagete_chunk_tag_update = propagete_chunk_tag_update_func
        self.repair_chunks = repair_chunks_func
    
    @property
    def solver(self):
        """Get the solver from state."""
        return self.state.get_solver()

    def handle_analyze_button(self):
        """Handle analyze button click."""
        response = CallbackResponse()
        return response.create_standard_response(self.recalculate_view())

    def handle_repair_exclusion_button(self):
        """Handle repair exclusion button click."""
        state = self.state
        solver = self.solver

        response = CallbackResponse()
        res, gepp = solver.repair_by_exclusion(state.common_packets)

        if res:
            solver.decoder.GEPP = gepp
            self.propagate_gepp_update()
            return response.create_standard_response(self.recalculate_view())
        else:
            response.info_str = "No solution without the corrupt packet(s) found."
            return response.create_standard_response()

    def handle_calculate_rank_button(self):
        """Handle calculate rank button click."""
        response = CallbackResponse()
        rank_a = self.solver.calculate_rank_A()
        rank_augmented_matrix = self.solver.calculate_rank_augmented_matrix()

        if rank_augmented_matrix < self.solver.decoder.number_of_chunks:
            tmp_str = (f"augmented rank ({rank_augmented_matrix}) < number of chunks "
                      f"({self.solver.decoder.number_of_chunks}), "
                      f"but partial recovery might be possible.")
        else:
            tmp_str = "LES seems solvable."

        status_msg = (": Erroneous packet detectable!" if rank_a != rank_augmented_matrix
                     else f": {tmp_str} Either all packets are correct or the corrupt packet "
                          f"is not linear dependent in the LES. This will be a tough one.")

        response.info_str = f"rank(A)={rank_a}, rank(A|b)={rank_augmented_matrix}{status_msg}"
        return (response.info_str,) + response.create_no_update_tuple(13)

    def handle_reset_chunk_tag_button(self):
        """Handle reset chunk tag button click."""
        response = CallbackResponse()
        self.reset_chunk_tag()
        self.reset_column_tag()
        self.propagete_chunk_tag_update()
        return response.create_standard_response(self.recalculate_view())

    def handle_save_button(self):
        """Handle save button click."""
        response = CallbackResponse()
        try:
            filename = self.solver.decoder.saveDecodedFile(return_file_name=True, print_to_output=False)
        except ValueError as ve:
            filename = ve.args[1]

        response.info_str = filename
        return response.create_standard_response()

    def handle_packet_tag_buttons(self, trigger_id, packet_tag_chunk_input):
        """Handle packet tag buttons (valid/invalid)."""
        response = CallbackResponse()

        try:
            packet_id = int(packet_tag_chunk_input)
            if packet_id < 0 or packet_id > self.solver.decoder.GEPP.b.shape[0]:
                raise ValueError
        except (ValueError, TypeError):
            response.info_str = "Chosen packet is not a number or not in range!"
            return (response.info_str, dash.no_update)

        tag_num = 1 if trigger_id == "packet-tag-chunk-invalid-button" else 2
        self.update_chunk_tag(
            self.solver.get_corrupt_chunks_by_packets([packet_id], self.get_chunk_tag(), tag_num)
        )
        return response.create_standard_response(self.recalculate_view())

    def handle_mode_switch(self, mode_value):
        """Handle mode switch toggle."""
        response = CallbackResponse()
        self.solver.set_multi_error_mode(mode_value)
        return response.create_standard_response(self.recalculate_view())

    def handle_colorblind_switch(self, colorblind_value):
        """Handle colorblind switch toggle."""
        from app import (correct_button_style, incorrect_button_style,
                        colorblind_correct, colorblind_incorrect,
                        green_button_style, red_button_style)

        response = CallbackResponse()

        # This is a bit tricky due to global variables, but we need to update the module-level variables
        import app
        if colorblind_value:
            app.correct_button_style = colorblind_correct
            app.incorrect_button_style = colorblind_incorrect
        else:
            app.correct_button_style = green_button_style
            app.incorrect_button_style = red_button_style

        return response.create_standard_response(self.recalculate_view())

    def handle_repair_reorder_buttons(self, trigger_id):
        """Handle repair reorder buttons."""
        from app import fast_most_common_matrix

        response = CallbackResponse()
        only_possible_invalid_packets = trigger_id == "repair-reorder-button-possible"
        gepp_backup = copy.deepcopy(self.solver.decoder.GEPP)
        state = self.state

        if not state.common_packets or len(state.common_packets) == 0:
            raise RuntimeError("Calculate corrupt packets first!")

        mapping = self.solver.all_solutions_by_reordering(state.common_packets, only_possible_invalid_packets)
        differing_gepps = set()
        differing_gepp_ids = set()
        working_dir = "reordered_solution"

        # Clean and create working directory
        if Path(working_dir).exists():
            shutil.rmtree(working_dir)
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        # Find differing solutions
        for i, tmp_gepp in mapping.items():
            if not np.array_equal(tmp_gepp.b[:self.solver.decoder.number_of_chunks],
                                 self.solver.decoder.GEPP.b[:self.solver.decoder.number_of_chunks]):
                differing_gepps.add(tmp_gepp.b[:self.solver.decoder.number_of_chunks].tobytes())
                differing_gepp_ids.add(i)

        # Save differing solutions
        result = f"Saved {len(differing_gepps)} differing solutions by reordering the packets in folder {working_dir}: ["
        for differing_gepp_id in differing_gepp_ids:
            is_correct = (self.solver.headerChunk is not None and
                         self.solver.headerChunk.checksum_len_format is not None and
                         self.solver.is_checksum_correct())

            self.solver.decoder.GEPP = mapping[differing_gepp_id]
            try:
                filename = self.solver.decoder.saveDecodedFile(return_file_name=True, print_to_output=False)
            except ValueError as ve:
                filename = ve.args[1]

            # Rename file with ID
            file_path = Path(filename)
            stem = ("CORRECT_" if is_correct else "") + file_path.stem + f"_{differing_gepp_id}"
            new_path = Path(working_dir) / (stem + file_path.suffix)
            file_path.rename(new_path)
            result += f"{new_path.name}, "

        result += "]"

        # Process results
        tmp = [mapping[x].b for x in differing_gepp_ids]
        if len(tmp) == 0:
            response.info_str = "No differing solutions found!"
            return response.create_standard_response()

        matrix_3d = np.dstack(tmp)
        most_common_vals, has_single_val = fast_most_common_matrix(matrix_3d)
        self.solver.decoder.GEPP = gepp_backup

        # Update chunk tags for valid rows
        chunk_tag = state.get_chunk_tag()
        valid_rows = [i for i, v in enumerate(np.all(has_single_val, axis=1))
                     if v and i < self.solver.decoder.GEPP.A.shape[1]]

        for i in valid_rows:
            if chunk_tag[i] < 1 and i:
                chunk_tag[i] = 2

        state.update_chunk_tag(chunk_tag)
        self.propagete_chunk_tag_update()
        response.info_str = result
        return response.create_standard_response(self.recalculate_view())

    def handle_forceload_plugin_button(self, c_ctx):
        """Handle force load plugin button."""
        from app import all_plugins_childs, show_canvas

        response = CallbackResponse()
        canvas_style = dash.no_update
        plugin_manager = self.get_plugin_manager()

        for plugin in plugin_manager.plugin_instances:
            if plugin.__class__.__name__ == c_ctx.triggered_id["index"]:
                div = html.Div(
                    id="plugin_" + plugin.__class__.__name__.lower(),
                    className="box",
                    children=plugin_manager.load_plugin(plugin)
                )
                plugin.on_load()
                all_plugins_childs.append(div)
                canvas_style = {"display": "block"} if show_canvas else {"display": "none"}

        return (response.info_str, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                all_plugins_childs, canvas_style, response.canvas_image_content, response.kaitai_view)


class RepairCallbackHandler:
    """Handles repair-related callbacks."""

    def __init__(self, repair_callback_func, repair_chunks_func):
        self.repair_callback = repair_callback_func
        self.repair_chunks = repair_chunks_func

    def handle_repair_inputs(self, trigger_id, c_ctx):
        """Handle repair button and input callbacks."""
        response = CallbackResponse()

        repair_result = self.repair_callback(
            trigger_id,
            c_ctx.inputs.get("repair-button.n_clicks"),
            c_ctx.inputs.get('repair-id-input-box.value'),
            c_ctx.inputs.get('hex-repair-input.value'),
            c_ctx.inputs.get('txt-repair-input.value')
        )

        return repair_result + (dash.no_update, dash.no_update, "", dash.no_update,
                               dash.no_update, response.canvas_image_content, response.kaitai_view)

    def handle_repair_chunks_button(self, c_ctx):
        """Handle repair chunks button."""
        response = CallbackResponse()

        repair_result = self.repair_chunks(
            c_ctx.inputs.get('repair-id-input-box.value'),
            c_ctx.inputs.get('hex-repair-input.value')
        )

        return (response.info_str, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update) + repair_result + (
                   dash.no_update, dash.no_update, response.canvas_image_content, response.kaitai_view)
