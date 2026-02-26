# -*- coding: utf-8 -*-
"""
Callback handlers for DR4DNA application.

This module provides handler classes for processing different types of callbacks:
- PluginCallbackHandler: Handles plugin I/O operations
- ButtonCallbackHandler: Handles button click events
- RepairCallbackHandler: Handles repair-related operations

Each handler class follows the single responsibility principle and uses
dependency injection for testability.
"""

import copy
import shutil
import typing
from pathlib import Path

import dash_extensions.enrich as dash
import numpy as np
from dash_canvas.utils.io_utils import array_to_data_url
from dash_extensions.enrich import html

from constants import (
    CALLBACK_RESPONSE_SIZE,
    ERROR_CALCULATE_CORRUPT_PACKETS_FIRST,
    ERROR_CHOSEN_PACKET_INVALID,
    ERROR_NO_SOLUTION_WITHOUT_CORRUPT,
    RANK_STATUS_AMBIGUOUS,
    RANK_STATUS_DETECTABLE,
    RANK_STATUS_PARTIAL_RECOVERY,
    RANK_STATUS_SOLVABLE,
    WORKING_DIR_REORDERED_SOLUTION,
)
from logger import get_logger
from state import AppState
from utils import create_no_update_tuple

logger = get_logger(__name__)


class CallbackResponse:
    """
    Helper class to manage callback response data.

    This class provides a clean interface for building callback responses
    with consistent structure.

    Attributes:
        info_str: Information message to display
        canvas_image_content: Canvas image data
        kaitai_view: Kaitai view content
        refresh_view: Flag indicating if view should be refreshed
        update_gepp: Flag indicating if GEPP should be updated
    """

    def __init__(self):
        """Initialize callback response with default values."""
        self.info_str: typing.Any = dash.no_update
        self.canvas_image_content: typing.Any = dash.no_update
        self.kaitai_view: typing.Any = dash.no_update
        self.refresh_view: bool = False
        self.update_gepp: bool = False

    def create_no_update_tuple(self, size: int = CALLBACK_RESPONSE_SIZE) -> tuple:
        """
        Create a tuple of dash.no_update values.

        Args:
            size: Size of the tuple (default: CALLBACK_RESPONSE_SIZE)

        Returns:
            Tuple of no_update values
        """
        return create_no_update_tuple(size)

    def create_standard_response(self, recalc_view_result: typing.Optional[tuple] = None) -> tuple:
        """
        Create a standard response tuple.

        Args:
            recalc_view_result: Optional result from recalculate_view()

        Returns:
            Properly structured response tuple
        """
        if recalc_view_result:
            return (
                (
                    self.info_str,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
                + recalc_view_result
                + (dash.no_update, dash.no_update, self.canvas_image_content, self.kaitai_view)
            )
        return (
            (self.info_str,)
            + self.create_no_update_tuple(11)
            + (self.canvas_image_content, self.kaitai_view)
        )


class PluginCallbackHandler:
    """
    Handles plugin-related callbacks.

    This handler manages all plugin I/O operations including:
    - Plugin data upload/download
    - Plugin button clicks
    - Plugin value changes

    Args:
        state: Application state instance
        plugin_manager: Plugin manager instance
        get_chunk_tag_func: Function to get chunk tags
        update_chunk_tag_func: Function to update chunk tags
        update_column_tag_func: Function to update column tags
        recalculate_view_func: Function to recalculate the view
        propagate_gepp_update_func: Function to propagate GEPP updates
    """

    def __init__(
        self,
        state: AppState,
        plugin_manager: typing.Any,
        get_chunk_tag_func: typing.Callable,
        update_chunk_tag_func: typing.Callable,
        update_column_tag_func: typing.Callable,
        recalculate_view_func: typing.Callable,
        propagate_gepp_update_func: typing.Callable,
    ):
        """Initialize the plugin callback handler."""
        self.state = state
        self.plugin_manager = plugin_manager
        self.get_chunk_tag = get_chunk_tag_func
        self.update_chunk_tag = update_chunk_tag_func
        self.update_column_tag = update_column_tag_func
        self.recalculate_view = recalculate_view_func
        self.propagate_gepp_update = propagate_gepp_update_func

    def get_plugin_manager(self) -> typing.Any:
        """
        Get plugin manager from state.

        Returns:
            Plugin manager instance
        """
        return self.state.get_plugin_manager()

    def handle_plugin_io(self, trigger_id: str, c_ctx: typing.Any, *args, **kwargs) -> tuple:
        """
        Handle plugin I/O callbacks.

        Args:
            trigger_id: The ID of the triggered element
            c_ctx: Dash callback context
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        plugin_manager = self.get_plugin_manager()

        for plugin in plugin_manager.plugin_instances:
            if not plugin.active:
                continue

            ui = plugin.get_ui_elements()
            for key, value in ui.items():
                if trigger_id == key:
                    return self._process_plugin_callback(value, c_ctx, response, *args, **kwargs)

        # No matching plugin found
        logger.debug(f"No plugin found for trigger_id: {trigger_id}")
        return response.create_standard_response()

    def _process_plugin_callback(
        self, value: typing.Dict, c_ctx: typing.Any, response: CallbackResponse, *args, **kwargs
    ) -> tuple:
        """
        Process a plugin callback and build response.

        Args:
            value: Plugin callback configuration
            c_ctx: Dash callback context
            response: Callback response object
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Callback response tuple
        """
        chunk_tag = self.get_chunk_tag()
        res = value["callback"](chunk_tag=chunk_tag, c_ctx=c_ctx, *args, **kwargs)

        # Check for special "repair" handler
        special_return = self._process_plugin_response(res, response)
        if special_return is not None:
            return special_return

        # Handle standard response
        if response.refresh_view or response.update_gepp:
            if response.update_gepp:
                self.propagate_gepp_update()
            return response.create_standard_response(self.recalculate_view())
        else:
            return response.create_standard_response()

    def _process_plugin_response(
        self, res: typing.Dict, response: CallbackResponse
    ) -> typing.Optional[tuple]:
        """
        Process plugin callback response.

        Args:
            res: Plugin response dictionary
            response: Callback response object to update

        Returns:
            Special return value for "repair" handler, None otherwise
        """
        handlers = {
            "chunk_tag": lambda v: self.update_chunk_tag(v),
            "column_tag": lambda v: self.update_column_tag(v),
            "update_b": lambda v: setattr(response, "update_gepp", v),
            "refresh_view": lambda v: setattr(response, "refresh_view", v),
            "image_content": lambda v: setattr(response, "canvas_image_content", v),
            "canvas_data": lambda v: self._handle_canvas_data(v, res, response),
            "kaitai_content": lambda v: setattr(response, "kaitai_view", res["kaitai_content"]),
            "info": lambda v: setattr(response, "info_str", v),
            "repair_variations": lambda v: self._handle_repair_variations(v, response),
            "repair_for_each_packet": lambda v: self._handle_repair_for_each_packet(v, response),
        }

        for key, res_value in res.items():
            # Special case for "repair"
            if key == "repair":
                return self._handle_repair(res_value, res, response)
            elif key in handlers:
                handlers[key](res_value)
            elif key not in [
                "updates_canvas",
                "updates_b",
                "height",
                "width",
                "generate_all",
                "correctness_function",
                "repair_list",
            ]:
                logger.warning(f"Unknown key in callback result: {key}")

        return None

    def _handle_canvas_data(
        self, res_value: typing.Any, res: typing.Dict, response: CallbackResponse
    ) -> None:
        """
        Handle canvas data updates.

        Args:
            res_value: Canvas data value
            res: Full response dictionary
            response: Callback response object
        """
        if "updates_canvas" in res and res["updates_canvas"]:
            response.canvas_image_content = array_to_data_url(res_value)

    def _handle_repair_variations(self, res_value: typing.Dict, response: CallbackResponse) -> None:
        """
        Handle repair variations logic.

        Args:
            res_value: Repair variations data
            response: Callback response object
        """
        solver = self.state.get_solver()
        state = self.state

        result = "Saved to file(s): ["
        generate_all = res_value.get("generate_all", False)
        variations = res_value["variations"]
        tmp = []

        for i, packet_to_repair in enumerate(state.common_packets):
            if packet_to_repair:
                for chunk_id in range(solver.decoder.GEPP.chunk_to_used_packets.shape[1]):
                    if (
                        solver.decoder.GEPP.chunk_to_used_packets[chunk_id, i]
                        and chunk_id in variations
                    ):
                        tmp.append(
                            solver.repair_and_store_by_packet(
                                chunk_id, i, variations[chunk_id], len(tmp) == 0
                            )
                        )
                        if not generate_all:
                            break

        result += f"{', '.join(tmp)}]"
        response.info_str = result

    def _handle_repair_for_each_packet(
        self, res_value: typing.Dict, response: CallbackResponse
    ) -> None:
        """
        Handle repair for each packet logic.

        Args:
            res_value: Repair data
            response: Callback response object
        """
        solver = self.state.get_solver()

        result = "Saved to file(s): ["
        generate_all = res_value.get("generate_all", False)
        correctness_function = res_value.get("correctness_function", None)
        repair_list = res_value["repair_list"]
        tmp = []

        for possible_packet_ids, invalid_row, repaired_content_row in repair_list:
            for i, packet_to_repair in enumerate(possible_packet_ids):
                tmp.append(
                    solver.repair_and_store_by_packet(
                        invalid_row,
                        packet_to_repair,
                        repaired_content_row,
                        len(tmp) == 0,
                        correctness_function,
                    )
                )
                if not generate_all and any(x.startswith("CORRECT_") for x in tmp):
                    break

        result += f"{', '.join(tmp)}]"
        response.info_str = result

    def _handle_repair(
        self, res_value: typing.Dict, res: typing.Dict, response: CallbackResponse
    ) -> tuple:
        """
        Handle repair logic.

        Args:
            res_value: Repair value data
            res: Full response dictionary
            response: Callback response object

        Returns:
            Complete callback response tuple
        """
        from app import propagate_gepp_update, repair_chunks

        if "chunk_tag" in res:
            self.update_chunk_tag(res["chunk_tag"])

        # Perform the actual repair
        repair_chunks_res = repair_chunks(
            res_value["corrected_row"],
            "".join(
                [
                    x.replace("0x", "").zfill(2)
                    for x in np.vectorize(hex)(res_value["corrected_value"])
                ]
            ),
        )
        propagate_gepp_update()

        return (
            (
                response.info_str,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
            + repair_chunks_res
            + (dash.no_update, dash.no_update, response.canvas_image_content, response.kaitai_view)
        )


class ButtonCallbackHandler:
    """
    Handles button-related callbacks.

    This handler manages all button click events including:
    - Analysis buttons
    - Repair buttons
    - Mode switches
    - Plugin loading buttons

    Args:
        state: Application state instance
        recalculate_view_func: Function to recalculate the view
        propagate_gepp_update_func: Function to propagate GEPP updates
        get_chunk_tag_func: Function to get chunk tags
        update_chunk_tag_func: Function to update chunk tags
        reset_chunk_tag_func: Function to reset chunk tags
        reset_column_tag_func: Function to reset column tags
        propagate_chunk_tag_update_func: Function to propagate chunk tag updates
        repair_chunks_func: Function to repair chunks
    """

    def __init__(
        self,
        state: AppState,
        recalculate_view_func: typing.Callable,
        propagate_gepp_update_func: typing.Callable,
        get_chunk_tag_func: typing.Callable,
        update_chunk_tag_func: typing.Callable,
        reset_chunk_tag_func: typing.Callable,
        reset_column_tag_func: typing.Callable,
        propagate_chunk_tag_update_func: typing.Callable,
        repair_chunks_func: typing.Callable,
    ):
        """Initialize the button callback handler."""
        self.state = state
        self.recalculate_view = recalculate_view_func
        self.propagate_gepp_update = propagate_gepp_update_func
        self.get_chunk_tag = get_chunk_tag_func
        self.update_chunk_tag = update_chunk_tag_func
        self.reset_chunk_tag = reset_chunk_tag_func
        self.reset_column_tag = reset_column_tag_func
        self.propagate_chunk_tag_update = propagate_chunk_tag_update_func
        self.repair_chunks = repair_chunks_func

    @property
    def solver(self) -> typing.Any:
        """Get the solver from state."""
        return self.state.get_solver()

    def get_plugin_manager(self) -> typing.Any:
        """
        Get the plugin manager from state.

        Returns:
            PluginManager instance
        """
        return self.state.get_plugin_manager()

    def handle_analyze_button(self) -> tuple:
        """
        Handle analyze button click.

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        return response.create_standard_response(self.recalculate_view())

    def handle_repair_exclusion_button(self) -> tuple:
        """
        Handle repair exclusion button click.

        Returns:
            Callback response tuple
        """
        state = self.state
        solver = self.solver

        response = CallbackResponse()
        res, gepp = solver.repair_by_exclusion(state.common_packets)

        if res:
            solver.decoder.GEPP = gepp
            self.propagate_gepp_update()
            return response.create_standard_response(self.recalculate_view())
        else:
            response.info_str = ERROR_NO_SOLUTION_WITHOUT_CORRUPT
            return response.create_standard_response()

    def handle_calculate_rank_button(self) -> tuple:
        """
        Handle calculate rank button click.

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        solver = self.solver

        rank_a = solver.calculate_rank_A()
        rank_augmented_matrix = solver.calculate_rank_augmented_matrix()

        if rank_augmented_matrix < solver.decoder.number_of_chunks:
            tmp_str = RANK_STATUS_PARTIAL_RECOVERY.format(
                rank_augmented_matrix, solver.decoder.number_of_chunks
            )
        else:
            tmp_str = RANK_STATUS_SOLVABLE

        status_msg = (
            RANK_STATUS_DETECTABLE
            if rank_a != rank_augmented_matrix
            else RANK_STATUS_AMBIGUOUS.format(tmp_str)
        )

        response.info_str = f"rank(A)={rank_a}, rank(A|b)={rank_augmented_matrix}{status_msg}"
        return (response.info_str,) + create_no_update_tuple(13)

    def handle_reset_chunk_tag_button(self) -> tuple:
        """
        Handle reset chunk tag button click.

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        self.reset_chunk_tag()
        self.reset_column_tag()
        self.propagate_chunk_tag_update()
        return response.create_standard_response(self.recalculate_view())

    def handle_save_button(self) -> tuple:
        """
        Handle save button click.

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        solver = self.solver

        try:
            filename = solver.decoder.saveDecodedFile(return_file_name=True, print_to_output=False)
        except ValueError as ve:
            filename = ve.args[1] if len(ve.args) > 1 else str(ve)

        response.info_str = filename
        return response.create_standard_response()

    def handle_packet_tag_buttons(
        self, trigger_id: str, packet_tag_chunk_input: typing.Union[str, int, None]
    ) -> tuple:
        """
        Handle packet tag buttons (valid/invalid).

        Args:
            trigger_id: The triggered button ID
            packet_tag_chunk_input: Packet ID input value

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()

        try:
            packet_id = int(packet_tag_chunk_input)
            if packet_id < 0 or packet_id > self.solver.decoder.GEPP.b.shape[0]:
                raise ValueError()
        except (ValueError, TypeError):
            response.info_str = ERROR_CHOSEN_PACKET_INVALID
            return (response.info_str, dash.no_update)

        tag_num = 1 if trigger_id == "packet-tag-chunk-invalid-button" else 2
        self.update_chunk_tag(
            self.solver.get_corrupt_chunks_by_packets([packet_id], self.get_chunk_tag(), tag_num)
        )
        return response.create_standard_response(self.recalculate_view())

    def handle_mode_switch(self, mode_value: bool) -> tuple:
        """
        Handle mode switch toggle.

        Args:
            mode_value: New mode value

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        self.solver.set_multi_error_mode(mode_value)
        return response.create_standard_response(self.recalculate_view())

    def handle_colorblind_switch(self, colorblind_value: bool) -> tuple:
        """
        Handle colorblind switch toggle.

        Args:
            colorblind_value: New colorblind mode value

        Returns:
            Callback response tuple
        """
        # Update module-level style variables
        import app
        from app import (
            BUTTON_STYLE_GREEN,
            BUTTON_STYLE_RED,
            colorblind_correct,
            colorblind_incorrect,
        )

        if colorblind_value:
            app.correct_button_style = colorblind_correct
            app.incorrect_button_style = colorblind_incorrect
        else:
            app.correct_button_style = BUTTON_STYLE_GREEN
            app.incorrect_button_style = BUTTON_STYLE_RED

        return self.recalculate_view()

    def handle_repair_reorder_buttons(self, trigger_id: str) -> tuple:
        """
        Handle repair reorder buttons.

        Args:
            trigger_id: The triggered button ID

        Returns:
            Callback response tuple
        """
        from app import fast_most_common_matrix

        only_possible_invalid_packets = trigger_id == "repair-reorder-button-possible"
        gepp_backup = copy.deepcopy(self.solver.decoder.GEPP)
        state = self.state

        if not state.common_packets or len(state.common_packets) == 0:
            raise RuntimeError(ERROR_CALCULATE_CORRUPT_PACKETS_FIRST)

        return self._process_reorder_solutions(
            only_possible_invalid_packets, gepp_backup, state, fast_most_common_matrix
        )

    def _process_reorder_solutions(
        self,
        only_possible_invalid_packets: bool,
        gepp_backup: typing.Any,
        state: AppState,
        fast_most_common_matrix_func: typing.Callable,
    ) -> tuple:
        """
        Process reorder solutions and save them.

        Args:
            only_possible_invalid_packets: Flag for partial mode
            gepp_backup: Backup of original GEPP
            state: Application state
            fast_most_common_matrix_func: Function to find common matrix values

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()
        solver = self.solver

        mapping = solver.all_solutions_by_reordering(
            state.common_packets, only_possible_invalid_packets
        )

        # Find and save differing solutions
        differing_gepp_ids = self._find_differing_solutions(mapping)
        working_dir = WORKING_DIR_REORDERED_SOLUTION

        result = self._save_differing_solutions(mapping, differing_gepp_ids, working_dir)

        # Process results and update chunk tags
        self._update_chunk_tags_from_solutions(
            mapping, differing_gepp_ids, gepp_backup, fast_most_common_matrix_func
        )

        response.info_str = result
        return response.create_standard_response(self.recalculate_view())

    def _find_differing_solutions(self, mapping: typing.Dict) -> set:
        """
        Find solutions that differ from current GEPP.

        Args:
            mapping: Dictionary of solution mappings

        Returns:
            Set of differing solution IDs
        """
        solver = self.solver
        differing_gepp_ids = set()

        for i, tmp_gepp in mapping.items():
            if not np.array_equal(
                tmp_gepp.b[: solver.decoder.number_of_chunks],
                solver.decoder.GEPP.b[: solver.decoder.number_of_chunks],
            ):
                differing_gepp_ids.add(i)

        return differing_gepp_ids

    def _save_differing_solutions(
        self, mapping: typing.Dict, differing_gepp_ids: set, working_dir: str
    ) -> str:
        """
        Save differing solutions to files.

        Args:
            mapping: Dictionary of solution mappings
            differing_gepp_ids: Set of differing solution IDs
            working_dir: Directory to save solutions

        Returns:
            Result message string
        """
        import os

        # Clean and create working directory
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        result = (
            f"Saved {len(differing_gepp_ids)} differing solutions by "
            f"reordering the packets in folder {working_dir}: ["
        )

        for differing_gepp_id in differing_gepp_ids:
            filename = self._save_single_solution(
                mapping[differing_gepp_id], differing_gepp_id, working_dir
            )
            result += f"{filename}, "

        result += "]"
        return result

    def _save_single_solution(self, gepp: typing.Any, solution_id: int, working_dir: str) -> str:
        """
        Save a single solution to file.

        Args:
            gepp: GEPP instance to save
            solution_id: Solution ID for filename
            working_dir: Directory to save to

        Returns:
            New filename
        """
        solver = self.solver

        is_correct = (
            solver.headerChunk is not None
            and solver.headerChunk.checksum_len_format is not None
            and solver.is_checksum_correct()
        )

        solver.decoder.GEPP = gepp

        try:
            filename = solver.decoder.saveDecodedFile(return_file_name=True, print_to_output=False)
        except ValueError as ve:
            filename = ve.args[1] if len(ve.args) > 1 else str(ve)

        # Rename file with ID
        file_path = Path(filename)
        prefix = "CORRECT_" if is_correct else ""
        stem = f"{prefix}{file_path.stem}_{solution_id}"
        new_path = Path(working_dir) / (stem + file_path.suffix)
        file_path.rename(new_path)

        return new_path.name

    def _update_chunk_tags_from_solutions(
        self,
        mapping: typing.Dict,
        differing_gepp_ids: set,
        gepp_backup: typing.Any,
        fast_most_common_matrix_func: typing.Callable,
    ) -> None:
        """
        Update chunk tags based on solution analysis.

        Args:
            mapping: Dictionary of solution mappings
            differing_gepp_ids: Set of differing solution IDs
            gepp_backup: Backup of original GEPP
            fast_most_common_matrix_func: Function to find common matrix values
        """
        solver = self.solver

        tmp = [mapping[x].b for x in differing_gepp_ids]
        if len(tmp) == 0:
            return

        matrix_3d = np.dstack(tmp)
        _, has_single_val = fast_most_common_matrix_func(matrix_3d)
        solver.decoder.GEPP = gepp_backup

        # Update chunk tags for valid rows
        chunk_tag = self.get_chunk_tag()
        valid_rows = [
            i
            for i, v in enumerate(np.all(has_single_val, axis=1))
            if v and i < solver.decoder.GEPP.A.shape[1]
        ]

        for i in valid_rows:
            if chunk_tag[i] < 1 and i:
                chunk_tag[i] = 2

        self.update_chunk_tag(chunk_tag)
        self.propagate_chunk_tag_update()

    def handle_forceload_plugin_button(self, c_ctx: typing.Any) -> tuple:
        """
        Handle force load plugin button.

        Args:
            c_ctx: Dash callback context

        Returns:
            Callback response tuple
        """
        from app import all_plugins_childs

        response = CallbackResponse()
        canvas_style = dash.no_update
        plugin_manager = self.get_plugin_manager()
        state = self.state

        for plugin in plugin_manager.plugin_instances:
            if plugin.__class__.__name__ == c_ctx.triggered_id["index"]:
                div = html.Div(
                    id="plugin_" + plugin.__class__.__name__.lower(),
                    className="box",
                    children=plugin_manager.load_plugin(plugin),
                )
                plugin.on_load()
                all_plugins_childs.append(div)
                canvas_style = {"display": "block"} if state.show_canvas else {"display": "none"}

        return (
            response.info_str,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            all_plugins_childs,
            canvas_style,
            response.canvas_image_content,
            response.kaitai_view,
        )


class RepairCallbackHandler:
    """
    Handles repair-related callbacks.

    This handler manages repair operations including:
    - Repair input handling
    - Repair execution

    Args:
        repair_callback_func: Function to handle repair inputs
        repair_chunks_func: Function to execute repair
    """

    def __init__(self, repair_callback_func: typing.Callable, repair_chunks_func: typing.Callable):
        """Initialize the repair callback handler."""
        self.repair_callback = repair_callback_func
        self.repair_chunks = repair_chunks_func

    def handle_repair_inputs(self, trigger_id: str, c_ctx: typing.Any) -> tuple:
        """
        Handle repair button and input callbacks.

        Args:
            trigger_id: The triggered element ID
            c_ctx: Dash callback context

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()

        repair_result = self.repair_callback(
            trigger_id,
            c_ctx.inputs.get("repair-button.n_clicks"),
            c_ctx.inputs.get("repair-id-input-box.value"),
            c_ctx.inputs.get("hex-repair-input.value"),
            c_ctx.inputs.get("txt-repair-input.value"),
        )

        return repair_result + (
            dash.no_update,
            dash.no_update,
            "",
            dash.no_update,
            dash.no_update,
            response.canvas_image_content,
            response.kaitai_view,
        )

    def handle_repair_chunks_button(self, c_ctx: typing.Any) -> tuple:
        """
        Handle repair chunks button.

        Args:
            c_ctx: Dash callback context

        Returns:
            Callback response tuple
        """
        response = CallbackResponse()

        repair_result = self.repair_chunks(
            c_ctx.inputs.get("repair-id-input-box.value"),
            c_ctx.inputs.get("hex-repair-input.value"),
        )

        return (
            (
                response.info_str,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
            + repair_result
            + (dash.no_update, dash.no_update, response.canvas_image_content, response.kaitai_view)
        )
