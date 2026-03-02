"""Plugin manager for DR4DNA repair algorithms.

This module provides a singleton plugin manager that handles registration,
loading, and UI generation for file repair plugins. It dynamically creates
Dash UI elements based on plugin configurations.
"""

import typing

import dash_daq as daq
from dash_extensions.enrich import dcc, html
from singleton_decorator.decorator import singleton

from repair_algorithms.FileSpecificRepair import FileSpecificRepair


@singleton
class PluginManager:
    """Singleton manager for plugin registration and loading."""

    def __init__(self):
        """
        Initialize the plugin manager.

        Sets up empty lists for storing plugin classes and instances.
        """
        self.plugins = []  # list of cls references! NOT instances!
        self.plugin_instances: typing.List[FileSpecificRepair] = []  # list of instances

    def register_plugin(self, plugin):
        """
        Register a plugin class for later loading.

        Args:
            plugin: Plugin class to register
        """
        self.plugins.append(plugin)

    def get_plugins(self):
        """
        Get all registered plugin classes.

        Returns:
            List of registered plugin classes
        """
        return self.plugins

    def get_plugin_instances(self):
        """
        Get all loaded plugin instances.

        Returns:
            List of loaded plugin instances
        """
        return self.plugin_instances

    def _create_button_element(self, key, value):
        """
        Create a button UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary containing 'text'

        Returns:
            Dash html.Button component
        """
        return html.Button(
            value["text"],
            id={"type": "plugin_io_btn", "index": key},
            className="button",
        )

    def _create_int_input_element(self, key, value):
        """
        Create an integer input UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary containing 'text' and optional 'default'

        Returns:
            Dash html.Div containing input field
        """
        default_value = 0 if "default" not in value else value["default"]
        return html.Div(
            [
                html.Div(value["text"], className="label"),
                html.Div(
                    [
                        dcc.Input(
                            id={"type": "plugin_io_value", "index": key},
                            type="number",
                            className="input",
                            value=default_value,
                        ),
                    ],
                    className="control",
                ),
            ],
            className="field",
        )

    def _create_text_input_element(self, key, value):
        """
        Create a text input UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary containing 'text'

        Returns:
            Dash html.Div containing text input field
        """
        return html.Div(
            [
                html.Div(value["text"], className="label"),
                html.Div(
                    [
                        dcc.Input(
                            id={"type": "plugin_io_value", "index": key},
                            type="text",
                            className="input",
                        ),
                    ],
                    className="control",
                ),
            ],
            className="field",
        )

    def _create_upload_element(self, key, value):
        """
        Create a file upload UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary

        Returns:
            Dash html.Div containing upload component
        """
        return html.Div(
            [
                dcc.Upload(
                    id={"type": "plugin_io_upload-data", "index": key},
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    multiple=False,
                ),
                html.Div(id={"type": "output-data-upload", "index": key}),
            ]
        )

    def _create_download_element(self, key, value):
        """
        Create a download UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary

        Returns:
            List containing Dash download component and link
        """
        return [
            dcc.Download(id={"type": "plugin_io_download-data", "index": key}),
            html.A(
                "Download Data",
                id={"type": "plugin_io_download", "index": key},
                className="button",
            ),
        ]

    def _create_toggle_element(self, key, value):
        """
        Create a toggle switch UI element for a plugin.

        Args:
            key: Element identifier key
            value: Element configuration dictionary containing 'off_label', 'on_label', and 'label'

        Returns:
            Dash html.Div containing toggle switch
        """
        return html.Div(
            [
                html.Span(value["off_label"]),
                daq.ToggleSwitch(
                    id={"type": "plugin_io_switch", "index": key},
                    label=value["label"],
                    labelPosition="bottom",
                    className="inline-switch",
                ),
                html.Span(value["on_label"]),
            ]
        )

    def _check_updates_canvas(self, value):
        """
        Check if element updates canvas and update global flag.

        Args:
            value: Element configuration dictionary
        """
        global show_canvas
        if "updates_canvas" in value and value["updates_canvas"]:
            show_canvas = True

    def _create_ui_element(self, key, value):
        """
        Create UI element based on type specification.

        Creates the appropriate Dash UI component based on the element type
        specified in the value configuration.

        Args:
            key: Element identifier key
            value: Element configuration dictionary containing 'type' and other settings

        Returns:
            Dash UI element or None for canvas type elements
        """
        element_creators = {
            "button": self._create_button_element,
            "int": self._create_int_input_element,
            "text": self._create_text_input_element,
            "upload": self._create_upload_element,
            "download": self._create_download_element,
            "toggle": self._create_toggle_element,
        }

        elem_type = value["type"]

        # Handle special types
        if elem_type == "canvas":
            self._check_updates_canvas(value)
            return None
        elif elem_type == "kaitai_view":
            return self._create_button_element(key, value)

        # Handle standard types
        if elem_type in element_creators:
            element = element_creators[elem_type](key, value)
            self._check_updates_canvas(value)
            return element

        return None

    def load_plugin(self, plugin_inst):
        """
        Load a plugin and generate its UI elements.

        Args:
            plugin_inst: Plugin instance to load

        Returns:
            List of UI elements for the plugin
        """
        global input_callback_handler, show_canvas

        plugin_inst.on_load()

        # Get the UI elements from the plugin instance:
        ui: typing.Dict[
            str, typing.Dict[str, typing.Union[str, bool, typing.Callable]]
        ] = plugin_inst.get_ui_elements()
        # Initialize a list to store the plugin's child elements:
        _plugin_childs = [html.H4(f'Plugin: "{plugin_inst.__class__.__name__}"', className="tag")]

        # Iterate over the UI elements and create the corresponding Dash elements:
        for key, value in ui.items():
            element = self._create_ui_element(key, value)
            if element is not None:
                if isinstance(element, list):
                    _plugin_childs.extend(element)
                else:
                    _plugin_childs.append(element)
        return _plugin_childs
