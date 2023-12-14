import typing
from singleton_decorator.decorator import singleton
from dash_extensions.enrich import html, dcc
import dash_daq as daq

from repair_algorithms.FileSpecificRepair import FileSpecificRepair


@singleton
class PluginManager:

    def __init__(self):
        self.plugins = []  # list of cls references! NOT instances!
        self.plugin_instances: typing.List[FileSpecificRepair] = []  # list of instances

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    def get_plugins(self):
        return self.plugins

    def get_plugin_instances(self):
        return self.plugin_instances

    def load_plugin(self, plugin_inst):
        global input_callback_handler, show_canvas

        plugin_inst.on_load()

        # Get the UI elements from the plugin instance:
        ui: typing.Dict[
            str, typing.Dict[str, typing.Union[str, bool, typing.Callable]]] = plugin_inst.get_ui_elements()
        # Initialize a list to store the plugin's child elements:
        _plugin_childs = [html.H4(f'Plugin: "{plugin_inst.__class__.__name__}"', className="tag")]

        # Iterate over the UI elements and create the corresponding Dash elements:
        for key, value in ui.items():
            if value["type"] == "button":
                _plugin_childs.append(
                    html.Button(value["text"], id={'type': 'plugin_io_btn', 'index': key}, className="button"))
                if "updates_canvas" in value and value["updates_canvas"]:
                    show_canvas = True
            elif value["type"] == "int":
                default_value = 0 if "default" not in value else value["default"]
                _plugin_childs.append(html.Div([html.Label(value["text"], className="label"),
                                                html.Div(
                                                    [dcc.Input(id={'type': 'plugin_io_value', 'index': key},
                                                               type="number",
                                                               className="input", value=default_value), ],
                                                    className="control")], className="field"))
                if "updates_canvas" in value and value["updates_canvas"]:
                    show_canvas = True
            elif value["type"] == "text":
                _plugin_childs.append(html.Div([html.Label(value["text"], className="label"),
                                                html.Div(
                                                    [dcc.Input(id={'type': 'plugin_io_value', 'index': key},
                                                               type="text",
                                                               className="input"), ],
                                                    className="control")], className="field"))
                if "updates_canvas" in value and value["updates_canvas"]:
                    show_canvas = True
            elif value["type"] == "canvas":
                show_canvas = True
            elif value["type"] == "kaitai_view":
                _plugin_childs.append(
                    html.Button(value["text"], id={'type': 'plugin_io_btn', 'index': key}, className="button"))
            elif value["type"] == "upload":
                _plugin_childs.append(html.Div([
                    dcc.Upload(
                        id={'type': 'plugin_io_upload-data', 'index': key},
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Don't allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Div(id={'type': 'output-data-upload', 'index': key}),
                ]))
                if "updates_canvas" in value and value["updates_canvas"]:
                    show_canvas = True
            elif value["type"] == "download":
                _plugin_childs.append(dcc.Download(id={'type': 'plugin_io_download-data', 'index': key}))
                _plugin_childs.append(
                    html.A('Download Data', id={'type': 'plugin_io_download', 'index': key}, className="button")
                )
            elif value["type"] == "toggle":
                _plugin_childs.append(html.Div([html.Label(value["off_label"]),
                                                daq.ToggleSwitch(id={'type': 'plugin_io_switch', 'index': key},
                                                                 label=value["label"],
                                                                 labelPosition='bottom', className="inline-switch"
                                                                 ), html.Label(value["on_label"])]))
        return _plugin_childs
