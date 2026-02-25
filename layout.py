import dash_extensions.enrich as dash
from dash_extensions.enrich import html, dcc, State, MATCH, ALL, Output, DashProxy, Input, MultiplexerTransform
from dash import ctx
import dash_daq as daq
from dash_canvas.DashCanvas import DashCanvas

def gen_app_layout(semi_automatic_solver, max_chunk_tag, force_load_plugins, all_plugins_childs, show_canvas, child):
    return html.Div(children=[
                       # Hidden stores for state management
                       dcc.Store(id='error-store', data=[]),
                       dcc.Store(id='loading-state', data={'is_loading': False, 'message': ''}),
                       
                       # Error notification container
                       html.Div(id="error-notification-container", children=[], className="error-container"),
                       
                       # Global loading indicator
                       html.Div(id="global-loading-indicator", children=[], className="global-loading"),
                       
                       dcc.Interval(id='interval-component', interval=1 * 1000,  # in milliseconds
                                    n_intervals=0),
                       # Genereic overview:
                       html.H1(children='DR4DNA', id="analytics-input"),
                       html.H3(children=semi_automatic_solver.predict_file_type(), id="analytics-output"),
                       html.H3(children="Possible invalid packets:", id="analyze-count-output",
                               className="box"),
                       html.Div([dcc.Loading(id="ls-loading-2", type="circle", color="#1890ff",
                                             children=[html.Div([html.Div(id="ls-loading-output-2")])])]),
                       # Single- vs Multi-Error-Mode:
                       html.Div([html.Label("Single"),
                                 daq.ToggleSwitch(id="mode-switch",
                                                  label='Currupt packet mode',
                                                  labelPosition='bottom', className="inline-switch"
                                                  ), html.Label("Multiple"),
                                 ]),
                       # Colorblind switch:
                       html.Div([html.Label("Normal mode"),
                                 daq.ToggleSwitch(id="colorblind-switch",
                                                  label='Colorblind mode',
                                                  labelPosition='bottom', className="inline-switch"
                                                  ), html.Label("Colorblind mode"),
                                 ]),
                       # Manage + repair Buttons:
                       html.Div([
                           html.Button('Calculate rank of the LES', id='calculate-rank-button',
                                       className="button"),
                           html.Button('Reset chunk tag', id='reset-chunk-tag-button', className="button"),
                           html.Button('Calculate corrupt packet', id='analyze-button', n_clicks=0,
                                       className="button"),
                           html.Button('Repair by exclusion', id='repair-exclusion-button', n_clicks=0,
                                       className="button"),
                           html.Button('Find solutions by reordering', id='repair-reorder-button',
                                       n_clicks=0,
                                       className="button"),
                           html.Button('Find solutions by reordering (partial)',
                                       id='repair-reorder-button-possible',
                                       n_clicks=0,
                                       className="button"),
                           html.Button('Save file', id='save-button', n_clicks=0, className="button"),
                           # (In)Valid Packet tagging:
                           html.Div([html.Button('Tag affected chunks as invalid',
                                                 id='packet-tag-chunk-invalid-button', className="button"),
                                     html.Button('Tag affected chunks as valid',
                                                 id='packet-tag-chunk-valid-button', className="button"),
                                     dcc.Input(id='packet-tag-chunk-input', type="number",
                                               className="input",
                                               placeholder="Packet id")]),
                           # Repair Window code:
                           html.Div([html.Button('Open repair window', id='repair-button', n_clicks=0,
                                                 className="button"),
                                     dcc.Input(id='repair-id-input-box', type='number', min=0, step=1,
                                               max=len(max_chunk_tag), className="input",
                                               placeholder='Id of chunk to repair')]),
                           html.Div(
                               [dcc.Input(id='hex-repair-input', type='text', placeholder='Hex to repair',
                                          className="input"),
                                dcc.Input(id='txt-repair-input', type='text', placeholder='Text to repair',
                                          className="input"),
                                html.Button('Repair', id='repair-chunks-button', n_clicks=0,
                                            className="button")],
                               hidden=True, id="repair-input"),
                       ], className="box"),
                       # manual plugin loading:
                       html.Div(id="plugin_load_container", children=force_load_plugins),
                       # plugins:
                       html.Div(id="plugin_view", children=all_plugins_childs, n_clicks=0),
                       # canvas:
                       html.Div(id="canvas",
                                style=(
                                    {"image-rendering": "pixelated",
                                     "display": "block"} if show_canvas else {
                                        "image-rendering": "pixelated", "display": "none"}),
                                children=html.Div([
                                    html.Div([
                                        DashCanvas(
                                            id='dashCanvas',
                                            lineWidth=1,
                                            image_content='{}',
                                            tool='line',
                                            hide_buttons=['pencil'],  # 'line', 'zoom', 'pan'],
                                        ),
                                        html.Canvas(id='canvas-output'),
                                    ], className="six columns"),
                                ])),
                       html.Div(id="kaitai_view"),
                       # hex / normal view for decoded data:
                       html.Div(id="row_view", children=child, n_clicks=0, className="box"),
                       ])