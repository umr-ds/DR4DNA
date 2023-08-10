# noinspection PyUnresolvedReferences
from dash_extensions.enrich import html, dcc, State, MATCH, ALL, Output, DashProxy, Input, MultiplexerTransform

# ----------------------- Callbacks -----------------------------

def callbacks(app):
    """
    @app.callback(Output('dashCanvas', 'lineColor'),
                  Input('color-picker', 'value'))
    def update_canvas_linecolor(value):
        if isinstance(value, dict):
            return value['hex']
        else:
            return value
    """
    """
    @app.callback(output=[Output('dashCanvas', 'image_content'), Output('dashCanvas', 'json_data')],
                  inputs=[Input({'type': 'plugin_io_btn', 'index': ALL}, 'n_clicks'),
                          Input({'type': 'output-data-upload', 'index': ALL}, 'value'),
                          State('dashCanvas', 'image_content'),
                          State('dashCanvas', 'json_data')],
                  prevent_inital_call=True)
    def canvas_callback_handler(*args, **kwargs):
        print("canvas_callback_handler", args, kwargs)
        return [None, None]
    """
    """
    @app.callback(Output('dashCanvas', 'lineWidth'),
                  Input('bg-width-slider', 'value'))
    def update_canvas_linewidth(value):
        return value
    """

    @app.callback(Output({'type': 'dashCanvas', 'index': MATCH}, 'width'),
                  Output({'type': 'output-data-upload', 'index': MATCH}, 'children'),
                  Input({'type': 'upload-data', 'index': MATCH}, 'contents'),
                  State({'type': 'upload-data', 'index': MATCH}, 'filename'),
                  State({'type': 'upload-data', 'index': MATCH}, 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = []
            # parse_contents(c, n, d) for c, n, d in
            # zip(list_of_contents, list_of_names, list_of_dates)]
            return children

