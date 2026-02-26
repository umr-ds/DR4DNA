# noinspection PyUnresolvedReferences
import traceback

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

# Store for error messages
_error_store = []


def add_error(message, traceback_str=None):
    """Add an error to the error store."""
    global _error_store
    _error_store.append({"message": message, "traceback": traceback_str or ""})
    # Keep only last 10 errors
    if len(_error_store) > 10:
        _error_store = _error_store[-10:]


def get_error_store():
    """Get the current error store."""
    return _error_store


def clear_errors():
    """Clear all errors."""
    global _error_store
    _error_store = []


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

    @app.callback(
        Output({"type": "dashCanvas", "index": MATCH}, "width"),
        Output({"type": "output-data-upload", "index": MATCH}, "children"),
        Input({"type": "upload-data", "index": MATCH}, "contents"),
        State({"type": "upload-data", "index": MATCH}, "filename"),
        State({"type": "upload-data", "index": MATCH}, "last_modified"),
    )
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = []
            # parse_contents(c, n, d) for c, n, d in
            # zip(list_of_contents, list_of_names, list_of_dates)]
            return children

    # Hidden store for error messages
    @app.callback(Output("error-store", "data"), Input("interval-component", "n_intervals"))
    def check_for_errors(n_intervals):
        """Periodically check for errors and update the store."""
        from app_callbacks import get_error_store

        return get_error_store()[-5:]  # Return last 5 errors

    @app.callback(
        Output("error-notification-container", "children"),
        Input("error-store", "data"),
        prevent_initial_call=False,
    )
    def update_error_notifications(errors):
        """Display error notifications in the UI."""
        if not errors:
            return []

        notifications = []
        for i, error_info in enumerate(errors):
            error_msg = error_info.get("message", "Unknown error")
            traceback_str = error_info.get("traceback", "")

            notification = html.Div(
                [
                    html.Button(
                        "×", className="close-btn", id={"type": "close-error-btn", "index": i}
                    ),
                    html.Strong("❌ Error:"),
                    html.Div(f"{error_msg}\n\n{traceback_str}"),
                ],
                className="error-notification",
                id={"type": "error-notification", "index": i},
            )
            notifications.append(notification)

        return notifications

    @app.callback(
        Output("error-store", "data", allow_duplicate=True),
        Input({"type": "close-error-btn", "index": ALL}, "n_clicks"),
        State("error-store", "data"),
        prevent_initial_call=True,
    )
    def close_error_notification(n_clicks_list, errors):
        """Close error notifications by removing from store."""
        from app_callbacks import get_error_store

        if not n_clicks_list or not any(n_clicks_list):
            return errors

        if not ctx.triggered:
            return errors

        # Get the index of the clicked button
        triggered_prop = ctx.triggered[0]["prop_id"]
        if triggered_prop == ".":
            return errors

        try:
            import json

            id_dict_str = triggered_prop.split(".n_clicks")[0]
            id_dict = json.loads(id_dict_str)
            button_index = int(id_dict.get("index", 0))
        except (ValueError, IndexError, json.JSONDecodeError, AttributeError):
            return errors

        # Remove the error from the list
        if errors and 0 <= button_index < len(errors):
            errors = errors[:button_index] + errors[button_index + 1 :]

        return errors
