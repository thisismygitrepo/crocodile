
"""P
"""


from crocodile.file_management import P
from crocodile.core import randstr
from crocodile.meta import Terminal

# import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import dash_bootstrap_components as dbc

import dash
from dash import Input, Output, State, callback_context as ctx
from dash import dcc
from dash import html
import dash_daq as daq
from typing import Any, Optional

from types import SimpleNamespace
import sys

pio.renderers.default = "browser"
tm = Terminal()
_ = Input, Output, State, dcc, html, daq, ctx, sys, dbc
__ = go, px, make_subplots


CONFIG = SimpleNamespace(displayModeBar=True,  # always visible.
                         staticPlot=False,
                         scrollZoom=False,
                         doubleClick="reset",
                         showTips=True,
                         toImageButtonOptions={
                             'format': 'png',  # one of png, svg, jpeg, webp
                             'filename': 'custom_image',
                             'height': 1500,  # None means use currently rendered size.
                             'width': 1500,
                             'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
                         },
                         modeBarButtonsToAdd=['drawline',
                                              'drawopenpath',
                                              'drawclosedpath',
                                              'drawcircle',
                                              'drawrect',
                                              'eraseshape']
                         )


def get_random_port() -> int:
    import random
    port = random.randint(1024, 49151)
    def is_port_in_use(port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    if is_port_in_use(port):
        return get_random_port()
    return port


class App:
    @staticmethod
    def run(app: Any, debug: bool = False, port: Optional[int] = None, start_browser: bool = True, lan_share: bool = True):
        host = "localhost"  # 0.0.0.0"  #  "127.0.0.1"
        if port is None: port__ = get_random_port()
        else: port__ = port
        # pynoinspection HTTP
        if start_browser:
            try: P(rf'http://{host}:{port__}/')()
            except Exception as ex:
                print(ex)
        app.run_server(debug=debug, port=port__, host="0.0.0.0" if lan_share else "localhost")  # , processes=2, threaded=False)

    @staticmethod
    def get_app(name: str = ""):
        _theme = {
            'dark': True,
            'detail': '#007439',
            'primary': '#00EA64',
            'secondary': '#6E6E6E',
        }
        return dash.Dash(name=name + randstr(), external_stylesheets=[r'https://codepen.io/chriddyp/pen/bWLwgP.css'])
        # [dbc.themes.DARKLY])
        #
    # @staticmethod
    # def run_async_decorator(func):  # Decorate functions with this to make them run_command asynchornously."""
    #     raise NotImplementedError


if __name__ == '__main__':
    pass
