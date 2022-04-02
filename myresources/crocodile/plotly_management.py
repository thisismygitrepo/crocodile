
import crocodile.toolbox as tb
# import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import dash_daq as daq

from types import SimpleNamespace
import webbrowser


pio.renderers.default = "browser"
tm = tb.Terminal()
_ = Input, Output, State, dcc, html, daq
_ = go, px, make_subplots


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
                                              'eraseshape'
                                              ]
                         )


class App:
    @staticmethod
    def run(app, debug=False, random_port=True):
        """"Random ports prevent multile programs from crashing into each other."""
        host = "127.0.0.1"
        port = tb.randstr(lower=False, upper=False, length=4) if random_port else "8050"
        # pynoinspection HTTP
        webbrowser.open(rf'http://{host}:{port}/')
        app.run_server(debug=debug, port=port)  # , processes=2, threaded=False)

    @staticmethod
    def get_app(name=""): return dash.Dash(name=name+tb.randstr(), external_stylesheets=[r'https://codepen.io/chriddyp/pen/bWLwgP.css'])

    @staticmethod
    def run_async(func):
        if func.__name__ != func.__qualname__:  # it is a method of a class, must be instantiated first.
            cn = func.__qualname__.split(".")[0]  # class name is obtained.
            cmd = f"import {func.__module__} as m; inst=m.{cn}(); inst.{func.__name__}()"
        else:  # it is a standalone function.
            # module = func.__module__  # fails if the function comes from main as it returns __main__.
            module = tb.P(func.__code__.co_filename)
            tb.sys.path.insert(0, module.parent)  # potentially dangerous as it leads to perplexing behaviour.
            cmd = f"import {module.stem} as m; m.{func.__name__}()"
        return tm.run_async("python", "-c", cmd)

    @staticmethod
    def run_async_decorator(func):
        """Decorate functions with this to make them run_command asynchornously."""
        def get_async_version(): return App.run_async(func)
        return get_async_version


if __name__ == '__main__':
    pass
