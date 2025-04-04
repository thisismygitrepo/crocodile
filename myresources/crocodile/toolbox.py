
"""
A collection of classes extending the functionality of Python's builtins.
author: Alex Al-Saffar
email: programmer@usa.com

This is `export file` where one can dictate what will be exposed with toolbox
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from crocodile.meta import generate_readme, Terminal, Log, Scheduler, SSH
import logging
import subprocess
import sys
import time
from crocodile.file_management import P, Read, Compression, Cache, encrypt, decrypt, modify_text
from crocodile.core import List, Base, Struct, Save, Display
from crocodile.core import str2timedelta, timestamp, randstr, validate_name, install_n_import

import datetime

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _ = np, pd, px, plt, go
except ImportError:
    pass

_ = str2timedelta, timestamp, randstr, validate_name, install_n_import
__ = P, Read, Compression, Cache, encrypt, decrypt, modify_text, datetime
___ = generate_readme, Terminal, Log, Scheduler, SSH
____ = logging, subprocess, sys, time

Path = P
L = List
S = Struct
D = Display
T = Terminal
tmp = P.tmp

_1 = Base, timestamp, Save, Terminal, List, Struct, Display, P, Read, Compression
# from crocodile.core import datetime, dt, os, sys, string, random, np, copy, dill


# _ = False
# if _:
#     from crocodile import matplotlib_management as _pm
#     from crocodile.matplotlib_management import plt, enum, FigureManager
#
#     Artist, FigurePolicy, ImShow, FigureSave = _pm.Artist, _pm.FigurePolicy, _pm.ImShow, _pm.FigureSave
#     VisibilityViewer, VisibilityViewerAuto = _pm.VisibilityViewer, _pm.VisibilityViewerAuto
#
#     _ = plt, enum, FigureManager, FigurePolicy, ImShow, FigureSave, VisibilityViewer, VisibilityViewerAuto, Artist
#
#     import pandas as pd
#     _ = pd


if __name__ == '__main__':
    # get_parser()
    pass
