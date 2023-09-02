
"""
A collection of classes extending the functionality of Python's builtins.
author: Alex Al-Saffar
email: programmer@usa.com

This is `export file` where one can dictate what will be exposed with toolbox
"""

from crocodile.meta import Experimental, Terminal, Log, Null, Scheduler, SSH
from crocodile.meta import logging, subprocess, sys, time
# from crocodile import run
from crocodile.file_management import P, Read, Compression, Cache, encrypt, decrypt, modify_text, datetime
from crocodile.core import List, Base, Struct, Display, Save
from crocodile.core import str2timedelta, timestamp, randstr, validate_name, install_n_import

_ = str2timedelta, timestamp, randstr, validate_name, install_n_import
__ = P, Read, Compression, Cache, encrypt, decrypt, modify_text, datetime
___ = Experimental, Terminal, Log, Null, Scheduler, SSH
____ = logging, subprocess, sys, time

Path = P
L = List
S = Struct
D = Display
T = Terminal
E = Experimental
tmp = P.tmp

_1 = Base, timestamp, Save, Terminal, List, Struct, Display, P, Read, Compression, Experimental
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
