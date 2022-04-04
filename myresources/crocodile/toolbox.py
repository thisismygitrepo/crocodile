
"""
A collection of classes extending the functionality of Python's builtins.
author: Alex Al-Saffar
email: programmer@usa.com

This is `export file` while one can dictate what will be exposed with toolbox
"""

from crocodile import core
from crocodile import meta
from crocodile import file_management as _fm


# CORE =====================================
Base, timestamp, randstr, validate_name = core.Base, core.timestamp, core.randstr, core.validate_name
Struct, Display, str2timedelta, Save, List = core.Struct, core.Display, core.str2timedelta, core.Save, core.List
install_n_import = core.install_n_import

# File Management ==========================
P, Read, Compression = _fm.P, _fm.Read, _fm.Compression
Fridge, MemoryDB = _fm.Fridge, _fm.MemoryDB
encrypt, decrypt = _fm.encrypt, _fm.decrypt

# META ====================================
Experimental, Terminal, Manipulator = meta.Experimental, meta.Terminal, meta.Manipulator
Log, Null = meta.Log, meta.Null
Scheduler, SSH = meta.Scheduler, meta.SSH


L = List
tmp = P.tmp
E = Experimental
M = Manipulator
D = Display
# _ = P, Read, Compression
# _ = Experimental


_ = Base, timestamp, Save, Terminal, List, Struct, Display
# from crocodile.core import datetime, dt, os, sys, string, random, np, copy, dill
# _ = datetime, dt, os, sys, np, copy, random, dill
import logging, subprocess, sys


_ = False
if _:
    from crocodile import matplotlib_management as _pm
    from crocodile.matplotlib_management import plt, enum, FigureManager

    Artist, FigurePolicy, ImShow, SaveType = _pm.Artist, _pm.FigurePolicy, _pm.ImShow, _pm.SaveType
    VisibilityViewer, VisibilityViewerAuto = _pm.VisibilityViewer, _pm.VisibilityViewerAuto

    _ = plt, enum, FigureManager, FigurePolicy, ImShow, SaveType, VisibilityViewer, VisibilityViewerAuto, Artist

    import pandas as pd
    _ = pd


def reload(verbose=True):
    import inspect
    import importlib
    tb = __import__("sys").modules["crocodile.toolbox"]
    for val in tb.__dict__.values():
        if inspect.ismodule(val):
            importlib.reload(val)
            if verbose: print(f"Reloading {val}")
    # importlib.reload(tb)
    # # import runpy
    # # mod = runpy.run_path(__file__)
    # # # globals().update(mod)
    # # print(f"{__file__=},  {__name__=}")
    # # current_module = sys.modules["crocodile.toolbox"]
    # # current_module.__dict__.update(mod)
    return tb


def get_parser():
    # TODO: has no purpose so far.
    import argparse
    parser = argparse.ArgumentParser(description="Crocodile toolbox parser")
    parser.add_argument()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get_parser()
    pass
