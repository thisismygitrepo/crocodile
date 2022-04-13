
"""
A collection of classes extending the functionality of Python's builtins.
author: Alex Al-Saffar
email: programmer@usa.com

This is `export file` where one can dictate what will be exposed with toolbox
"""

from crocodile import core
from crocodile import meta
from crocodile import file_management as _fm

# CORE =====================================
str2timedelta, timestamp, randstr, validate_name, install_n_import = core.str2timedelta, core.timestamp, core.randstr, core.validate_name, core.install_n_import
Base, Struct, Display, Save, List = core.Base, core.Struct, core.Display, core.Save, core.List
# File Management ==========================
P, Read, Compression, Fridge, MemoryDB, encrypt, decrypt = _fm.P, _fm.Read, _fm.Compression, _fm.Fridge, _fm.MemoryDB, _fm.encrypt, _fm.decrypt
# META ====================================
Experimental, Terminal, Log, Null, Scheduler, SSH = meta.Experimental, meta.Terminal, meta.Log, meta.Null, meta.Scheduler, meta.SSH

L = List
tmp = P.tmp
E = Experimental
D = Display


_ = Base, timestamp, Save, Terminal, List, Struct, Display, P, Read, Compression, Experimental
# from crocodile.core import datetime, dt, os, sys, string, random, np, copy, dill
logging, subprocess, sys = meta.logging, meta.subprocess, meta.sys
datetime = _fm.datetime

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
