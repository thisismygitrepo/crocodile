
"""
A collection of classes extending the functionality of Python's builtins.
author: Alex Al-Saffar
email: programmer@usa.com

This is `export file` while one can dictate what will be exposed with toolbox
"""

import importlib
import inspect

from crocodile import core
from crocodile.core import datetime, dt, os, string, random, np, pd, copy, dill

from crocodile import file_management as _fm
from crocodile.file_management import re, typing, sys, shutil, glob, tempfile  # Path

from crocodile import plot_management as _pm
from crocodile.plot_management import plt, enum, FigureManager

from crocodile import meta
from crocodile.meta import logging, subprocess, time


Base, timestamp, randstr = core.Base, core.timestamp, core.randstr
Struct, DisplayData, str2timedelta, Save, List = core.Struct, core.DisplayData, core.str2timedelta, core.Save, core.List

P, Read, Compression = _fm.P, _fm.Read, _fm.Compression
Fridge, MemoryDB = _fm.Fridge, _fm.MemoryDB

Artist, FigurePolicy, ImShow, SaveType = _pm.Artist, _pm.FigurePolicy, _pm.ImShow, _pm.SaveType
VisibilityViewer, VisibilityViewerAuto = _pm.VisibilityViewer, _pm.VisibilityViewerAuto

Cycle, Experimental, Terminal, Manipulator = meta.Cycle, meta.Experimental, meta.Terminal, meta.Manipulator
batcher, batcherv2, Log, accelerate, Null = meta.batcher, meta.batcherv2, meta.Log, meta.accelerate, meta.Null
Scheduler = meta.Scheduler

L = List
tmp = P.tmp
E = Experimental

_ = Base, timestamp, Save, Terminal, List, Struct, DisplayData
_ = datetime, dt, os, np, pd, copy, random, dill

_ = P, Read, Compression, re, typing, string, sys, shutil, glob, tempfile  # , Path
_ = Experimental, Cycle, batcher, batcherv2, accelerate
_ = plt, enum, FigureManager, FigurePolicy, ImShow, SaveType, VisibilityViewer, VisibilityViewerAuto, Artist
_ = logging, subprocess, time


def reload(verbose=True):
    tb = sys.modules["crocodile.toolbox"]
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
