"""
A collection of classes extending the functionality of Python's builtins.

email programmer@usa.com
"""

from crocodile import core
from crocodile.core import datetime, dt, os, string, random, np, pd, copy, dill

from crocodile import file_management as _fm
from crocodile.file_management import re, typing, sys, shutil, glob, tempfile  # Path

from crocodile import plot_management as _pm
from crocodile.plot_management import plt, enum, FigureManager

from crocodile import meta
from crocodile.meta import logging


Base, get_time_stamp, get_random_string = core.Base, core.get_time_stamp, core.get_random_string
Struct, DisplayData, str2timedelta, Save, List = core.Struct, core.DisplayData, core.str2timedelta, core.Save, core.List

P, Read, Compression = _fm.P, _fm.Read, _fm.Compression

Artist, FigurePolicy, ImShow, SaveType = _pm.Artist, _pm.FigurePolicy, _pm.ImShow, _pm.SaveType
VisibilityViewer, VisibilityViewerAuto = _pm.VisibilityViewer, _pm.VisibilityViewerAuto

Cycle, Experimental, Terminal, Manipulator = meta.Cycle, meta.Experimental, meta.Terminal, meta.Manipulator
batcher, batcherv2, Log, accelerate, Null = meta.batcher, meta.batcherv2, meta.Log, meta.accelerate, meta.Null


L = List
tmp = P.tmp
E = Experimental

_ = Base, get_time_stamp, Save, Terminal, List, Struct, DisplayData
_ = datetime, dt, os, np, pd, copy

_ = P, Read, Compression, re, typing, string, sys, shutil, glob  # , Path
_ = Experimental, Cycle, batcher, batcherv2, accelerate
_ = plt, enum, FigureManager, FigurePolicy, ImShow, SaveType, VisibilityViewer, VisibilityViewerAuto, Artist


def reload():
    import importlib
    return importlib.__import__(__file__)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
