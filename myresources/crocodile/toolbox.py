"""
A collection of classes extending the functionality of Python's builtins.
email programmer@usa.com
"""

from crocodile.core import Base, get_time_stamp, Save, Terminal, List, Struct, DisplayData
from crocodile.core import datetime, dt, os, np, pd, copy

from crocodile.file_management import P, Read, Compression
from crocodile.file_management import re, typing, string, sys, shutil, glob, Path

from crocodile.plot_management import plt, enum, FigureManager, FigurePolicy, ImShow, SaveType
from crocodile.plot_management import VisibilityViewer, VisibilityViewerAuto, Artist
from crocodile.meta import Cycle, Experimental, batcher, batcherv2

L = List
tmp = P.tmp
_ = Base, get_time_stamp, Save, Terminal, List, Struct, DisplayData
_ = datetime, dt, os, np, pd, copy

_ = P, Read, Compression, re, typing, string, sys, shutil, glob, Path
_ = Experimental, Cycle, batcher, batcherv2
_ = plt, enum, FigureManager, FigurePolicy, ImShow, SaveType, VisibilityViewer, VisibilityViewerAuto, Artist


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
