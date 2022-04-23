

"""
Crocodile Philosophy:
Make Python even friendlier, by making available the common functionality for everyday use, e.g., path management, file management
At the risk of vandalizing the concept, Crocodile is about making Python more MATLAB-like, in that more libraries are loaded up at
start time than mere basic arithmetic, but just enought to make it more useful for everyday errands. Thus, the terseness of Crocodile makes Python REPL a proper shell
In implementation, the focus is on ease of use, not efficiency.
"""

from crocodile.core import *
from crocodile.file_management import *
from crocodile.meta import *
# import crocodile.environment as env
from crocodile.matplotlib_management import *
import crocodile.toolbox as tb
D = Display; L = List; E = Experimental; S = Struct

import numpy as np
import pandas as pd


print(f"Crocodile Shell {__import__('crocodile').__version__}")
# link to tutorial or github
print(f"Made with ❤️")
print(P(__file__).parent.joinpath("art").search().sample(size=1)[0].read_text())
