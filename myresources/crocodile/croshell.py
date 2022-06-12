

"""
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
