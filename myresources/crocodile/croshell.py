

"""
"""

import argparse
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


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")
    args = parser.parse_args()
    # tb.Struct(args.__dict__).print(as_config=True)
    print(args.cmd)
    exec(args.cmd, globals())


if __name__ == "__main__":
    build_parser()
