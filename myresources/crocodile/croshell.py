

"""Crocodile Shell
"""

import argparse
import os
import random
from rich import pretty, inspect, progress, traceback, print
from rich.text import Text
from rich.console import Console

from crocodile.core import *
from crocodile.file_management import *
from crocodile.meta import *
# import crocodile.environment as env
from crocodile.matplotlib_management import *
import crocodile.toolbox as tb
import numpy as np
import pandas as pd
import platform


console = Console()
pretty.install()

_ = f"Python {platform.python_version()} in VE `{os.getenv('VIRTUAL_ENV')}` On {platform.system()}."
_ = Text(_); _.stylize("bold blue")
console.rule(_, style="bold red", align="center")
# link to tutorial or github
_ = Text(f"Crocodile Shell")
_.stylize("#93e6c7 on #093006")
print(_, __import__('crocodile').__version__)
print("Made with 🐍 | Built with ❤️")

tb.D.set_numpy_display()
tb.D.set_pandas_display()
D = Display; L = List; E = Experimental; S = Struct

__ = P(__file__).parent.joinpath("art").search().sample(size=1)[0]
if platform.system() == "Windows": print(__.read_text())
else:
    try:
        surprise = random.choice([True, True, True, True, False])  # classic art (True) or boxes (False)
        if surprise:
            from crocodile.msc.ascii_art import get_art
            artlib = random.choice(['boxes', '/usr/games/cowsay'])
            __ = P.tmpfile("croco_art", folder="tmp_arts")
            # print(f"{artlib=}, file={__}")
            get_art("crocodile", calliagraphy=None, artlib=artlib, file=__, verbose=False)
        os.system(f"cat {__} | /usr/games/lolcat")  # full path since lolcat might not be in PATH.
    except: print(__.read_text())
print("\n\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")
    args = parser.parse_args()
    # tb.Struct(args.__dict__).print(as_config=True)
    print(args.cmd)
    exec(args.cmd, globals())


if __name__ == "__main__":
    build_parser()
