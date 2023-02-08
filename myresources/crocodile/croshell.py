

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
if platform.system() == "Windows":
    _1x = P.home().joinpath(r"AppData/Roaming/npm/figlet").exists()
    _2x = P.home().joinpath(r"AppData/Roaming/npm/lolcatjs").exists()
    _3x = P.home().joinpath(r"AppData/Local/Microsoft/WindowsApps/boxes.exe").exists()

    if _1x and _2x and _3x:
        if random.choice([True, True, False]):
            from crocodile.msc.ascii_art import FIGJS_FONTS, BoxStyles
            font = random.choice(FIGJS_FONTS)
            print(f"{font}\n")
            box_style = random.choice(['whirly', 'xes', 'columns', 'parchment', 'scroll', 'scroll-akn', 'diamonds', 'headline', 'nuke', 'spring', 'stark1'])
            os.system(f'figlet -f "{font}" "crocodile" | boxes -d "{box_style}" | lolcatjs')  # | lolcat
        else:
            from crocodile.msc.ascii_art import ArtLib
            # from rgbprint import gradient_scroll, Color
            # gradient_scroll(ArtLib.cowsay("crocodile"), start_color=0x4BBEE3, end_color=Color.medium_violet_red, times=3)
            __ = tb.P.tmpfile("croco_art", folder="tmp_arts").write_text(ArtLib.cowsay("crocodile"))  # utf-8 encoding?
            os.system(f'type {__} | lolcatjs')  # | lolcat
    else:
        print(f"Missing ascii art dependencies. Install with: iwr bit.ly/cfgasciiartwindows | iex")
        print(__.read_text())
else:
    _x1 = P(r"/usr/games/cowsay").exists()
    _x2 = P(r"/usr/games/lolcat").exists()
    _x3 = P(r"/usr/bin/boxes").exists()
    _x4 = P(r"/usr/bin/figlet").exists()
    if _x1 and _x2 and _x3 and _x4:
        _dynamic_art = random.choice([True, True, True, True, False])  # classic art (True) or boxes (False)
        if _dynamic_art:
            from crocodile.msc.ascii_art import get_art
            __ = P.tmpfile("croco_art", folder="tmp_arts")
            get_art("crocodile", artlib=None, file=__, verbose=False)
            os.system(f"cat {__} | /usr/games/lolcat")  # full path since lolcat might not be in PATH.
        else: print(__.read_text())
    else:
        print(f"Missing ascii art dependencies. Install with: curl bit.ly/cfgasciiartlinux -L | sudo bash")
        print(__.read_text())
print("\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Generic Parser to launch a script in a separate window.")
    parser.add_argument("--cmd", "-c", dest="cmd", help="Python command.", default="")
    parser.add_argument("--file", "-f", dest="file", help="Python command.", default="")

    args = parser.parse_args()
    # tb.Struct(args.__dict__).print(as_config=True)
    if args.file:
        code = P(args.file).read_text()
        print(code)
        exec(code, globals())
    elif args.cmd:
        print(args.cmd)
        exec(args.cmd, globals())


if __name__ == "__main__":
    build_parser()
