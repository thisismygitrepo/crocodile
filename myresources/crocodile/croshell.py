"""Crocodile Shell
"""

# import argparse
import os
import random
from rich import pretty, inspect, progress, traceback, print as pprint
from rich.text import Text
from rich.console import Console

import crocodile.toolbox as tb
import crocodile
from crocodile.core import *  # type: ignore # pylint: disable=W0401,W0614 # noqa: F403,F401
from crocodile.file_management import P
import crocodile.core as core
from crocodile.file_management import *  # type: ignore # noqa: F403,F401 # pylint: disable=W0401,W0614
from crocodile.meta import *  # type: ignore # noqa: F403,F401 # pylint: disable=W0401,W0614
# import crocodile.environment as env
from crocodile.matplotlib_management import *  # noqa: F403,F401 # type: ignore # pylint: disable=W0401,W0614

import numpy as np
import pandas as pd
import platform

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _ = px, go
except ImportError:
    print("⚠️  IMPORT ERROR: Plotly is not installed ⚠️")


def print_header():
    console = Console()
    pretty.install()

    _header = f"🐍 Python {platform.python_version()} in VE `{os.getenv('VIRTUAL_ENV')}` On {platform.system()} 🐍"
    _header = Text(_header)
    _header.stylize("bold blue")
    console.rule(_header, style="bold red", align="center")

    # link to tutorial or github
    _ = Text(f"✨ 🐊 Crocodile Shell {crocodile.__version__} ✨" + " Made with 🐍 | Built with ❤️\n")
    _.stylize("#05f8fc on #293536")
    console.print(_)


tb.D.set_numpy_display()
tb.D.set_pandas_display()
D = core.Display
L = core.List
S = core.Struct
_ = D, L, S, inspect, progress, pprint, traceback, pd, np


def print_logo(logo: str):
    from crocodile.msc.ascii_art import font_box_color, character_color, character_or_box_color
    if platform.system() == "Windows":
        _1x = P.home().joinpath(r"AppData/Roaming/npm/figlet").exists()
        _2x = P.home().joinpath(r"AppData/Roaming/npm/lolcatjs").exists()
        _3x = P.home().joinpath(r"AppData/Local/Microsoft/WindowsApps/boxes.exe").exists()
        if _1x and _2x and _3x:
            if random.choice([True, True, False]): font_box_color(logo)
            else: character_color(logo)
        else:
            print("\n" + "🚫 " + "-" * 70 + " 🚫")
            print("🔍 Missing ASCII art dependencies. Install with: iwr bit.ly/cfgasciiartwindows | iex")
            print("🚫 " + "-" * 70 + " 🚫\n")
            _default_art = P(__file__).parent.joinpath("art").search().sample(size=1)[0]
            print(_default_art.read_text())
    else:
        def is_executable_in_path(executable_name: str) -> bool:
            path_dirs = os.environ['PATH'].split(os.pathsep)
            for path_dir in path_dirs:
                path_to_executable = os.path.join(path_dir, executable_name)
                if os.path.isfile(path_to_executable) and os.access(path_to_executable, os.X_OK): return True
            return False
        # _x1 = P.home().joinpath(".nix-profile/bin/cowsay").exists()  # P(r"/usr/games/cowsay").exists()
        # _x2 = P.home().joinpath(".nix-profile/bin/lolcat").exists()  # P(r"/usr/games/lolcat").exists()
        # _x3 = P.home().joinpath(".nix-profile/bin/boxes").exists()  # P(r"/usr/bin/boxes").exists()
        # _x4 = P.home().joinpath(".nix-profile/bin/figlet").exists()  # P(r"/usr/bin/figlet").exists()
        _x1 = is_executable_in_path("cowsay")
        _x2 = is_executable_in_path("lolcat")
        _x3 = is_executable_in_path("boxes")
        _x4 = is_executable_in_path("figlet")

        if _x1 and _x2 and _x3 and _x4:
            _dynamic_art = random.choice([True, True, True, True, False])  # classic art (True) or boxes (False)
            if _dynamic_art: character_or_box_color(logo)
            else: print(P(__file__).parent.joinpath("art").search().sample(size=1).list[0].read_text())
        else:
            print("\n" + "🚫 " + "-" * 70 + " 🚫")
            print("🔍 Missing ASCII art dependencies. Install with: curl bit.ly/cfgasciiartlinux -L | sudo bash")
            print("🚫 " + "-" * 70 + " 🚫\n")
            _default_art = P(__file__).parent.joinpath("art").search().sample(size=1)[0]
            print(_default_art.read_text())


if __name__ == "__main__":
    print_header()
    print_logo(logo="crocodile")
