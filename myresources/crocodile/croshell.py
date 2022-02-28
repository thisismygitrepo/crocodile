

from crocodile.toolbox import *
from crocodile import __version__

croc = P(__file__).parent.joinpath("art/happy_croco").read_text()
print(f"Crocodile Shell {__version__}")
print(f"Made with ❤️")
print(croc)
