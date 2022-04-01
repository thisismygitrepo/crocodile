

"""
Crocodile Philosophy:
Make Python even friendlier, by making available the common functionality for everyday use, e.g., path management, file management
At the risk of vandalizing the concept, Crocodile is about making Python more MATLAB-like, in that more libraries are loaded up at
start time than mere basic arithmetic, but just enought to make it more useful for everyday errands. Thus, the terseness of Crocodile makes Python REPL a proper shell
In implementation, the focus is on ease of use, not efficiency.
"""

from crocodile.toolbox import *
from crocodile import __version__

croc = P(__file__).parent.joinpath("art").search().sample()[0].read_text()
print(f"Crocodile Shell {__version__}")
print(f"Made with ❤️")
print(croc)

