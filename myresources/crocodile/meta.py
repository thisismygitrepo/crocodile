"""
This is a module for handling meta operations like logging, terminal operations, SSH, etc.

"""

# Import from meta_helpers.meta1
from crocodile.meta_helpers.meta1 import (
    Scout,
    Log,
    STD,
    Response
)

# Import from meta_helpers.meta2
from crocodile.meta_helpers.meta2 import (
    Terminal,
    SHELLS,
    CONSOLE,
    MACHINE
)

# Import from meta_helpers.meta3
from crocodile.meta_helpers.meta3 import SSH

from crocodile.meta_helpers.meta4 import (
    Scheduler
)

# Import from meta_helpers.meta5
from crocodile.meta_helpers.meta5 import (
    generate_readme,
    RepeatUntilNoException,
    show_globals
)

# Define public API
__all__ = [
    # From meta1
    'Scout',
    'Log',
    'STD',
    'Response',

    # From meta2
    'Terminal',
    'SHELLS',
    'CONSOLE',
    'MACHINE',

    # From meta3
    'SSH',

    # From meta4
    'Scheduler',

    # From meta5
    'generate_readme',
    'RepeatUntilNoException',
    'show_globals'
]


if __name__ == '__main__':
    pass
