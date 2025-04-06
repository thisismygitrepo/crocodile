"""
File Management Module

This module provides utilities for file management, compression, encryption, and caching.
"""

from crocodile.core_modules.core_1 import Save

# Import security-related functions from file1
from crocodile.file_management_helpers.file1 import (
    obscure, unobscure, hashpwd, pwd2key, encrypt, decrypt, unlock, modify_text
)

# Import compression class from file2
from crocodile.file_management_helpers.file2 import Compression, FILE_MODE, SHUTIL_FORMATS

# Import cache classes from file3
from crocodile.file_management_helpers.file3 import Cache, CacheV2, PrintFunc

# Import path utilities from file4
from crocodile.file_management_helpers.file4 import P, PLike, OPLike

# Import from file5 (assuming these are the key imports)
from crocodile.file_management_helpers.file5 import Read

# Define public API
__all__ = [
    # From core_1 - Save utilities
    'Save',

    # From file1 - Security functions
    'obscure', 'unobscure', 'hashpwd', 'pwd2key', 'encrypt', 'decrypt', 'unlock', 'modify_text',

    # From file2 - Compression utilities
    'Compression', 'FILE_MODE', 'SHUTIL_FORMATS',

    # From file3 - Cache classes
    'Cache', 'CacheV2', 'PrintFunc',

    # From file4 - Path utilities
    'P', 'PLike', 'OPLike',

    # From file5 - Other utilities
    'Read',
]


if __name__ == '__main__':
    # print('hi from file_managements')
    pass

# %%
