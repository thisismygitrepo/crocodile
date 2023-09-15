
"""SelfSSh
"""

import crocodile.toolbox as tb
from crocodile.file_management import PLike
from typing import Optional, Any


class SelfSSH:
    def __init__(self):
        self.hostname = "This Machine"
        self._remote_machine = None
        self.remote_env_cmd = ". activate_ve"
        self.sftp: Any
    def run(self, cmd: str, desc: str = "", verbose: bool = False):
        _ = desc, verbose
        return tb.Terminal().run(cmd, shell="powershell")
    def run_py(self, cmd: str, verbose: bool = False, desc: str = ''):
        _ = verbose, cmd, desc
        return None
    def get_ssh_conn_str(self): return "ssh localhost"
    def get_remote_machine(self): return ("Windows" if (self.run("$env:OS").op.rstrip("\n") == "Windows_NT" or self.run("echo %OS%").op == "Windows_NT") else "Linux") if self._remote_machine is None else self._remote_machine
    def get_repr(self, which: str, add_machine: bool = False):
        _ = add_machine
        return f"SelfSSH({which})"
    def open_console(self, cmd: str = "", shell: str = "powershell"):
        _ = cmd, shell
        return tb.Terminal().run_async("-i", new_window=True, shell=shell)
    def copy_to_here(self, source: PLike = '', target: Optional[str] = '', z: bool = True, r: bool = True, desc: str = '', overwrite: bool = False):
        _ = source, target, z, r, desc, overwrite
        return None
    def copy_from_here(self, source: PLike = '', target: Optional[str] = '', z: bool = True, r: bool = True, desc: str = '', overwrite: bool = False):
        _ = source, target, z, r, desc, overwrite
        return None
