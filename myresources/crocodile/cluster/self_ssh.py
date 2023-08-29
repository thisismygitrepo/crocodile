
# import crocodile.toolbox as tb


# class SelfSSH:
#     def __init__(self):
#         self._remote_machine = None
#         self.remote_env_cmd = ". activate_ve"
#     def run(self, cmd: str, desc: str = ""): return tb.Terminal().run(cmd, shell="powershell")
#     def run_py(self, cmd: str, verbose: bool = False): return None
#     def get_remote_machine(self): return ("Windows" if (self.run("$env:OS").op.rstrip("\n") == "Windows_NT" or self.run("echo %OS%").op == "Windows_NT") else "Linux") if self._remote_machine is None else self._remote_machine
#     def get_repr(self, which: str, add_machine: bool = False): return f"SelfSSH({which})"
#     def open_console(self, cmd: str = "", shell: str = "powershell"): return tb.Terminal().run_async("-i", new_window=True, shell=shell)
#     def copy_to_here(self, path: str, z: str): return None
