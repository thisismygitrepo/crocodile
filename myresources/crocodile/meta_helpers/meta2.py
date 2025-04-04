from crocodile.core import Save, install_n_import
from crocodile.file_management import P, OPLike, PLike
import time
import platform
import subprocess
import sys
import os
from typing import Union, Any, Optional, Callable, Literal
from typing_extensions import TypeAlias
from crocodile.meta_helpers.meta1 import Response


SHELLS: TypeAlias = Literal["default", "cmd", "powershell", "pwsh", "bash"]  # pwsh.exe is PowerShell (community) and powershell.exe is Windows Powershell (msft)
CONSOLE: TypeAlias = Literal["wt", "cmd"]
MACHINE: TypeAlias = Literal["Windows", "Linux", "Darwin"]


class Terminal:
    def __init__(self, stdout: Optional[int] = subprocess.PIPE, stderr: Optional[int] = subprocess.PIPE, stdin: Optional[int] = subprocess.PIPE, elevated: bool = False):
        self.machine: str = platform.system()
        self.elevated: bool = elevated
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
    # def set_std_system(self): self.stdout = sys.stdout; self.stderr = sys.stderr; self.stdin = sys.stdin
    def set_std_pipe(self):
        self.stdout = subprocess.PIPE
        self.stderr = subprocess.PIPE
        self.stdin = subprocess.PIPE
    def set_std_null(self):
        self.stdout, self.stderr, self.stdin = subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL  # Equivalent to `echo 'foo' &> /dev/null`
    def run(self, *cmds: str, shell: Optional[SHELLS] = "default", check: bool = False, ip: Optional[str] = None) -> Response:  # Runs SYSTEM commands like subprocess.run
        """Blocking operation. Thus, if you start a shell via this method, it will run in the main and won't stop until you exit manually IF stdin is set to sys.stdin, otherwise it will run and close quickly. Other combinations of stdin, stdout can lead to funny behaviour like no output but accept input or opposite.
        * This method is short for: res = subprocess.run("powershell command", capture_output=True, shell=True, text=True) and unlike os.system(cmd), subprocess.run(cmd) gives much more control over the output and input.
        * `shell=True` loads up the profile of the shell called so more specific commands can be run. Importantly, on Windows, the `start` command becomes availalbe and new windows can be launched.
        * `capture_output` prevents the stdout to redirect to the stdout of the script automatically, instead it will be stored in the Response object returned. # `capture_output=True` same as `stdout=subprocess.PIPE, stderr=subprocess.PIPE`"""
        my_list = list(cmds)  # `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job. `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task. `suprocess.call` is an archaic command for pre-Python-3.5.
        if self.machine == "Windows" and shell in {"powershell", "pwsh"}: my_list = [shell, "-Command"] + my_list  # alternatively, one can run "cmd"
        if self.elevated is False or self.is_user_admin():
            resp: subprocess.CompletedProcess[str] = subprocess.run(my_list, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=check, input=ip)
        else:
            resp = __import__("ctypes").windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        return Response.from_completed_process(resp)
    def run_script(self, script: str, shell: SHELLS = "default", verbose: bool = False):
        if self.machine == "Linux": script = "#!/bin/bash" + "\n" + script  # `source` is only available in bash.
        script_file = P.tmpfile(name="tmp_shell_script", suffix=".ps1" if self.machine == "Windows" else ".sh", folder="tmp_scripts").write_text(script, newline={"Windows": None, "Linux": "\n"}[self.machine])
        if shell == "default":
            if self.machine == "Windows":
                start_cmd = "powershell"  # default shell on Windows is cmd which is not very useful. (./source is not available)
                full_command: Union[list[str], str] = [start_cmd, str(script_file)]  # shell=True will cause this to be a string anyway (with space separation)
            else:
                start_cmd  = "bash"
                full_command = f"{start_cmd} {script_file}"  # full_command = [start_cmd, str(script_file)]
        else: full_command = f"{shell} {script_file}"  # full_command = [shell, str(tmp_file)]
        if verbose:
            from machineconfig.utils.utils import print_code
            print_code(code=script, lexer="shell", desc="Script to be executed:")
            import rich.progress as pb
            with pb.Progress(transient=True) as progress:
                _task = progress.add_task(f"Running Script @ {script_file}", total=None)
                resp = subprocess.run(full_command, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=False)
        else: resp = subprocess.run(full_command, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=False)
        return Response.from_completed_process(resp)
    def run_async(self, *cmds: str, new_window: bool = True, shell: Optional[str] = None, terminal: Optional[str] = None):  # Runs SYSTEM commands like subprocess.Popen
        """Opens a new terminal, and let it run asynchronously. Maintaining an ongoing conversation with another process is very hard. It is adviseable to run all
        commands in one go without interaction with an ongoing channel. Use this only for the purpose of producing a different window and humanly interact with it. Reference: https://stackoverflow.com/questions/54060274/dynamic-communication-between-main-and-subprocess-in-python & https://www.youtube.com/watch?v=IynV6Y80vws and https://www.oreilly.com/library/view/windows-powershell-cookbook/9781449359195/ch01.html"""
        if terminal is None: terminal = ""  # this means that cmd is the default console. alternative is "wt"
        if shell is None: shell = "" if self.machine == "Windows" else ""  # other options are "powershell" and "cmd". # if terminal is wt, then it will pick powershell by default anyway.
        new_window_cmd = "start" if new_window is True else ""  # start is alias for Start-Process which launches a new window.  adding `start` to the begining of the command results in launching a new console that will not inherit from the console python was launched from e.g. conda
        extra, my_list = ("-Command" if shell in {"powershell", "pwsh"} and len(cmds) else ""), list(cmds)
        if self.machine == "Windows": my_list = [new_window_cmd, terminal, shell, extra] + my_list  # having a list is equivalent to: start "ipython -i file.py". Thus, arguments of ipython go to ipython, not start.
        print(f"""🚀 [ASYNC EXECUTION] About to run command: {my_list}""")
        print("Meta.Terminal.run_async: Subprocess command: ", my_list := [item for item in my_list if item != ""])
        return subprocess.Popen(my_list, stdin=subprocess.PIPE, shell=True)  # stdout=self.stdout, stderr=self.stderr, stdin=self.stdin. # returns Popen object, not so useful for communcation with an opened terminal
    def run_py(self, script: str, wdir: OPLike = None, interactive: bool = True, ipython: bool = True, shell: Optional[str] = None, terminal: str = "", new_window: bool = True, header: bool = True):  # async run, since sync run is meaningless.
        script = (Terminal.get_header(wdir=wdir, toolbox=True) if header else "") + script + ("\nDisplayData.set_pandas_auto_width()\n" if terminal in {"wt", "powershell", "pwsh"} else "")
        py_script = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmp_scripts/terminal").write_text(f"""print(r'''{script}''')""" + "\n" + script)
        print(f"""🚀 [ASYNC PYTHON SCRIPT] Script URI:
   {py_script.absolute().as_uri()}""")
        print("Script to be executed asyncronously: ", py_script.absolute().as_uri())
        shell_script = f"""
{f'cd {wdir}' if wdir is not None else ''}
{'ipython' if ipython else 'python'} {'-i' if interactive else ''} {py_script}
"""
        shell_script = P.tmpfile(name="tmp_shell_script", suffix=".sh" if self.machine == "Linux" else ".ps1", folder="tmp_scripts/shell").write_text(shell_script)
        if shell is None and self.machine == "Windows": shell = "pwsh"
        window = "start" if new_window and self.machine == "Windows" else ""
        os.system(f"{window} {terminal} {shell} {shell_script}")
    @staticmethod
    def is_user_admin() -> bool:  # adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows"""
        if os.name == 'nt':
            try: return __import__("ctypes").windll.shell32.IsUserAnAdmin()
            except Exception:
                import traceback
                traceback.print_exc()
                print("Admin check failed, assuming not an admin.")
                return False
        else:
            return os.getuid() == 0  # Check for root on Posix
    @staticmethod
    def run_as_admin(file: PLike, params: Any, wait: bool = False):
        proce_info = install_n_import(library="win32com", package="pywin32", fromlist=["shell.shell.ShellExecuteEx"]).shell.shell.ShellExecuteEx(lpVerb='runas', lpFile=file, lpParameters=params)
        # TODO update PATH for this to take effect immediately.
        if wait: time.sleep(1)
        return proce_info
    @staticmethod
    def pickle_to_new_session(obj: Any, cmd: str = ""):
        return Terminal().run_py(f"""path = P(r'{Save.pickle(obj=obj, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}')\n obj = path.readit()\npath.delete(sure=True, verbose=False)\n {cmd}""")
    @staticmethod
    def import_to_new_session(func: Union[None, Callable[[Any], Any]] = None, cmd: str = "", header: bool = True, interactive: bool = True, ipython: bool = True, run: bool = False, **kwargs: Any):
        load_kwargs_string = f"""kwargs = P(r'{Save.pickle(obj=kwargs, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}').readit()\nkwargs.print()\n""" if kwargs else "\n"
        run_string = "\nobj(**loaded_kwargs)\n" if run else "\n"
        if callable(func) and func.__name__ != func.__qualname__:  # it is a method of a class, must be instantiated first.
            tmp = sys.modules['__main__'].__file__  # type: ignore  # pylint: disable=E1101
            assert isinstance(tmp, str), f"Cannot import a function from a module that is not a file. The module is: {tmp}"
            module = P(tmp).rel2cwd().stem if (module := func.__module__) == "__main__" else module
            load_func_string = f"import {module} as m\ninst=m.{func.__qualname__.split('.')[0]}()\nobj = inst.{func.__name__}"
        elif callable(func) and hasattr(func, "__code__"):  # it is a standalone function...
            module = P(func.__code__.co_filename)  # module = func.__module__  # fails if the function comes from main as it returns __main__.
            load_func_string = f"sys.path.insert(0, r'{module.parent}')\nimport {module.stem} as m\nobj=m.{func.__name__}"
        else: load_func_string = f"""obj = P(r'{Save.pickle(obj=func, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}').readit()"""
        return Terminal().run_py(load_func_string + load_kwargs_string + f"\n{cmd}\n" + run_string, header=header, interactive=interactive, ipython=ipython)  # Terminal().run_async("python", "-c", load_func_string + f"\n{cmd}\n{load_kwargs_string}\n")
    @staticmethod
    def replicate_session(cmd: str = ""):
        import dill
        file = P.tmpfile(suffix=".pkl")
        script = f"""
path = P(r'{file}')
import dill
sess = dill.load_session(str(path))
path.delete(sure=True, verbose=False)
{cmd}"""
        dill.dump_session(file, main=sys.modules[__name__])
        Terminal().run_py(script=script)
    @staticmethod
    def get_header(wdir: OPLike, toolbox: bool): return f"""
# >> Code prepended
{"from crocodile.toolbox import *" if toolbox else "# No toolbox import."}
{'''sys.path.insert(0, r'{wdir}') ''' if wdir is not None else "# No path insertion."}
# >> End of header, start of script passed
"""
