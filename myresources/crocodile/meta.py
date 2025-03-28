"""
This is a module for handling meta operations like logging, terminal operations, SSH, etc.

"""

from crocodile.core import randstr, str2timedelta, Save, install_n_import, List, Struct
from crocodile.file_management import P, datetime, OPLike, PLike
import time
import datetime as dtm
import logging
import subprocess
import sys
import os
from typing import Union, Any, Optional, Callable, TextIO, BinaryIO, IO, TypeAlias, Literal, TypeVar, ParamSpec
from dataclasses import dataclass


_ = IO, TextIO
SHELLS: TypeAlias = Literal["default", "cmd", "powershell", "pwsh", "bash"]  # pwsh.exe is PowerShell (community) and powershell.exe is Windows Powershell (msft)
CONSOLE: TypeAlias = Literal["wt", "cmd"]
MACHINE: TypeAlias = Literal["Windows", "Linux", "Darwin"]
T = TypeVar('T')
PS = ParamSpec('PS')


@dataclass
class Scout:
    source_full: P
    source_rel2home: P
    exists: bool
    is_dir: bool
    files: Optional[List[P]]


class Log(logging.Logger):  #
    def __init__(self, dialect: Literal["colorlog", "logging", "coloredlogs"] = "colorlog",
                 name: Optional[str] = None, file: bool = False, file_path: OPLike = None, stream: bool = True, fmt: Optional[str] = None, sep: str = " | ",
                 s_level: int = logging.DEBUG, f_level: int = logging.DEBUG, l_level: int = logging.DEBUG, verbose: bool = False,
                 log_colors: Optional[dict[str, str]] = None):
        if name is None:
            name = randstr(noun=True)
            print(f"""🔔 Logger name not provided.
   Please provide a descriptive name for proper identification!""")
        super().__init__(name, level=l_level)  # logs everything, finer level of control is given to its handlers
        print(f"Logger `{name}` from `{dialect}` is instantiated with level {l_level}.")
        self.file_path = file_path  # proper update to this value by self.add_filehandler()
        if dialect == "colorlog":
            import colorlog
            module: Any = colorlog
            processed_fmt: Any = colorlog.ColoredFormatter(fmt or (r"%(log_color)s" + Log.get_format(sep)), datefmt="%d %H:%M:%S", log_colors=log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'fg_bold_red,bg_black', 'CRITICAL': 'thin_red', })  # see here for format: https://pypi.org/project/colorlog/
        else:
            module = logging
            processed_fmt = logging.Formatter(fmt or Log.get_format(sep))
        if file or file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.add_filehandler(file_path=file_path, fmt=processed_fmt, f_level=f_level)
        if stream: self.add_streamhandler(s_level, fmt=processed_fmt, module=module)
        self.specs = dict(dialect=dialect, name=self.name, file=file, file_path=self.file_path, stream=bool(self.get_shandler()), fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level, verbose=verbose, log_colors=log_colors)  # # this way of creating relative path makes transferrable across machines.
    def get_shandler(self): return List(handler for handler in self.handlers if "StreamHandler" in str(handler))
    def get_fhandler(self): return List(handler for handler in self.handlers if "FileHandler" in str(handler))
    def set_level(self, level: int, which: Literal["logger", "stream", "file", "all"] = "logger"):
        if which in {"logger", "all"}: self.setLevel(level)
        if which in {"stream", "all"}: self.get_shandler().setLevel(level)
        if which in {"file", "all"}: self.get_fhandler().setLevel(level)
    def __reduce_ex__(self, protocol: Any): _ = protocol; return self.__class__, tuple(self.specs.values())  # reduce_ex is enchanced reduce. It is lower than getstate and setstate. It uses init method to create an instance.
    def __repr__(self): return "".join([f"Logger {self.name} (level {self.level}) with handlers: \n"] + [repr(h) + "\n" for h in self.handlers])
    @staticmethod
    def get_format(sep: str = ' | ', datefmt: str = "%d %H:%M:%S"):
        _ = datefmt  # TODO: add datefmt to the format string
        return f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s(%(levelno)s){sep}%(message)s{sep}"  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes logging.BASIC_FORMAT
    def manual_degug(self, path: PLike):
        _ = self
        sys.stdout = open(path, 'w', encoding="utf-8")
        sys.stdout.close()
        print(f"""✅ Debug operation completed.
   Output debug file located at: {path}""")
    @staticmethod
    def get_coloredlogs(name: Optional[str] = None, file: bool = False, file_path: OPLike = None, stream: bool = True, fmt: Optional[str] = None, sep: str = " | ", s_level: int = logging.DEBUG, f_level: int = logging.DEBUG, l_level: int = logging.DEBUG, verbose: bool = False):
        level_styles = {'spam': {'color': 'green', 'faint': True}, 'debug': {'color': 'white'}, 'verbose': {'color': 'blue'}, 'info': {'color': "green"}, 'notice': {'color': 'magenta'}, 'warning': {'color': 'yellow'}, 'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True, "underline": True}, 'critical': {'color': 'red', 'bold': True, "inverse": False}}  # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        field_styles = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'color': 'black', 'bold': True}, 'path': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}}
        if verbose:
            logger = install_n_import("verboselogs").VerboseLogger(name=name)
            logger.setLevel(l_level)  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
        else: logger = Log(name=name, dialect="logging", l_level=l_level, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
        install_n_import("coloredlogs").install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger
    def add_streamhandler(self, s_level: int = logging.DEBUG, fmt: Optional[Any] = None, module: Any = logging, name: str = "myStreamHandler"):
        shandler = module.StreamHandler()
        shandler.setLevel(level=s_level)
        shandler.setFormatter(fmt=fmt)
        shandler.set_name(name)
        self.addHandler(shandler)
        print(f"""✅ [STREAM HANDLER] Created for Logger: {self.name}
   Handler Level: {s_level}""")
    def add_filehandler(self, file_path: OPLike = None, fmt: Optional[Any] = None, f_level: int = logging.DEBUG, mode: str = "a", name: str = "myFileHandler"):
        filename = P.tmpfile(name=self.name, suffix=".log", folder="tmp_loggers") if file_path is None else P(file_path).expanduser()
        fhandler = logging.FileHandler(filename=filename, mode=mode)
        fhandler.setFormatter(fmt=fmt)
        fhandler.setLevel(level=f_level)
        fhandler.set_name(name)
        self.addHandler(fhandler)
        self.file_path = filename.collapseuser(strict=False)
        print(f"""📁 [FILE HANDLER] Created for Logger: {self.name}
   Level: {f_level} at location: {P(filename).clickable()}""")
    def test(self):
        List([self.debug, self.info, self.warning, self.error, self.critical]).apply(lambda func: func(f"this is a {func.__name__} message"))
        for level in range(0, 60, 5): self.log(msg=f"This is a message of level {level}", level=level)
    def get_history(self, lines: int = 200, to_html: bool = False):
        assert isinstance(self.file_path, P), f"Logger `{self.name}` does not have a file handler. Thus, no history is available."
        logs = "\n".join(self.file_path.expanduser().absolute().read_text().split("\n")[-lines:])
        return install_n_import("ansi2html").Ansi2HTMLConverter().convert(logs) if to_html else logs


@dataclass
class STD:
    stdin: str
    stdout: str
    stderr: str
    returncode: int


class Response:
    @staticmethod
    def from_completed_process(cp: subprocess.CompletedProcess[str]):
        resp = Response(cmd=cp.args)
        resp.output.stdout = cp.stdout
        resp.output.stderr = cp.stderr
        resp.output.returncode = cp.returncode
        return resp
    def __init__(self, stdin: Optional[BinaryIO] = None, stdout: Optional[BinaryIO] = None, stderr: Optional[BinaryIO] = None, cmd: Optional[str] = None, desc: str = ""):
        self.std = dict(stdin=stdin, stdout=stdout, stderr=stderr)
        self.output = STD(stdin="", stdout="", stderr="", returncode=0)
        self.input = cmd
        self.desc = desc  # input command
    def __call__(self, *args: Any, **kwargs: Any) -> Optional[str]:
        _ = args, kwargs
        return self.op.rstrip() if type(self.op) is str else None
    @property
    def op(self) -> str: return self.output.stdout
    @property
    def ip(self) -> str: return self.output.stdin
    @property
    def err(self) -> str: return self.output.stderr
    @property
    def returncode(self) -> int: return self.output.returncode
    def op2path(self, strict_returncode: bool = True, strict_err: bool = False) -> Union[P, None]:
        if self.is_successful(strict_returcode=strict_returncode, strict_err=strict_err): return P(self.op.rstrip())
        return None
    def op_if_successfull_or_default(self, strict_returcode: bool = True, strict_err: bool = False) -> Optional[str]: return self.op if self.is_successful(strict_returcode=strict_returcode, strict_err=strict_err) else None
    def is_successful(self, strict_returcode: bool = True, strict_err: bool = False) -> bool:
        return ((self.returncode in {0, None}) if strict_returcode else True) and (self.err == "" if strict_err else True)
    def capture(self):
        for key in ["stdin", "stdout", "stderr"]:
            val: Optional[BinaryIO] = self.std[key]
            if val is not None and val.readable():
                self.output.__dict__[key] = val.read().decode().rstrip()
        return self
    def print_if_unsuccessful(self, desc: str = "TERMINAL CMD", strict_err: bool = False, strict_returncode: bool = False, assert_success: bool = False):
        success = self.is_successful(strict_err=strict_err, strict_returcode=strict_returncode)
        if assert_success: assert success, self.print(capture=False, desc=desc)
        if success: print(desc)
        else: self.print(capture=False, desc=desc)
        return self
    def print(self, desc: str = "TERMINAL CMD", capture: bool = True):
        if capture: self.capture()
        from rich import console
        con = console.Console()
        from rich.panel import Panel
        from rich.text import Text  # from rich.syntax import Syntax; syntax = Syntax(my_code, "python", theme="monokai", line_numbers=True)
        tmp1 = Text("Input Command:\n")
        tmp1.stylize("u bold blue")
        tmp2 = Text("\nTerminal Response:\n")
        tmp2.stylize("u bold blue")
        list_str = [f"{f' {idx} - {key} '}".center(40, "-") + f"\n{val}" for idx, (key, val) in enumerate(self.output.__dict__.items())]
        txt = tmp1 + Text(str(self.input), style="white") + tmp2 + Text("\n".join(list_str), style="white")
        con.print(Panel(txt, title=self.desc, subtitle=desc, width=150, style="bold cyan on black"))
        return self


class Terminal:
    def __init__(self, stdout: Optional[int] = subprocess.PIPE, stderr: Optional[int] = subprocess.PIPE, stdin: Optional[int] = subprocess.PIPE, elevated: bool = False):
        import platform
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


class SSH:  # inferior alternative: https://github.com/fabric/fabric
    def __init__(self, host: Optional[str] = None, username: Optional[str] = None, hostname: Optional[str] = None, sshkey: Optional[str] = None, pwd: Optional[str] = None, port: int = 22, ve: Optional[str] = "ve", compress: bool = False):  # https://stackoverflow.com/questions/51027192/execute-command-script-using-different-shell-in-ssh-paramiko
        self.pwd = pwd
        self.ve = ve
        self.compress = compress  # Defaults: (1) use localhost if nothing provided.

        self.host: Optional[str] = None
        self.hostname: str
        self.username: str
        self.port: int = port
        self.proxycommand: Optional[str] = None
        import platform
        import paramiko  # type: ignore
        import getpass
        if isinstance(host, str):
            try:
                import paramiko.config as pconfig
                config = pconfig.SSHConfig.from_path(P.home().joinpath(".ssh/config").to_str())
                config_dict = config.lookup(host)
                self.hostname = config_dict["hostname"]
                self.username = config_dict["user"]
                self.host = host
                self.port = int(config_dict.get("port", port))
                tmp = config_dict.get("identityfile", sshkey)
                if isinstance(tmp, list): sshkey = tmp[0]
                else: sshkey = tmp
                self.proxycommand = config_dict.get("proxycommand", None)
                if sshkey is not None:
                    tmp = config.lookup("*").get("identityfile", sshkey)
                    if type(tmp) is list: sshkey = tmp[0]
                    else: sshkey = tmp
            except (FileNotFoundError, KeyError):
                assert "@" in host or ":" in host, f"Host must be in the form of `username@hostname:port` or `username@hostname` or `hostname:port`, but it is: {host}"
                if "@" in host: self.username, self.hostname = host.split("@")
                else:
                    self.username = username or getpass.getuser()
                    self.hostname = host
                if ":" in self.hostname:
                    self.hostname, port_ = self.hostname.split(":")
                    self.port = int(port_)
        elif username is not None and hostname is not None:
            self.username, self.hostname = username, hostname
            self.proxycommand = None
        else:
            print(f"Provided values: host={host}, username={username}, hostname={hostname}")
            raise ValueError("Either host or username and hostname must be provided.")

        self.sshkey = str(P(sshkey).expanduser().absolute()) if sshkey is not None else None  # no need to pass sshkey if it was configured properly already
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        import rich
        rich.inspect(Struct(host=self.host, hostname=self.hostname, username=self.username, password="***", port=self.port, key_filename=self.sshkey, ve=self.ve), value=False, title="SSHing To", docs=False, sort=False)
        sock = paramiko.ProxyCommand(self.proxycommand) if self.proxycommand is not None else None
        try:
            if pwd is None:
                allow_agent = True
                look_for_keys = True
            else:
                allow_agent = False
                look_for_keys = False
            self.ssh.connect(hostname=self.hostname, username=self.username, password=self.pwd, port=self.port, key_filename=self.sshkey, compress=self.compress, sock=sock,
                             allow_agent=allow_agent, look_for_keys=look_for_keys)  # type: ignore
        except Exception as _err:
            rich.console.Console().print_exception()
            self.pwd = getpass.getpass(f"Enter password for {self.username}@{self.hostname}: ")
            self.ssh.connect(hostname=self.hostname, username=self.username, password=self.pwd, port=self.port, key_filename=self.sshkey, compress=self.compress, sock=sock,
                             allow_agent=False,look_for_keys=False)  # type: ignore

        try: self.sftp: Optional[paramiko.SFTPClient] = self.ssh.open_sftp()
        except Exception as err:
            self.sftp = None
            print(f"""⚠️  WARNING: Failed to open SFTP connection to {hostname}.
   Error Details: {err}
   Data transfer may be affected!""")
        def view_bar(slf: Any, a: Any, b: Any):
            slf.total = int(b)
            slf.update(int(a - slf.n))  # update pbar with increment
        from tqdm import tqdm
        self.tqdm_wrap = type('TqdmWrap', (tqdm,), {'view_bar': view_bar})
        self._local_distro: Optional[str] = None
        self._remote_distro: Optional[str] = None
        self._remote_machine: Optional[MACHINE] = None
        self.terminal_responses: list[Response] = []
        self.platform = platform
        self.remote_env_cmd = rf"""~/venvs/{self.ve}/Scripts/Activate.ps1""" if self.get_remote_machine() == "Windows" else rf"""source ~/venvs/{self.ve}/bin/activate"""
        self.local_env_cmd = rf"""~/venvs/{self.ve}/Scripts/Activate.ps1""" if self.platform.system() == "Windows" else rf"""source ~/venvs/{self.ve}/bin/activate"""  # works for both cmd and pwsh
    def __getstate__(self): return {attr: self.__getattribute__(attr) for attr in ["username", "hostname", "host", "port", "sshkey", "compress", "pwd", "ve"]}
    def __setstate__(self, state: dict[str, Any]): SSH(**state)
    def get_remote_machine(self) -> MACHINE:
        if self._remote_machine is None:
            if (self.run("$env:OS", verbose=False, desc="Testing Remote OS Type").op == "Windows_NT" or self.run("echo %OS%", verbose=False, desc="Testing Remote OS Type Again").op == "Windows_NT"): self._remote_machine = "Windows"
            else: self._remote_machine = "Linux"
        return self._remote_machine  # echo %OS% TODO: uname on linux
    def get_local_distro(self) -> str:
        if self._local_distro is None:
            res = install_n_import("distro").name(pretty=True)
            self._local_distro = res
            return res
        return self._local_distro
    def get_remote_distro(self):
        if self._remote_distro is None: self._remote_distro = self.run_py("print(install_n_import('distro').name(pretty=True))", verbose=False).op_if_successfull_or_default() or ""
        return self._remote_distro
    def restart_computer(self): self.run("Restart-Computer -Force" if self.get_remote_machine() == "Windows" else "sudo reboot")
    def send_ssh_key(self):
        self.copy_from_here("~/.ssh/id_rsa.pub")
        assert self.get_remote_machine() == "Windows"
        self.run(P(install_n_import("machineconfig").scripts.windows.__path__.__dict__["_path"][0]).joinpath("openssh_server_add_sshkey.ps1").read_text())
    def copy_env_var(self, name: str):
        assert self.get_remote_machine() == "Linux"
        return self.run(f"{name} = {os.environ[name]}; export {name}")
    def get_remote_repr(self, add_machine: bool = False) -> str: return f"{self.username}@{self.hostname}:{self.port}" + (f" [{self.get_remote_machine()}][{self.get_remote_distro()}]" if add_machine else "")
    def get_local_repr(self, add_machine: bool = False) -> str:
        import getpass
        return f"{getpass.getuser()}@{self.platform.node()}" + (f" [{self.platform.system()}][{self.get_local_distro()}]" if add_machine else "")
    def __repr__(self): return f"local {self.get_local_repr(add_machine=True)} >>> SSH TO >>> remote {self.get_remote_repr(add_machine=True)}"
    def run_locally(self, command: str):
        print(f"""💻 [LOCAL EXECUTION] Running command on node: {self.platform.node()}
Command:
{command}""")
        print(f"Executing Locally @ {self.platform.node()}:\n{command}")
        res = Response(cmd=command)
        res.output.returncode = os.system(command)
        return res
    def get_ssh_conn_str(self, cmd: str = ""): return "ssh " + (f" -i {self.sshkey}" if self.sshkey else "") + self.get_remote_repr().replace(':', ' -p ') + (f' -t {cmd} ' if cmd != '' else ' ')
    def open_console(self, cmd: str = '', new_window: bool = True, terminal: Optional[str] = None, shell: str = "pwsh"): Terminal().run_async(*(self.get_ssh_conn_str(cmd=cmd).split(" ")), new_window=new_window, terminal=terminal, shell=shell)
    def run(self, cmd: str, verbose: bool = True, desc: str = "", strict_err: bool = False, strict_returncode: bool = False, env_prefix: bool = False) -> Response:  # most central method.
        cmd = (self.remote_env_cmd + "; " + cmd) if env_prefix else cmd
        raw = self.ssh.exec_command(cmd)
        res = Response(stdin=raw[0], stdout=raw[1], stderr=raw[2], cmd=cmd, desc=desc)  # type: ignore
        if not verbose: res.capture().print_if_unsuccessful(desc=desc, strict_err=strict_err, strict_returncode=strict_returncode, assert_success=False)
        else: res.print()
        self.terminal_responses.append(res)
        return res
    def run_py(self, cmd: str, desc: str = "", return_obj: bool = False, verbose: bool = True, strict_err: bool = False, strict_returncode: bool = False) -> Union[Any, Response]:
        assert '"' not in cmd, 'Avoid using `"` in your command. I dont know how to handle this when passing is as command to python in pwsh command.'
        if not return_obj: return self.run(cmd=f"""{self.remote_env_cmd}; python -c "{Terminal.get_header(wdir=None, toolbox=True)}{cmd}\n""" + '"', desc=desc or f"run_py on {self.get_remote_repr()}", verbose=verbose, strict_err=strict_err, strict_returncode=strict_returncode)
        assert "obj=" in cmd, "The command sent to run_py must have `obj=` statement if return_obj is set to True"
        source_file = self.run_py(f"""{cmd}\npath = Save.pickle(obj=obj, path=P.tmpfile(suffix='.pkl'))\nprint(path)""", desc=desc, verbose=verbose, strict_err=True, strict_returncode=True).op.split('\n')[-1]
        return self.copy_to_here(source=source_file, target=P.tmpfile(suffix='.pkl')).readit()
    def copy_from_here(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False, overwrite: bool = False, init: bool = True) -> Union[P, list[P]]:
        if init:
            print(f"""⬆️⬆️⬆️⬆️⬆️  [SFTP UPLOAD] Initiating file upload from: {source} to: {target}""")
        if init: print(f"{'⬆️' * 5} SFTP UPLOADING FROM `{source}` TO `{target}`")  # TODO: using return_obj do all tests required in one go.
        source_obj = P(source).expanduser().absolute()
        if target is None:
            target = P(source_obj).expanduser().absolute().collapseuser(strict=True)
            assert target.is_relative_to("~"), "If target is not specified, source must be relative to home."
            if z: target += ".zip"
        if not z and source_obj.is_dir():
            if r is False: raise RuntimeError(f"Meta.SSH Error: source `{source_obj}` is a directory! either set `r=True` for recursive sending or raise `z=True` flag to zip it first.")
            source_list: List[P] = source_obj.search("*", folders=False, r=True)
            remote_root = self.run_py(f"path=P(r'{P(target).as_posix()}').expanduser()\n{'path.delete(sure=True)' if overwrite else ''}\nprint(path.create())", desc=f"Creating Target directory `{P(target).as_posix()}` @ {self.get_remote_repr()}", verbose=False).op or ''
            _ = [self.copy_from_here(source=item, target=P(remote_root).joinpath(item.name)) for item in source_list]
            return list(source_list)
        if z:
            print("🗜️ ZIPPING ...")
            source_obj = P(source_obj).expanduser().zip(content=True)  # .append(f"_{randstr()}", inplace=True)  # eventually, unzip will raise content flag, so this name doesn't matter.
        remotepath = self.run_py(f"path=P(r'{P(target).as_posix()}').expanduser()\n{'path.delete(sure=True)' if overwrite else ''}\nprint(path.parent.create())", desc=f"Creating Target directory `{P(target).parent.as_posix()}` @ {self.get_remote_repr()}", verbose=False).op or ''
        remotepath = P(remotepath.split("\n")[-1]).joinpath(P(target).name)
        print(f"""📤 [UPLOAD] Sending file:
   {repr(P(source_obj))}  ==>  Remote Path: {remotepath.as_posix()}""")
        print(f"SENDING `{repr(P(source_obj))}` ==> `{remotepath.as_posix()}`")
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar: self.sftp.put(localpath=P(source_obj).expanduser(), remotepath=remotepath.as_posix(), callback=pbar.view_bar)  # type: ignore # pylint: disable=E1129
        if z:
            _resp = self.run_py(f"""P(r'{remotepath.as_posix()}').expanduser().unzip(content=False, inplace=True, overwrite={overwrite})""", desc=f"UNZIPPING {remotepath.as_posix()}", verbose=False, strict_err=True, strict_returncode=True)
            source_obj.delete(sure=True)
            print("\n")
        return source_obj
    def copy_to_here(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False, init: bool = True) -> P:
        if init:
            print(f"""⬇️⬇️⬇️⬇️⬇️ 
[SFTP DOWNLOAD] Initiating download from: {source}
                to: {target}""")
        if init: print(f"{'⬇️' * 5} SFTP DOWNLOADING FROM `{source}` TO `{target}`")
        if not z and self.run_py(f"print(P(r'{source}').expanduser().absolute().is_dir())", desc=f"Check if source `{source}` is a dir", verbose=False, strict_returncode=True, strict_err=True).op.split("\n")[-1] == 'True':
            if r is False: raise RuntimeError(f"source `{source}` is a directory! either set r=True for recursive sending or raise zip_first flag.")
            source_list = self.run_py(f"obj=P(r'{source}').search(folders=False, r=True).collapseuser(strict=False)", desc="Searching for files in source", return_obj=True, verbose=False)
            assert isinstance(source_list, List), f"Could not resolve source path {source} due to error"
            for file in source_list:
                self.copy_to_here(source=file.as_posix(), target=P(target).joinpath(P(file).relative_to(source)) if target else None, r=False)
        if z:
            tmp: Response = self.run_py(f"print(P(r'{source}').expanduser().zip(inplace=False, verbose=False))", desc=f"Zipping source file {source}", verbose=False)
            tmp2 = tmp.op2path(strict_returncode=True, strict_err=True)
            if not isinstance(tmp2, P): raise RuntimeError(f"Could not zip {source} due to {tmp.err}")
            else: source = tmp2
        if target is None:
            tmpx = self.run_py(f"print(P(r'{P(source).as_posix()}').collapseuser(strict=False).as_posix())", desc="Finding default target via relative source path", strict_returncode=True, strict_err=True, verbose=False).op2path()
            if isinstance(tmpx, P): target = tmpx
            else: raise RuntimeError(f"Could not resolve target path {target} due to error")
            assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home.\n{target=}"
        target_obj = P(target).expanduser().absolute().create(parents_only=True)
        if z and '.zip' not in target_obj.suffix: target_obj += '.zip'
        if "~" in str(source):
            tmp3 = self.run_py(f"print(P(r'{source}').expanduser())", desc="# Resolving source path address by expanding user", strict_returncode=True, strict_err=True, verbose=False).op2path()
            if isinstance(tmp3, P): source = tmp3
            else: raise RuntimeError(f"Could not resolve source path {source} due to")
        else: source = P(source)
        print(f"""📥 [DOWNLOAD] Receiving:
   {source}  ==>  Local Path: {target_obj}""")
        print(f"RECEVING `{source}` ==> `{target_obj}`")
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar:  # type: ignore # pylint: disable=E1129
            assert self.sftp is not None, f"Could not establish SFTP connection to {self.hostname}."
            self.sftp.get(remotepath=source.as_posix(), localpath=str(target_obj), callback=pbar.view_bar)  # type: ignore
        if z:
            target_obj = target_obj.unzip(inplace=True, content=True)
            self.run_py(f"P(r'{source.as_posix()}').delete(sure=True)", desc="Cleaning temp zip files @ remote.", strict_returncode=True, strict_err=True, verbose=False)
        print("\n")
        return target_obj
    def receieve(self, source: PLike, target: OPLike = None, z: bool = False, r: bool = False) -> P:
        scout = self.run_py(cmd=f"obj=SSH.scout(r'{source}', z={z}, r={r})", desc=f"Scouting source `{source}` path on remote", return_obj=True, verbose=False)
        assert isinstance(scout, Scout)
        if not z and scout.is_dir and scout.files is not None:
            if r:
                tmp: List[P] = scout.files.apply(lambda file: self.receieve(source=file.as_posix(), target=P(target).joinpath(P(file).relative_to(source)) if target else None, r=False))
                return tmp.list[0]
            else: print("Source is a directory! either set `r=True` for recursive sending or raise `zip_first=True` flag.")
        target = P(target).expanduser().absolute().create(parents_only=True) if target else scout.source_rel2home.expanduser().absolute().create(parents_only=True)
        if z and '.zip' not in target.suffix: target += '.zip'
        source = scout.source_full
        with self.tqdm_wrap(ascii=True, unit='b', unit_scale=True) as pbar: self.sftp.get(remotepath=source.as_posix(), localpath=target.as_posix(), callback=pbar.view_bar)  # type: ignore # pylint: disable=E1129
        if z:
            target = target.unzip(inplace=True, content=True)
            self.run_py(f"P(r'{source.as_posix()}').delete(sure=True)", desc="Cleaning temp zip files @ remote.", strict_returncode=True, strict_err=True)
        print("\n")
        return target
    @staticmethod
    def scout(source: PLike, z: bool = False, r: bool = False) -> Scout:
        source_full = P(source).expanduser().absolute()
        source_rel2home = source_full.collapseuser()
        exists = source_full.exists()
        is_dir = source_full.is_dir() if exists else False
        if z and exists:
            try: source_full = source_full.zip()
            except Exception as ex:
                raise Exception(f"Could not zip {source_full} due to {ex}") from ex  # type: ignore # pylint: disable=W0719
            source_rel2home = source_full.zip()
        files = source_full.search(folders=False, r=True).apply(lambda x: x.collapseuser()) if r and exists and is_dir else None
        return Scout(source_full=source_full, source_rel2home=source_rel2home, exists=exists, is_dir=is_dir, files=files)
    def print_summary(self):
        import pandas as pd
        df = pd.DataFrame.from_records(List(self.terminal_responses).apply(lambda rsp: dict(desc=rsp.desc, err=rsp.err, returncode=rsp.returncode)))
        print("\nSummary of operations performed:")
        print(df.to_markdown())
        if ((df['returncode'].to_list()[2:] == [None] * (len(df) - 2)) and (df['err'].to_list()[2:] == [''] * (len(df) - 2))): print("\nAll operations completed successfully.\n")
        else: print("\nSome operations failed. \n")
        return df


class Scheduler:
    def __init__(self, routine: Callable[['Scheduler'], Any], wait: Union[str, int, float] = "2m", max_cycles: int = 10000000000,
                 exception_handler: Optional[Callable[[Union[Exception, KeyboardInterrupt], str, 'Scheduler'], Any]] = None,
                 logger: Optional[Log] = None, sess_stats: Optional[Callable[['Scheduler'], dict[str, Any]]] = None, records: Optional[list[list[Any]]] = None):
        self.routine = routine  # main routine to be repeated every `wait` time period
        self.wait_sec = str2timedelta(shift=wait).total_seconds() if isinstance(wait, str) else wait  # wait period between routine cycles.
        self.logger = logger if logger is not None else Log(name="SchedLogger_" + randstr(noun=True))
        self.exception_handler = exception_handler if exception_handler is not None else self.default_exception_handler
        self.sess_start_time = datetime.now()  # to be reset at .run
        self.records: list[list[Any]] = records if records is not None else []
        self.cycle: int = 0
        self.max_cycles: int = max_cycles
        self.sess_stats = sess_stats or (lambda _sched: {})
    def __repr__(self): return f"Scheduler with {self.cycle} cycles ran so far. Last cycle was at {self.sess_start_time}."
    def run(self, max_cycles: Optional[int] = None, until: str = "2050-01-01", starting_cycle: Optional[int] = None):
        if starting_cycle is not None: self.cycle = starting_cycle
        self.max_cycles, self.sess_start_time = max_cycles or self.max_cycles, datetime.now()
        while datetime.now() < datetime.fromisoformat(until) and self.cycle < self.max_cycles:  # 1- Time before Ops, and Opening Message
            time1 = datetime.now()
            self.logger.info(f"🚦Cycle {str(self.cycle).zfill(6)}. Total Run⏳ = {str(datetime.now() - self.sess_start_time).split('.', maxsplit=1)[0]}. UTC🕜 {datetime.now(tz=dtm.UTC).strftime('%d %H:%M:%S')}")
            try: self.routine(self)
            except Exception as ex: self.exception_handler(ex, "routine", self)  # 2- Perform logic
            time_left = int(self.wait_sec - (datetime.now() - time1).total_seconds())  # 4- Conclude Message
            self.cycle += 1
            self.logger.info(f"🏁Cycle {str(self.cycle - 1).zfill(6)} in {str(datetime.now() - time1).split('.', maxsplit=1)[0]}. Sleeping for {self.wait_sec}s ({time_left}s left)\n" + "-" * 100)
            try: time.sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex:
                self.exception_handler(ex, "sleep", self)
                return  # that's probably the only kind of exception that can rise during sleep.
        self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until})")
    def get_records_df(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "termination reason", "logfile"] + list(self.sess_stats(self).keys()))
    def record_session_end(self, reason: str = "Not passed to function."):
        end_time = datetime.now()
        duration = end_time - self.sess_start_time
        sess_stats = self.sess_stats(self)
        self.records.append([self.sess_start_time, end_time, duration, self.cycle, reason, self.logger.file_path] + list(sess_stats.values()))
        summ = {
            "start time": f"{str(self.sess_start_time)}",
            "finish time": f"{str(end_time)}.",
            "duration": f"{str(duration)} | wait time {self.wait_sec}s",
            "cycles ran": f"{self.cycle} | Lifetime cycles = {self.get_records_df()['cycles'].sum()}",
            "termination reason": reason, "logfile": self.logger.file_path
                }
        tmp = Struct(summ).update(sess_stats).print(as_config=True, return_str=True, quotes=False)
        assert isinstance(tmp, str)
        self.logger.critical("\n--> Scheduler has finished running a session. \n" + tmp + "\n" + "-" * 100)
        df = self.get_records_df()
        df["start"] = df["start"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["finish"] = df["finish"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["duration"] = df["duration"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        self.logger.critical("\n--> Logger history.\n" + str(df))
        return self
    def default_exception_handler(self, ex: Union[Exception, KeyboardInterrupt], during: str, sched: 'Scheduler') -> None:  # user decides on handling and continue, terminate, save checkpoint, etc.  # Use signal library.
        print(sched)
        self.record_session_end(reason=f"during {during}, " + str(ex))
        self.logger.exception(ex)
        raise ex


class SchedulerV2:
    def __init__(self, routine: Callable[['SchedulerV2'], Any],
                 wait_ms: int,
                 exception_handler: Optional[Callable[[Union[Exception, KeyboardInterrupt], str, 'SchedulerV2'], Any]] = None,
                 logger: Optional[Log] = None,
                 sess_stats: Optional[Callable[['SchedulerV2'], dict[str, Any]]] = None,
                 records: Optional[list[list[Any]]] = None):
        self.routine = routine  # main routine to be repeated every `wait` time period
        self.logger = logger if logger is not None else Log(name="SchedLogger_" + randstr(noun=True))
        self.exception_handler = exception_handler if exception_handler is not None else self.default_exception_handler
        self.records: list[list[Any]] = records if records is not None else []
        self.wait_ms = wait_ms  # wait period between routine cycles.
        self.cycle: int = 0
        self.max_cycles: int
        self.sess_start_time: int
        self.sess_stats = sess_stats or (lambda _sched: {})
    def __repr__(self): return f"Scheduler with {self.cycle} cycles ran so far. Last cycle was at {self.sess_start_time}."
    def run(self, max_cycles: Optional[int], until_ms: int):
        if max_cycles is not None:
            self.max_cycles = max_cycles
        self.sess_start_time = time.time_ns() // 1_000_000
        while (time.time_ns() // 1_000_000) < until_ms and self.cycle < self.max_cycles:
            # 1- Time before Ops, and Opening Message
            time1 = time.time_ns() // 1_000_000
            self.logger.info(f"Starting Cycle {str(self.cycle).zfill(5)}. Total Run Time = {str(time1 - self.sess_start_time).split('.', maxsplit=1)[0]}. UTC🕜 {datetime.now(tz=dtm.UTC).strftime('%d %H:%M:%S')}")
            try:
                self.routine(self)
            except Exception as ex:
                self.exception_handler(ex, "routine", self)  # 2- Perform logic
            time2 = time.time_ns() // 1_000_000
            time_left = int(self.wait_ms - (time2 - time1))  # 4- Conclude Message
            self.cycle += 1
            self.logger.info(f"Finishing Cycle {str(self.cycle - 1).zfill(5)} in {str(time2 - time1).split('.', maxsplit=1)[0]}. Sleeping for {self.wait_ms}ms ({time_left}s left)\n" + "-" * 100)
            try: time.sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex:
                self.exception_handler(ex, "sleep", self)
                return  # that's probably the only kind of exception that can rise during sleep.
        self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until_ms})")
    def get_records_df(self):
        import pandas as pd
        return pd.DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "termination reason", "logfile"] + list(self.sess_stats(self).keys()))
    def record_session_end(self, reason: str):
        end_time = time.time_ns() // 1_000_000
        duration = end_time - self.sess_start_time
        sess_stats = self.sess_stats(self)
        self.records.append([self.sess_start_time, end_time, duration, self.cycle, reason, self.logger.file_path] + list(sess_stats.values()))
        summ = {"start time": f"{str(self.sess_start_time)}",
                "finish time": f"{str(end_time)}.",
                "duration": f"{str(duration)} | wait time {self.wait_ms / 1_000: 0.1f}s",
                "cycles ran": f"{self.cycle} | Lifetime cycles = {self.get_records_df()['cycles'].sum()}",
                "termination reason": reason, "logfile": self.logger.file_path
                }
        tmp = Struct(summ).update(sess_stats).print(as_config=True, return_str=True, quotes=False)
        assert isinstance(tmp, str)
        self.logger.critical("\n--> Scheduler has finished running a session. \n" + tmp + "\n" + "-" * 100)
        df = self.get_records_df()
        df["start"] = df["start"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["finish"] = df["finish"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        df["duration"] = df["duration"].apply(lambda x: str(x).split(".", maxsplit=1)[0])
        self.logger.critical("\n--> Logger history.\n" + str(df))
        return self
    def default_exception_handler(self, ex: Union[Exception, KeyboardInterrupt], during: str, sched: 'SchedulerV2') -> None:  # user decides on handling and continue, terminate, save checkpoint, etc.  # Use signal library.
        print(sched)
        self.record_session_end(reason=f"during {during}, " + str(ex))
        self.logger.exception(ex)
        raise ex


def generate_readme(path: PLike, obj: Any = None, desc: str = '', save_source_code: bool = True, verbose: bool = True):  # Generates a readme file to contextualize any binary files by mentioning module, class, method or function used to generate the data"""
    import inspect
    text: str = "# Description\n" + desc + (separator := "\n" + "-" * 50 + "\n\n")
    obj_path = P(inspect.getfile(obj)) if obj is not None else None
    path = P(path)
    if obj_path is not None:
        text += f"# Source code file generated me was located here: \n`{obj_path.collapseuser().as_posix()}`\n" + separator
        try:
            repo = install_n_import(library="git", package="gitpython").Repo(obj_path.parent, search_parent_directories=True)
            text += f"# Last Commit\n{repo.git.execute(['git', 'log', '-1'])}{separator}# Remote Repo\n{repo.git.execute(['git', 'remote', '-v'])}{separator}"
            try: tmppp = obj_path.relative_to(repo.working_dir).as_posix()
            except Exception: tmppp = ""  # type: ignore
            text += f"# link to files: \n{repo.remote().url.replace('.git', '')}/tree/{repo.active_branch.commit.hexsha}/{tmppp}{separator}"
        except Exception as ex: text += f"Could not read git repository @ `{obj_path.parent}`\n{ex}.\n"
    if obj is not None:
        try: source_code = inspect.getsource(obj)
        except OSError: source_code = f"Could not read source code from `{obj_path}`."
        text += ("\n\n# Code to reproduce results\n\n```python\n" + source_code + "\n```" + separator)
    readmepath = (path / "README.md" if path.is_dir() else (path.with_name(path.trunk + "_README.md") if path.is_file() else path)).write_text(text, encoding="utf-8")
    if verbose: print(f"💽 SAVED {readmepath.name} @ {readmepath.absolute().as_uri()}")
    if save_source_code:
        if hasattr(obj, "__code__"):
            save_path = obj.__code__.co_filename
        else:
            module_maybe = inspect.getmodule(obj)
            if module_maybe is not None: save_path = module_maybe.__file__
            else: save_path = None
        if save_path is None:
            print(f"Could not find source code for {obj}.")
            return readmepath
        P(save_path).zip(path=readmepath.with_name(P(readmepath).trunk + "_source_code.zip"), verbose=False)
        print("💽 SAVED source code @ " + readmepath.with_name("source_code.zip").absolute().as_uri())
        return readmepath


class RepeatUntilNoException:
    """
    Repeat function calling if it raised an exception and/or exceeded the timeout, for a maximum of `retry` times.
    * Alternative: `https://github.com/jd/tenacity`
    """
    def __init__(self, retry: int, sleep: float, timeout: Optional[float] = None, scaling: Literal["linear", "exponential"] = "exponential"):
        self.retry = retry
        self.sleep = sleep
        self.timeout = timeout
        self.scaling: Literal["linear", "exponential"] = scaling
    def __call__(self, func: Callable[PS, T]) -> Callable[PS, T]:
        from functools import wraps
        if self.timeout is not None:
            func = install_n_import("wrapt_timeout_decorator").timeout(self.timeout)(func)
        @wraps(wrapped=func)
        def wrapper(*args: PS.args, **kwargs: PS.kwargs):
            t0 = time.time()
            for idx in range(self.retry):
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    match self.scaling:
                        case "linear":
                            sleep_time = self.sleep * (idx + 1)
                        case "exponential":
                            sleep_time = self.sleep * (idx + 1)**2
                    print(f"""💥 [RETRY] Function {func.__name__} call failed with error:
{ex}
Retry count: {idx}/{self.retry}. Sleeping for {sleep_time} seconds.
Total elapsed time: {time.time() - t0:0.1f} seconds.""")
                    print(f"""💥 Robust call of `{func}` failed with ```{ex}```.\nretrying {idx}/{self.retry} more times after sleeping for {sleep_time} seconds.\nTotal wait time so far {time.time() - t0: 0.1f} seconds.""")
                    time.sleep(sleep_time)
            raise RuntimeError(f"💥 Robust call failed after {self.retry} retries and total wait time of {time.time() - t0: 0.1f} seconds.\n{func=}\n{args=}\n{kwargs=}")
        return wrapper


# @RepeatUntilNoException(retry=3, sleep=2.2)
# def add(a: int, b: int):
#     return a + b, "done"
# c = add(1, 2)


def show_globals(scope: dict[str, Any], **kwargs: Any):
    # see print_dir
    return Struct(scope).filter(lambda k, v: "__" not in k and not k.startswith("_") and k not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}).print(**kwargs)


if __name__ == '__main__':
    pass
