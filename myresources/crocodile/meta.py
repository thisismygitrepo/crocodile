
from crocodile.core import timestamp, randstr, str2timedelta, Save, install_n_import, List, Struct
from crocodile.file_management import P, datetime
import time
import logging
import subprocess
import sys


class Null:
    def __init__(self, return_='self'): self.return_ = return_
    def __getattr__(self, item): _ = item; return self if self.return_ == 'self' else self.return_
    def __getitem__(self, item): _ = item; return self if self.return_ == 'self' else self.return_
    def __call__(self, *args, **kwargs): return self if self.return_ == 'self' else self.return_
    def __len__(self): return 0
    def __bool__(self): return False
    def __contains__(self, item): _ = self, item; return False
    def __iter__(self): return iter([self])


class Log(logging.Logger):  #
    def __init__(self, dialect=["colorlog", "logging", "coloredlogs"][0], name=None, file: bool = False, file_path=None, stream=True, fmt=None, sep=" | ",
                 s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False, log_colors=None):
        super().__init__(name := randstr() if name is None and not print(f"Logger name not passed. It is recommended to pass a name indicating the owner.") else name, level=l_level)  # logs everything, finer level of control is given to its handlers
        print(f"Logger `{name}` from `{dialect}` is instantiated with level {l_level}."); self.file_path = file_path  # proper update to this value by self.add_filehandler()
        if dialect == "colorlog": module = install_n_import("colorlog"); processed_fmt = module.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), datefmt="%d %H:%M:%S", log_colors=log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'thin_red', 'CRITICAL': 'fg_bold_red,bg_black', })  # see here for format: https://pypi.org/project/colorlog/
        else: module = logging; processed_fmt = logging.Formatter(fmt or Log.get_format(sep, datefmt="%d %H:%M:%S"))
        self.add_filehandler(file_path=file_path, fmt=processed_fmt, f_level=f_level) if file or file_path else None; self.add_streamhandler(s_level, fmt=processed_fmt, module=module) if stream else None
        self.specs = dict(dialect=dialect, name=self.name, file=file, file_path=self.file_path, stream=bool(self.get_shandler()), fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level, verbose=verbose, log_colors=log_colors)  # # this way of creating relative path makes transferrable across machines.
    def get_shandler(self): return List(handler for handler in self.handlers if "StreamHandler" in str(handler))
    def get_fhandler(self): return List(handler for handler in self.handlers if "FileHandler" in str(handler))
    def set_level(self, level, which=["logger", "stream", "file", "all"][0]): self.setLevel(level) if which in {"logger", "all"} else None; self.get_shandler().setLevel(level) if which in {"stream", "all"} else None; self.get_fhandler().setLevel(level) if which in {"file", "all"} else None
    def __reduce_ex__(self, protocol): _ = protocol; return self.__class__, tuple(self.specs.values())  # reduce_ex is enchanced reduce. Its lower than getstate and setstate. It uses init method to create an instance.
    def __repr__(self): return "".join([f"Logger {self.name} (level {self.level}) with handlers: \n"] + [repr(h) + f"" + "\n" for h in self.handlers])
    get_format = staticmethod(lambda sep=' | ': f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s(%(levelno)s){sep}%(message)s{sep}")  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes logging.BASIC_FORMAT
    def manual_degug(self, path): _ = self; sys.stdout = open(path, 'w'); sys.stdout.close(); print(f"Finished ... have a look @ \n {path}")  # all print statements will write to this file.
    @staticmethod
    def get_coloredlogs(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False):
        level_styles = {'spam': {'color': 'green', 'faint': True}, 'debug': {'color': 'white'}, 'verbose': {'color': 'blue'}, 'info': {'color': "green"}, 'notice': {'color': 'magenta'}, 'warning': {'color': 'yellow'}, 'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True, "underline": True}, 'critical': {'color': 'red', 'bold': True, "inverse": False}}  # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        field_styles = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'color': 'black', 'bold': True}, 'path': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}}
        if verbose: logger = install_n_import("verboselogs").VerboseLogger(name=name); logger.setLevel(l_level)  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
        else: logger = Log(name=name, dialect="logging", l_level=l_level, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
        install_n_import("coloredlogs").install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True); return logger
    def add_streamhandler(self, s_level=logging.DEBUG, fmt=None, module=logging, name="myStreamHandler"):
        shandler = module.StreamHandler(); shandler.setLevel(level=s_level); shandler.setFormatter(fmt=fmt); shandler.set_name(name); self.addHandler(shandler); print(f"    Level {s_level} stream handler for Logger `{self.name}` is created.")
    def add_filehandler(self, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="myFileHandler"):
        fhandler = logging.FileHandler(filename := (P.tmpfile(name="logger_" + self.name, suffix=".log", folder="tmp_loggers") if file_path is None else P(file_path).expanduser()), mode=mode)
        fhandler.setFormatter(fmt=fmt); fhandler.setLevel(level=f_level); fhandler.set_name(name); self.addHandler(fhandler); self.file_path = filename.collapseuser(); print(f"    Level {f_level} file handler for Logger `{self.name}` is created @ " + P(filename).clickable())
    def test(self): List([self.debug, self.info, self.warning, self.error, self.critical]).apply(lambda func: func(f"this is a {func.__name__} message")); [self.log(msg=f"This is a message of level {level}", level=level) for level in range(0, 60, 5)]


class Terminal:
    class Response:
        @staticmethod
        def from_completed_process(cp: subprocess.CompletedProcess): (resp := Terminal.Response(cmd=cp.args)).output.update(dict(stdout=cp.stdout, stderr=cp.stderr, returncode=cp.returncode)); return resp
        def __init__(self, stdin=None, stdout=None, stderr=None, cmd=None): self.std, self.output, self.input = dict(stdin=stdin, stdout=stdout, stderr=stderr), dict(stdin="", stdout="", stderr="", returncode=None), cmd  # input command
        def __call__(self, *args, **kwargs): return self.op.rstrip() if type(self.op) is str else None
        op = property(lambda self: self.output["stdout"])
        ip = property(lambda self: self.output["stdin"])
        err = property(lambda self: self.output["stderr"])
        returncode = property(lambda self: self.output["returncode"])
        as_path = property(lambda self: P(self.op.rstrip()) if self.is_successful(strict_returcode=True, strict_err=False) else None)
        def is_successful(self, strict_returcode=True, strict_err=False): return ((self.output["returncode"] in {0, None}) if strict_returcode else True) and (self.err == "" if strict_err else True)
        def capture(self): [self.output.__setitem__(key, val.read().decode().rstrip()) for key, val in self.std.items() if val is not None and val.readable()]; return self
        def print(self, desc=""): self.capture(); print(desc.center(80, "=")); print(f"Input Command:\n{'~'*40}\n" + f"{self.input}" + f"\n{'~'*40}\nTerminal Response:\n" + "\n".join([f"{f' {idx} - {key} '}".center(40, "-") + f"\n{val}" for idx, (key, val) in enumerate(self.output.items())]) + "\n" + ('COMPLETED '+desc).center(80, "="), "\n\n"); return self
    def __init__(self, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, elevated=False):
        self.available_consoles = ["cmd", "Command Prompt", "wt", "powershell", "wsl", "ubuntu", "pwsh"]
        self.elevated, self.stdout, self.stderr, self.stdin = elevated, stdout, stderr, stdin
        self.machine = sys.platform  # 'win32', 'linux' OR: import platform; self.platform.system(): Windows, Linux, Darwin
    def set_std_system(self): self.stdout = sys.stdout; self.stderr = sys.stderr; self.stdin = sys.stdin
    def set_std_pipe(self): self.stdout = subprocess.PIPE; self.stderr = subprocess.PIPE; self.stdin = subprocess.PIPE
    def set_std_null(self): self.stdout, self.stderr, self.stdin = subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL  # Equivalent to `echo 'foo' &> /dev/null`
    def run(self, *cmds, shell=None, check=False, ip=None):  # Runs SYSTEM commands like subprocess.run
        """Blocking operation. Thus, if you start a shell via this method, it will run in the main and won't stop until you exit manually IF stdin is set to sys.stdin, otherwise it will run and close quickly. Other combinations of stdin, stdout can lead to funny behaviour like no output but accept input or opposite.
        * This method is short for: res = subprocess.run("powershell command", capture_output=True, shell=True, text=True) and unlike os.system(cmd), subprocess.run(cmd) gives much more control over the output and input.
        * `shell=True` loads up the profile of the shell called so more specific commands can be run. Importantly, on Windows, the `start` command becomes availalbe and new windows can be launched."""
        my_list = list(cmds)  # `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job. `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task. `suprocess.call` is an archaic command for pre-Python-3.5.
        if self.machine == "win32" and shell in {"powershell", "pwsh"}: my_list = [shell, "-Command"] + my_list  # alternatively, one can run "cmd"
        if self.elevated is False or self.is_user_admin(): resp = subprocess.run(my_list, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=check, input=ip)
        else: resp = __import__("ctypes").windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        """ The advantage of addig `powershell -Command` is to give access to wider range of options. Other wise, command prompt shell doesn't recognize commands like `ls`.
        `capture_output` prevents the stdout to redirect to the stdout of the script automatically, instead it will be stored in the Response object returned. # `capture_output=True` same as `stdout=subprocess.PIPE, stderr=subprocess.PIPE`"""
        return self.Response.from_completed_process(resp)
    @staticmethod
    def is_user_admin():  # adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows"""
        if __import__('os').name == 'nt':
            try: return __import__("ctypes").windll.shell32.IsUserAnAdmin()
            except: import traceback; traceback.print_exc(); print("Admin check failed, assuming not an admin."); return False
        else: return __import__('os').getuid() == 0  # Check for root on Posix
    @staticmethod
    def run_as_admin(file, params, wait=False): proce_info = install_n_import("win32com", fromlist=["shell.shell.ShellExecuteEx"]).shell.shell.ShellExecuteEx(lpVerb='runas', lpFile=file, lpParameters=params); time.sleep(wait) if wait is not False and wait is not True else None; return proce_info
    def run_async(self, *cmds, new_window=True, shell=None, terminal=None):  # Runs SYSTEM commands like subprocess.Popen
        """Opens a new terminal, and let it run asynchronously. Maintaining an ongoing conversation with another process is very hard. It is adviseable to run all
        commands in one go without interaction with an ongoing channel. Use this only for the purpose of producing a different window and humanly interact with it. Reference: https://stackoverflow.com/questions/54060274/dynamic-communication-between-main-and-subprocess-in-python & https://www.youtube.com/watch?v=IynV6Y80vws and https://www.oreilly.com/library/view/windows-powershell-cookbook/9781449359195/ch01.html"""
        if terminal is None: terminal = ""  # this means that cmd is the default console. alternative is "wt"
        if shell is None: shell = "" if self.machine == "win32" else ""  # other options are "powershell" and "cmd". # if terminal is wt, then it will pick powershell by default anyway.
        new_window = "start" if new_window is True else ""  # start is alias for Start-Process which launches a new window.  adding `start` to the begining of the command results in launching a new console that will not inherit from the console python was launched from e.g. conda
        extra, my_list = ("-Command" if shell in {"powershell", "pwsh"} else ""), list(cmds)
        if self.machine == "win32": my_list = [new_window, terminal, shell, extra] + my_list  # having a list is equivalent to: start "ipython -i file.py". Thus, arguments of ipython go to ipython, not start.
        print("Meta.Terminal.run_async: Subprocess command: ", my_list := [item for item in my_list if item != ""])
        return subprocess.Popen(my_list, stdin=subprocess.PIPE, shell=True)  # stdout=self.stdout, stderr=self.stderr, stdin=self.stdin. # returns Popen object, not so useful for communcation with an opened terminal
    @staticmethod
    def run_py(script, wdir=None, interactive=True, ipython=True, shell=None, delete=False, terminal="", new_window=True, header=True):  # async run, since sync run is meaningless.
        script = ((Terminal.get_header(wdir=wdir) if header else "") + script) + ("\ntb.DisplayData.set_pandas_auto_width()\n" if terminal in {"wt", "powershell", "pwsh"} else "")
        file = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmp_scripts").write_text(f"""print(r'''{script}''')""" + "\n" + script); print(f"Script to be executed asyncronously: ", file.absolute().as_uri())
        Terminal().run_async(f"{'ipython' if ipython else 'python'}", f"{'-i' if interactive else ''}", f"{file}", terminal=terminal, shell=shell, new_window=new_window)  # python will use the same dir as the one from console this method is called.
        _ = delete  # we need to ensure that async process finished reading before deleteing: file.delete(sure=delete, verbose=False)
    pickle_to_new_session = staticmethod(lambda obj, cmd="": Terminal.run_py(f"""path = tb.P(r'{Save.pickle(obj=obj, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}')\n obj = path.readit()\npath.delete(sure=True, verbose=False)\n {cmd}"""))
    @staticmethod
    def import_to_new_session(func, cmd="", header=True, interactive=True, ipython=True, **kwargs):
        load_kwargs_string = f"""loaded_kwargs = tb.P(r'{Save.pickle(obj=kwargs, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}').readit()\nloaded_kwargs.print()\nobj(**loaded_kwargs)""" if kwargs is not {} else ""
        if func.__name__ != func.__qualname__:  # it is a method of a class, must be instantiated first.
            module = P(sys.modules['__main__'].__file__).rel2cwd().stem if (module := func.__module__) == "__main__" else module
            load_func_string = f"import {module} as m\ninst=m.{func.__qualname__.split('.')[0]}()\nobj = inst.{func.__name__}"
        else:  # it is a standalone function...
            module = P(func.__code__.co_filename)  # module = func.__module__  # fails if the function comes from main as it returns __main__.
            load_func_string = f"tb.sys.path.insert(0, r'{module.parent}')\nimport {module.stem} as m\nobj=m.{func.__name__}"
        return Terminal.run_py(load_func_string + f"\n{cmd}\n{load_kwargs_string}\n", header=header, interactive=interactive, ipython=ipython)  # Terminal().run_async("python", "-c", load_func_string + f"\n{cmd}\n{load_kwargs_string}\n")
    @staticmethod
    def replicate_session(cmd=""): __import__("dill").dump_session(file := P.tmpfile(suffix=".pkl"), main=sys.modules[__name__]); Terminal().run_py(script=f"""path = tb.P(r'{file}'); tb.dill.load_session(str(path)); path.delete(sure=True, verbose=False); {cmd}""".replace("; ", "\n"))
    @staticmethod
    def get_header(wdir=None): return f"""\n# {'Code prepended'.center(1, '=')}\nimport crocodile.toolbox as tb""" + (f"""\ntb.sys.path.insert(0, r'{wdir}')""" if wdir is not None else '') + f"""\n# {'End of header, start of script passed'.center(1, '=')}\n"""


class SSH(object):  # if remote is Windows, this class assumed default shell in pwsh, as opposed to cmd
    def __init__(self, username, hostname=None, sshkey=None, pwd=None, env="ve"):  # https://stackoverflow.com/questions/51027192/execute-command-script-using-different-shell-in-ssh-paramiko
        _ = False; super().__init__() if _ else None; username, hostname = username.split("@") if "@" in username else (username, hostname)
        self.sshkey = str(sshkey) if sshkey is not None else None  # no need to pass sshkey if it was configured properly already
        self.ssh = (paramiko := __import__("paramiko")).SSHClient(); self.ssh.load_system_host_keys(); self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=hostname, username=username, password=pwd, port=22, key_filename=self.sshkey)
        self.hostname, self.username, self.platform = hostname, username, __import__("platform")
        try: self.sftp = self.ssh.open_sftp()
        except: self.sftp = None; print(f"WARNING: could not open SFTP connection to {hostname}. No data transfer is possible.")
        self.remote_machine = "Windows" if self.run("$env:OS", verbose=False).capture().op == "Windows_NT" else "Linux"  # echo %OS%
        self.remote_env_cmd = rf"""~/venvs/{env}/Scripts/Activate.ps1""" if self.remote_machine == "Windows" else rf"""source ~/venvs/{env}/bin/activate"""
        self.local_env_cmd = rf"""~/venvs/{env}/Scripts/Activate.ps1""" if self.platform.system() == "Windows" else rf"""source ~/venvs/{env}/bin/activate"""  # works for both cmd and pwsh
    def restart_computer(self): self.run("Restart-Computer -Force" if self.remote_machine == "Windows" else "sudo reboot")
    def send_ssh_key(self): self.copy_from_here("~/.ssh/id_rsa.pub"); assert self.remote_machine == "Windows"; self.run(P(install_n_import("machineconfig").scripts.windows.__path__.__dict__["_path"][0]).joinpath("openssh_server_add_sshkey.ps1").read_text())
    def __repr__(self): return f"local {self.get_repr('local')} [{self.platform.system()}] >>>>>>>>> SSH TO >>>>>>>>> remote {self.get_repr('remote')} [{self.remote_machine}] "
    def get_repr(self, which="remote"): return (f"{self.username}@{self.hostname}" if which == "remote" else f"{__import__('os').getlogin()}@{self.platform.node()}")
    def open_console(self, cmd='', new_window=True, terminal=None): Terminal().run_async("ssh", f"-i {self.sshkey}" if self.sshkey else "", '-t' if cmd!='' else '', *f""" {self.get_repr('remote')}""".split(" "), cmd, new_window=new_window, terminal=terminal)
    def run(self, cmd, verbose=True, desc="", strict_err=False, strict_returncode=False, env_prefix=False):
        cmd = (self.remote_env_cmd + "; " + cmd) if env_prefix else cmd; res = Terminal.Response(stdin=(raw := self.ssh.exec_command(cmd))[0], stdout=raw[1], stderr=raw[2], cmd=cmd)
        if strict_err or strict_returncode: assert res.is_successful(strict_err=strict_err, strict_returcode=strict_returncode), res.print(desc=desc)
        res.print(desc=desc) if verbose else None; return res
    def run_py(self, cmd, desc="", return_obj=False, verbose=True, strict_err=False, strict_returncode=False):
        assert '"' not in cmd, f'Avoid using `"` in your command. I dont know how to handle this when passing is as command to python in pwsh command.'
        if not return_obj: return self.run(f"""{self.remote_env_cmd}; python -c "{Terminal.get_header(wdir=None)}{cmd}\n""" + '"', desc=desc or f"run_py on {self.get_repr('remote')}", verbose=verbose, strict_err=strict_err, strict_returncode=strict_returncode)
        else: assert "obj=" in cmd, f"The command sent to run_py must have `obj=` statement if return_obj is set to True"; source_file = self.run_py(f"""{cmd}\npath = tb.Save.pickle(obj=obj, path=tb.P.tmpfile())\nprint(path)""", desc=desc, verbose=verbose, strict_err=True, strict_returncode=True).op.split('\n')[-1]; return self.copy_to_here(source=source_file, target=P.tmpfile(suffix='.pkl')).readit()
    def run_locally(self, command): print(f"Executing Locally @ {self.platform.node()}:\n{command}"); return Terminal.Response(__import__('os').system(command))
    def copy_from_here(self, source, target=None, zip_first=False, r=False, overwrite=False):
        if not zip_first and (source := P(source).expanduser()).is_dir(): return source.search("*", folders=False, r=True).apply(lambda file: self.copy_from_here(source=file, target=target)) if r is True else print(f"source is a directory! either set r=True for recursive sending or raise zip_first flag.")
        if zip_first: print(f"ZIPPING ..."); source = P(source).expanduser().zip(content=True)  # .append(f"_{randstr()}", inplace=True)  # eventually, unzip will raise content flag, so this name doesn't matter.
        if target is None: target = P(source).collapseuser(); assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home."
        remotepath = self.run_py(f"path=tb.P(r'{P(target).as_posix()}').expanduser()\n{'path.delete(sure=True)' if overwrite else ''}\nprint(path.parent.create())", desc=f"Creating Target directory `{P(target).parent.as_posix()}` @ {self.get_repr('remote')}").op or ''; remotepath=P(remotepath.split("\n")[-1]).joinpath(P(target).name)
        print(f"SENDING `{P(source)}` ==> `{remotepath.as_posix()}`"); self.sftp.put(localpath=P(source).expanduser(), remotepath=remotepath.as_posix()); print(f"SENDING COMPLETED", "\n" * 2)
        if zip_first: resp = self.run_py(f"""tb.P(r'{remotepath.as_posix()}').expanduser().unzip(content=False, inplace=True, overwrite={overwrite})""", desc=f"UNZIPPING"); source.delete(sure=True); return resp
    def copy_to_here(self, source, target=None, zip_first=False, r=False):
        if not zip_first and self.run_py(f"print(tb.P(r'{source}').expanduser().is_dir())", desc="Check if source is a dir", verbose=True, strict_returncode=True, strict_err=True).op.split("\n")[-1] == 'True':
            return self.run_py(f"obj=tb.P(r'{source}').search(folders=False, r=True)", desc="Searching for files in source", return_obj=True).apply(lambda file: self.copy_to_here(source=file, target=P(target).joinpath(P(file).relative_to(source)) if target else None, r=False)) if r else print(f"source is a directory! either set r=True for recursive sending or raise zip_first flag.")
        if zip_first: source = self.run_py(f"print(tb.P(r'{source}').expanduser().zip(inplace=False, verbose=False))", desc=f"Zipping source file", strict_returncode=True, strict_err=True).as_path
        if target is None: target = self.run_py(f"print(tb.P(r'{P(source).as_posix()}').collapseuser())", desc=f"Finding default target via relative source path", strict_returncode=True, strict_err=True).as_path; assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home."
        target = P(target).expanduser().create(parents_only=True); target += '.zip' if zip_first and '.zip' not in target.suffix else ''
        source = self.run_py(f"print(tb.P(r'{source}').expanduser())", desc=f"# Resolving source path address by expanding user", strict_returncode=True, strict_err=True).as_path if "~" in str(source) else P(source)
        print(f"RECEVING `{source}` ==> `{target}`"); self.sftp.get(remotepath=source.as_posix(), localpath=target.as_posix()); print(f"RECEVING COMPLETED", "\n" * 2)
        if zip_first: target = target.unzip(inplace=True, content=True); self.run_py(f"tb.P(r'{source}').delete(sure=True)", desc="Cleaning temp files", strict_returncode=True, strict_err=True)
        return target
    def copy_env_var(self, name): assert self.remote_machine == "Linux"; self.run(f"{name} = {__import__('os').environ[name]}; export {name}")


class Scheduler:
    def __init__(self, routine=None, wait: str = "2m", other_routine=None, other_ratio: int = 10, max_cycles=float("inf"), exception_handler=None, logger: Log = None, sess_stats=None):
        self.routine = (lambda sched: None) if routine is None else routine  # main routine to be repeated every `wait` time period
        self.other_routine = (lambda sched: None) if other_routine is None else other_routine  # routine to be repeated every `other` time period
        self.wait, self.other_ratio = str2timedelta(wait).total_seconds(), other_ratio  # wait period between routine cycles.
        self.logger, self.exception_handler = logger or Log(name="SchedLogger_" + randstr(length=2)), exception_handler
        self.sess_start_time, self.records, self.cycle, self.max_cycles, self.sess_stats = None, List([]), 0, max_cycles, sess_stats or (lambda sched: {})
    def run(self, max_cycles=None, until="2050-01-01"):
        self.max_cycles, self.cycle, self.sess_start_time = max_cycles or self.max_cycles, 0, datetime.now()
        while datetime.now() < datetime.fromisoformat(until) and self.cycle < self.max_cycles:  # 1- Time before Ops, and Opening Message
            time1 = datetime.now(); self.logger.warning(f"Starting Cycle {str(self.cycle).zfill(5)}. Total Run Time = {str(datetime.now() - self.sess_start_time)}. UTC {datetime.utcnow().strftime('%d %H:%M:%S')}")
            try: self.routine(sched=self)
            except BaseException as ex: self._handle_exceptions(ex=ex, during="routine")  # 2- Perform logic
            if self.cycle % self.other_ratio == 0:
                try: self.other_routine(sched=self)
                except BaseException as ex: self._handle_exceptions(ex=ex, during="occasional")  # 3- Optional logic every while
            time_left = int(self.wait - (datetime.now() - time1).total_seconds())  # 4- Conclude Message
            self.cycle += 1; self.logger.warning(f"Finishing Cycle {str(self.cycle - 1).zfill(5)}. Sleeping for {self.wait}s ({time_left}s left)\n" + "-" * 100)
            try: time.sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex: self._handle_exceptions(ex, during="sleep")  # that's probably the only kind of exception that can rise during sleep.
        else: self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until})"); return self
    def history(self): return __import__("pandas").DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "termination reason", "logfile"] + list(self.sess_stats(sched=self).keys()))
    def record_session_end(self, reason="Not passed to function."):
        self.records.append([self.sess_start_time, end_time := datetime.now(), duration := end_time-self.sess_start_time, self.cycle, reason, self.logger.file_path] + list((sess_stats := self.sess_stats(sched=self)).values()))
        summ = {"start time": f"{str(self.sess_start_time)}", "finish time": f"{str(end_time)}.", "duration": f"{str(duration)} | wait time {self.wait}s", "cycles ran": f"{self.cycle} | Lifetime cycles = {self.history()['cycles'].sum()}", f"termination reason": reason, "logfile": self.logger.file_path}
        self.logger.critical(f"\n--> Scheduler has finished running a session. \n" + Struct(summ).update(sess_stats).print(as_config=True, return_str=True, quotes=False) + "\n" + "-" * 100); self.logger.critical(f"\n--> Logger history.\n"+str(self.history())); return self
    def _handle_exceptions(self, ex, during):
        if self.exception_handler is not None: self.exception_handler(ex, during=during, sched=self)  # user decides on handling and continue, terminate, save checkpoint, etc.  # Use signal library.
        else: self.record_session_end(reason=f"during {during}, " + str(ex)); raise ex


def try_this(func, return_=None, raise_=None, run=None, handle=None, **kwargs):
    try: return func()
    except BaseException as ex:  # or Exception
        if raise_ is not None: raise raise_
        if handle is not None: return handle(ex, **kwargs)
        return run() if run is not None else return_
def show_globals(scope, **kwargs): return Struct(scope).filter(lambda k, v: "__" not in k and not k.startswith("_") and k not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}).print(**kwargs)
def monkey_patch(class_inst, func): setattr(class_inst.__class__, func.__name__, func)
def capture_locals(func, scope, args=None, self: str = None, update_scope=True): res = dict(); exec(extract_code(func, args=args, self=self, include_args=True, verbose=False), scope, res); scope.update(res) if update_scope else None; return Struct(res)
def generate_readme(path, obj=None, meta=None, save_source_code=True, verbose=True):  # Generates a readme file to contextualize any binary files by mentioning module, class, method or function used to generate the data"""
    text = "# Meta\n" + (meta if meta is not None else '') + (separator := "\n" + "-----" + "\n\n")
    text += (f"# Code to generate the result\n```python\n" + __import__("inspect").getsource(obj) + "\n```" + separator) if obj is not None else ""
    text += (f"# Source code file generated me was located here: \n`{__import__('inspect').getfile(obj)}`\n" + separator) if obj is not None else ""
    if (res := Terminal().run("echo '## Last Commit'; git log -1; echo '## Remote Repo:'; git remote -v", shell="pwsh")).is_successful(strict_err=True, strict_returcode=True):
        text += res.op + "\nlink to files: " + res.op.split("## Remote Re")[1].split("\n")[1].split("\t")[1].split(" ")[0].replace(".git", "") + f"/tree/" + res.op.split('commit ')[1].split('\n')[0]
    readmepath = (P(path) / f"README.md" if P(path).is_dir() else P(path)).write_text(text); print(f"SAVED README.md @ {readmepath.absolute().as_uri()}") if verbose else None
    if save_source_code: P((obj.__code__.co_filename if hasattr(obj, "__code__") else None) or __import__("inspect").getmodule(obj).__file__).zip(path=readmepath.with_name("source_code.zip"), verbose=False); print("SAVED source code @ " + readmepath.with_name("source_code.zip").absolute().as_uri())
def load_from_source_code(directory, obj=None, delete=False):
    P(directory).find("source_code*", r=True).unzip(tmpdir := P.tmp() / timestamp(name="tmp_sourcecode"))
    sys.path.insert(0, str(tmpdir)); sourcefile = __import__(tmpdir.find("*").stem); tmpdir.delete(sure=delete, verbose=False)
    return getattr(sourcefile, obj) if obj is not None else sourcefile
def extract_code(func, code: str = None, include_args=True, verbose=True, copy2clipboard=False, **kwargs):  # TODO: how to handle decorated functions.  add support for lambda functions.  ==> use dill for powerfull inspection"""
    import inspect; import textwrap  # assumptions: first line could be @classmethod or @staticmethod. second line could be def(...). Then function body must come in subsequent lines, otherwise ignored.
    raw = inspect.getsourcelines(func)[0]; lines = textwrap.dedent("".join(raw[1 + (1 if raw[0].lstrip().startswith("@") else 0):])).split("\n")
    code_string = '\n'.join([aline if not textwrap.dedent(aline).startswith("return ") else aline.replace("return ", "return_ = ") for aline in lines])  # remove return statements if there else keep line as is.
    title, args_header, injection_header, body_header, suffix = ((f"\n# " + f"{item} {func.__name__}".center(50, "=") + "\n") for item in ["CODE EXTRACTED FROM", "ARGS KWARGS OF", "INJECTED CODE INTO", "BODY OF", "BODY END OF"])
    code_string = title + ((args_header + extract_arguments(func, **kwargs)) if include_args else '') + ((injection_header + code) if code is not None else '') + body_header + code_string + suffix  # added later so it has more overwrite authority.
    install_n_import("clipboard").copy(code_string) if copy2clipboard else None; print(code_string) if verbose else None; return code_string  # ready to be run with exec()
def extract_arguments(func, copy2clipboard=False, **kwargs):
    ak = Struct(dict((inspect := __import__("inspect")).signature(func).parameters)).values()  # ignores self for methods automatically but also ignores args and kwargs.
    res = Struct.from_keys_values(ak.name, ak.default).update(kwargs).print(as_config=True, return_str=True, justify=0, quotes=True).replace("<class 'inspect._empty'>", "None").replace("= '", "= rf'")
    ak = inspect.getfullargspec(func); res = res + (f"{ak.varargs} = (,)\n" if ak.varargs else '') + (f"{ak.varkw} = " + "{}\n" if ak.varkw else '')  # add args = () and kwargs = {}
    install_n_import("clipboard").copy(res) if copy2clipboard else None; return res
def run_cell(pointer, module=sys.modules[__name__]):
    for cell in P(module.__file__).read_text().split("#%%"):
        if pointer in cell.split('\n')[0]: break  # bingo
    else: raise KeyError(f"The pointer `{pointer}` was not found in the module `{module}`")
    print(cell); install_n_import("clipboard").copy(cell); return cell
class Experimental: try_this = try_this; show_globals = show_globals; monkey_patch = monkey_patch; capture_locals = capture_locals; generate_readme = generate_readme; load_from_source_code = load_from_source_code; extract_code = extract_code; extract_arguments = extract_arguments; run_cell = run_cell  # Debugging and Meta programming tools"""


if __name__ == '__main__':
    pass
