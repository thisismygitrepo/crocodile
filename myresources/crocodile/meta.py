

from crocodile.core import timestamp, randstr, str2timedelta, Save, install_n_import, List, Struct
from crocodile.file_management import P, datetime
import logging
import subprocess
import sys


class Null:
    def __getattr__(self, item): _ = item; return self
    def __getitem__(self, item): _ = item; return self
    def __call__(self, *args, **kwargs): return self
    def __len__(self): return 0
    def __bool__(self): return False
    def __contains__(self, item): _ = item; return False
    def __iter__(self): return iter([self])


class Log(object):  #
    def __init__(self, dialect=["colorlog", "logging", "coloredlogs"][0], name=None, file: bool = False, file_path=None, stream=True, fmt=None, sep=" | ",
                 s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False, log_colors=None):
        self.specs = dict(name=name, file=file, file_path=file_path, stream=stream, fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level)  # save speces that are essential to re-create the object at
        self.dialect = dialect  # specific to this class
        self.verbose = verbose  # specific to coloredlogs dialect
        self.log_colors = log_colors  # specific kwarg to colorlog dialect
        self._install()  # update specs after intallation.
        self.specs["path"] = self.logger.name
        if file: self.specs["file_path"] = self.logger.handlers[0].baseFilename  # first handler is a file handler
    def __getattr__(self, item): return getattr(self.logger, item)  # makes it twice as slower as direct access 300 ns vs 600 ns
    def debug(self, msg): return self.logger.debug(msg)  # to speed up the process and avoid falling back to __getattr__
    def info(self, msg): return self.logger.info(msg)
    def warn(self, msg): return self.logger.warn(msg)
    def error(self, msg): return self.logger.error(msg)
    def critical(self, msg): return self.logger.critical(msg)
    file = property(lambda self: P(self.specs["file_path"]) if self.specs["file_path"] else None)
    def close(self): raise NotImplementedError
    def get_shandler(self, first=True): shandlers = List(handler for handler in self.logger.handlers if "StreamHandler" in str(handler)); return shandlers[0] if first else shandlers
    def get_fhandler(self, first=True): fhandlers = List(handler for handler in self.logger.handlers if "FileHandler" in str(handler)); return fhandlers[0] if first else fhandlers
    def set_level(self, level, which=["logger", "stream", "file", "all"][0]): self.logger.setLevel(level) if which in {"logger", "all"} else None; self.get_shandler().setLevel(level) if which in {"stream", "all"} else None; self.get_fhandler().setLevel(level) if which in {"file", "all"} else None
    def _install(self):  # populates self.logger attribute according to specs and dielect.
        if self.specs["file"] is False and self.specs["stream"] is False: self.logger = Null()
        elif self.dialect == "colorlog": self.logger = Log.get_colorlog(log_colors=self.log_colors, **self.specs)
        elif self.dialect == "logging": self.logger = Log.get_logger(**self.specs)
        elif self.dialect == "coloredlogs": self.logger = Log.get_coloredlogs(verbose=self.verbose, **self.specs)
        else: self.logger = Log.get_colorlog(**self.specs)
    def __setstate__(self, state):
        self.__dict__ = state   # this way of creating relative path makes transferrable across machines.
        self.specs["file_path"] = P(self.specs["file_path"]).rel2home() if self.specs["file_path"] is not None else None; self._install()
    def __getstate__(self):  # logger can be pickled without this method, but its handlers are lost, so what's the point? no perfect reconstruction.
        state = self.__dict__.copy(); state["specs"] = state["specs"].copy(); del state["logger"]
        state["specs"]["file_path"] = P(self.specs["file_path"]).expanduser() if self.specs["file_path"] is not None else None; return state
    def __repr__(self): return "".join([f"{self.logger} with handlers: \n"] + [repr(h) + "\n" for h in self.logger.handlers])
    get_basic_format = staticmethod(lambda: logging.BASIC_FORMAT)
    get_format = staticmethod(lambda sep: f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s{sep}%(levelno)s{sep}%(message)s{sep}")  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes
    test_all = staticmethod(lambda: [Log.test_logger(logger) if not print("=" * 100) else None for logger in [Log.get_logger(), Log.get_colorlog(), Log.get_coloredlogs()]])
    @staticmethod
    def manual_degug(path): sys.stdout = open(path, 'w'); sys.stdout.close(); print(f"Finished ... have a look @ \n {path}")  # all print statements will write to this file.
    @staticmethod
    def get_coloredlogs(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False):
        # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        level_styles = {'spam': {'color': 'green', 'faint': True}, 'debug': {'color': 'white'}, 'verbose': {'color': 'blue'}, 'info': {'color': "green"}, 'notice': {'color': 'magenta'}, 'warning': {'color': 'yellow'}, 'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True, "underline": True}, 'critical': {'color': 'red', 'bold': True, "inverse": False}}
        field_styles = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'color': 'black', 'bold': True}, 'path': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}}
        coloredlogs = install_n_import("coloredlogs")
        if verbose: logger = install_n_import("verboselogs").VerboseLogger(name=name); logger.setLevel(l_level)  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
        else: logger = Log.get_base_logger(logging, name=name, l_level=l_level); Log.add_handlers(logger, module=logging, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
        coloredlogs.install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger
    @staticmethod
    def get_colorlog(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, log_colors=None, ):
        log_colors = log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'thin_red', 'CRITICAL': 'fg_bold_red', }  # see here for format: https://pypi.org/project/colorlog/
        colorlog = install_n_import("colorlog"); logger = Log.get_base_logger(colorlog, name, l_level)
        Log.add_handlers(logger, colorlog, file, f_level, file_path, colorlog.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), log_colors=log_colors), stream, s_level); return logger
    @staticmethod
    def get_logger(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG):  # Basic Python logger."""
        logger = Log.get_base_logger(logging, name, l_level); Log.add_handlers(logger, logging, file, f_level, file_path, logging.Formatter(fmt or Log.get_format(sep)), stream, s_level); return logger
    @staticmethod
    def get_base_logger(module, name, l_level):
        if name is None: print(f"Logger name not passed. It is preferable to pass a name indicating the owner.")
        else: print(f"Logger `{name}` from `{module.__name__}` is instantiated with level {l_level}.")
        logger = module.getLogger(name=name or randstr()); logger.setLevel(level=l_level)  # logs everything, finer level of control is given to its handlers
        return logger  # https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    @staticmethod
    def add_handlers(logger, module, file, f_level, file_path, fmt, stream, s_level):
        if file or file_path:  Log.add_filehandler(logger, file_path=file_path, fmt=fmt, f_level=f_level)  # create file handler for the logger.
        if stream: Log.add_streamhandler(logger, s_level, fmt, module=module)  # ==> create stream handler for the logger.
    @staticmethod
    def add_streamhandler(logger, s_level=logging.DEBUG, fmt=None, module=logging, name="myStream"):
        shandler = module.StreamHandler(); shandler.setLevel(level=s_level); shandler.setFormatter(fmt=fmt); shandler.set_name(name); logger.addHandler(shandler); print(f"    Level {s_level} stream handler for Logger `{logger.name}` is created.")
    @staticmethod
    def add_filehandler(logger, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="myFileHandler"):
        if file_path is None: file_path = P.tmpfile(name="logger", suffix=".log", folder="tmp_loggers")
        fhandler = logging.FileHandler(filename=str(file_path), mode=mode); fhandler.setFormatter(fmt=fmt); fhandler.setLevel(level=f_level); fhandler.set_name(name); logger.addHandler(fhandler)
        print(f"    Level {f_level} file handler for Logger `{logger.name}` is created @ " + P(file_path).clickable())
    @staticmethod
    def test_logger(logger):
        logger.debug("this is a debugging message"); logger.info("this is an informational message"); logger.warning("this is a warning message")
        logger.error("this is an error message"); logger.critical("this is a critical message"); [logger.log(msg=f"This is a message of level {level}", level=level) for level in range(0, 60, 5)]


class Terminal:
    class Response:
        @staticmethod
        def from_completed_process(cp: subprocess.CompletedProcess): (resp := Terminal.Response(cmd=cp.args)).output.update(dict(stdout=cp.stdout, stderr=cp.stderr, returncode=cp.returncode)); return resp
        def __init__(self, stdin=None, stdout=None, stderr=None, cmd=None): self.std, self.output, self.input = dict(stdin=stdin, stdout=stdout, stderr=stderr), dict(stdin="", stdout="", stderr="", returncode=None), cmd  # input command
        def __call__(self, *args, **kwargs): return self.op.rstrip() if type(self.op) is str else None
        op = property(lambda self: self.output["stdout"])
        ip = property(lambda self: self.output["stdin"])
        err = property(lambda self: self.output["stderr"])
        success = property(lambda self: self.output["returncode"] == 0)
        returncode = property(lambda self: self.output["returncode"])
        as_path = property(lambda self: P(self.op.rstrip()) if self.err == "" else None)
        def capture(self): [self.output.__setitem__(key, val.read().decode().rstrip()) for key, val in self.std.items() if val is not None and val.readable()]; return self
        def print(self): self.capture(); print(f"Terminal Response:\nInput Command: {self.input}\n" + "\n".join([f"{f' {idx} - {key} '}".center(40, "-") + f"\n{val}" for idx, (key, val) in enumerate(self.output.items())]) + "\n" + "=" * 50, "\n\n"); return self
    def __init__(self, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, elevated=False):
        self.available_consoles = ["cmd", "Command Prompt", "wt", "powershell", "wsl", "ubuntu", "pwsh"]
        self.elevated, self.stdout, self.stderr, self.stdin = elevated, stdout, stderr, stdin
        self.machine = sys.platform  # 'win32', 'linux' OR: import platform; self.platform.system(): Windows, Linux, Darwin
    def set_std_system(self): self.stdout = sys.stdout; self.stderr = sys.stderr; self.stdin = sys.stdin
    def set_std_pipe(self): self.stdout = subprocess.PIPE; self.stderr = subprocess.PIPE; self.stdin = subprocess.PIPE
    def set_std_null(self): self.stdout, self.stderr, self.stdin = subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL  # Equivalent to `echo 'foo' &> /dev/null`
    @staticmethod
    def is_admin(): return Experimental.try_this(lambda: __import__("ctypes").windll.shell32.IsUserAnAdmin(), return_=False)  # https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script
    def run(self, *cmds, shell=None, check=False, ip=None):
        """Blocking operation. Thus, if you start a shell via this method, it will run in the main an won't stop until you exit manually IF stdin is set to sys.stdin, otherwise it will run and close quickly. Other combinations of stdin, stdout can lead to funny behaviour like no output but accept input or opposite.
        * This method is short for: res = subprocess.run("powershell command", capture_output=True, shell=True, text=True) and unlike os.system(cmd), subprocess.run(cmd) gives much more control over the output and input.
        * `shell=True` loads up the profile of the shell called so more specific commands can be run. Importantly, on Windows, the `start` command becomes availalbe and new windows can be launched."""
        my_list = list(cmds)  # `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job. `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task. `suprocess.call` is an archaic command for pre-Python-3.5.
        if self.machine == "win32" and shell in {"powershell", "pwsh"}: my_list = [shell, "-Command"] + my_list  # alternatively, one can run "cmd"
        if self.elevated is False or self.is_admin(): resp = subprocess.run(my_list, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=check, input=ip)
        else: resp = __import__("ctypes").windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        """ The advantage of addig `powershell -Command` is to give access to wider range of options. Other wise, command prompt shell doesn't recognize commands like `ls`.
        `capture_output` prevents the stdout to redirect to the stdout of the script automatically, instead it will be stored in the Response object returned. # `capture_output=True` same as `stdout=subprocess.PIPE, stderr=subprocess.PIPE`"""
        return self.Response.from_completed_process(resp)
    def run_async(self, *cmds, new_window=True, shell=None, terminal=None):
        """Opens a new terminal, and let it run asynchronously. Maintaining an ongoing conversation with another process is very hard. It is adviseable to run all
        commands in one go without interaction with an ongoing channel. Use this only for the purpose of producing a different window and humanly interact with it.
        https://stackoverflow.com/questions/54060274/dynamic-communication-between-main-and-subprocess-in-python & https://www.youtube.com/watch?v=IynV6Y80vws and https://www.oreilly.com/library/view/windows-powershell-cookbook/9781449359195/ch01.html"""
        if terminal is None: terminal = ""  # this means that cmd is the default console. alternative is "wt"
        if shell is None: shell = "" if self.machine == "win32" else ""  # other options are "powershell" and "cmd". # if terminal is wt, then it will pick powershell by default anyway.
        new_window = "start" if new_window is True else ""  # start is alias for Start-Process which launches a new window.  adding `start` to the begining of the command results in launching a new console that will not inherit from the console python was launched from e.g. conda
        extra, my_list = ("-Command" if shell in {"powershell", "pwsh"} else ""), list(cmds)
        if self.machine == "win32": my_list = [new_window, terminal, shell, extra] + my_list  # having a list is equivalent to: start "ipython -i file.py". Thus, arguments of ipython go to ipython, not start.
        print("Meta.Terminal.run_async: Subprocess command: ", my_list := [item for item in my_list if item != ""])
        return subprocess.Popen(my_list, stdin=subprocess.PIPE, shell=True)  # stdout=self.stdout, stderr=self.stderr, stdin=self.stdin. # returns Popen object, not so useful for communcation with an opened terminal
    @staticmethod
    def run_script(script, wdir=None, interactive=True, ipython=True, shell=None, delete=False, terminal="", new_window=True, header=True):
        """This method is a wrapper on top of `run_async" except that the command passed will launch python terminal that will run script passed by user. """
        header_script = f"""\n# {'Code prepended by Terminal.run_script'.center(80, '=')}; import crocodile.toolbox as tb; tb.sys.path.insert(0, r'{wdir or P.cwd()}'); # {'End of header, start of script passed'.center(80, '=')}\n""".replace("; ", "\n")  # this header is necessary so import statements in the script passed are identified relevant to wdir.
        script = (header_script + script if header else script) + ("\ntb.DisplayData.set_pandas_auto_width()\n" if terminal in {"wt", "powershell", "pwsh"} else "")
        file = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmp_scripts").write_text(f"""print(r'''{script}''')""" + "\n" + script); print(f"Script to be executed asyncronously: ", file.absolute().as_uri())
        Terminal().run_async(f"{'ipython' if ipython else 'python'}", f"{'-i' if interactive else ''}", f"{file}", terminal=terminal, shell=shell, new_window=new_window)  # python will use the same dir as the one from console this method is called.
        _ = delete  # we need to ensure that async process finished reading before deleteing: file.delete(sure=delete, verbose=False)
    replicate_in_new_session = staticmethod(lambda obj, cmd="": Terminal.run_script(f"""path = tb.P(r'{Save.pickle(obj=obj, path=P.tmpfile(tstamp=False, suffix=".pkl"), verbose=False)}'); obj = path.readit(); path.delete(sure=True, verbose=False); {cmd}""".replace("; ", "\n")))
    @staticmethod
    def replicate_session(cmd=""): __import__("dill").dump_session(file := P.tmpfile(suffix=".pkl"), main=sys.modules[__name__]); Terminal().run_script(script=f"""path = tb.P(r'{file}'); tb.dill.load_session(str(path)); path.delete(sure=True, verbose=False); {cmd}""".replace("; ", "\n"))
    @staticmethod
    def is_user_admin():  # adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows"""
        if __import__('os').name == 'nt':
            import ctypes
            try: return ctypes.windll.shell32.IsUserAnAdmin()
            except: import traceback; traceback.print_exc(); print("Admin check failed, assuming not an admin."); return False
        else: return __import__('os').getuid() == 0  # Check for root on Posix
    @staticmethod
    def run_code_as_admin(params):
        _ = install_n_import("win32api", name="pypiwin32")
        win32com = __import__("win32com", fromlist=["shell.shell.ShellExecuteEx"])
        win32com.shell.shell.ShellExecuteEx(lpVerb='runas', lpFile=sys.executable, lpParameters=params)
    @staticmethod
    def run_as_admin(cmd_line=None, wait=True):
        """Attempt to relaunch the current script as an admin using the same command line parameters.  Pass cmdLine in to override and set a new command.  It must be a list of [command, arg1, arg2...] format.
        Set wait to False to avoid waiting for the sub-process to finish. You will not be able to fetch the exit code of the process if wait is False.
        Returns the sub-process return code, unless wait is False in which case it returns None. adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows"""
        if __import__('os').name != 'nt': raise RuntimeError("This function is only implemented on Windows.")
        _ = install_n_import("win32api", name="pypiwin32")
        win32event, win32process = install_n_import("win32event"), install_n_import("win32process")
        win32com = __import__("win32com", fromlist=["shell.shell.ShellExecuteEx"])
        if cmd_line is None: cmd_line = [sys.executable] + sys.argv
        elif type(cmd_line) not in (tuple, list): raise ValueError("cmdLine is not a sequence.")
        cmd, params = '"%s"' % (cmd_line[0],), " ".join(['"%s"' % (x,) for x in cmd_line[1:]])
        proce_info = win32com.shell.shell.ShellExecuteEx(nShow=__import__("win32con").SW_SHOWNORMAL, fMask=__import__("win32com", fromlist=["shell.shellcon"]).shell.shellcon.SEE_MASK_NOCLOSEPROCESS, lpVerb='runas', lpFile=cmd, lpParameters=params)  # causes UAC elevation prompt.
        if wait: proc_handle = proce_info['hProcess']; _ = win32event.WaitForSingleObject(proc_handle, win32event.INFINITE); rc = win32process.GetExitCodeProcess(proc_handle)
        else: rc = None; return rc


class SSH(object):
    def __init__(self, username, hostname, sshkey=None, pwd=None):
        _ = False; super().__init__() if _ else None
        self.sshkey = str(sshkey) if sshkey is not None else None  # no need to pass sshkey if it was configured properly already
        self.ssh = (paramiko := __import__("paramiko")).SSHClient(); self.ssh.load_system_host_keys(); self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=hostname, username=username, password=pwd, port=22, key_filename=self.sshkey)
        self.hostname, self.username, self.sftp, self.platform = hostname, username, self.ssh.open_sftp(), __import__("platform")
        self.remote_machine = "Windows" if self.run("$env:OS", verbose=False).output["stdout"] == "Windows_NT" else "Linux"
        self.remote_python_cmd = rf"""~/venvs/ve/Scripts/activate""" if self.remote_machine == "Windows" else rf"""source ~/venvs/ve/bin/activate"""
        self.local_python_cmd = rf"""~/venvs/ve/Scripts/activate""" if self.platform.system() == "Windows" else rf"""source ~/venvs/ve/bin/activate"""  # works for both cmd and pwsh
    def get_key(self): return f"""-i "{str(P(self.sshkey).expanduser())}" """ if self.sshkey is not None else ""  # SSH cmd: scp -r {self.get_key()} "{str(source.expanduser())}" "{self.username}@{self.hostname}:'{target}'
    def __repr__(self): return f"{self.get_repr('local')} [{self.platform.system()}] SSH connection to {self.get_repr('remote')} [{self.remote_machine}] "
    def get_repr(self, which="remote"): return f"{self.username}@{self.hostname}" if which == "remote" else f"{__import__('os').getlogin()}@{self.platform.node()}"
    def open_console(self, new_window=True): Terminal().run_async(f"""ssh -i {self.sshkey} {self.username}@{self.hostname}""", new_window=new_window)
    def copy_env_var(self, name): assert self.remote_machine == "Linux"; self.run(f"{name} = {__import__('os').environ[name]}; export {name}")
    def copy_to_here(self, source, target=None): pass
    def runpy(self, cmd): return self.run(f"""{self.remote_python_cmd}; python -c 'import crocodile.toolbox as tb; {cmd} ' """)
    def run_locally(self, command): print(f"Executing Locally @ {self.platform.node()}:\n{command}"); return Terminal.Response(__import__('os').system(command))
    def run(self, cmd, verbose=True): res = Terminal.Response(stdin=(raw := self.ssh.exec_command(cmd))[0], stdout=raw[1], stderr=raw[2], cmd=cmd); res.print() if verbose else None; return res
    def copy_from_here(self, source, target=None, zip_n_encrypt=False):
        pwd = randstr(length=10, safe=True)
        if zip_n_encrypt: print(f"ZIPPING & ENCRYPTING".center(80, "=")); source = P(source).expanduser().zip_n_encrypt(pwd=pwd)
        if target is None: target = P(source).collapseuser(); print(target, P(source), P(source).collapseuser()); assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home."; target = target.as_posix()
        print("\n" * 3, f"Creating Target directory {target} @ remote machine.".center(80, "="))
        remotepath = P(self.runpy(f'print(tb.P(r"{target}").expanduser().parent.create())').op or '').joinpath(P(target).name).as_posix()
        print(f"SENT `{source}` ==> `{remotepath}`".center(80, "="), "\n" * 2)
        self.sftp.put(localpath=P(source).expanduser(), remotepath=remotepath)
        if zip_n_encrypt: print(f"UNZIPPING & DECRYPTING".center(80, "=")); resp = self.runpy(f"""tb.P(r"{remotepath}").expanduser().decrypt_n_unzip(pwd="{pwd}", inplace=True)"""); source.delete(sure=True); return resp


class Scheduler:
    def __init__(self, routine=None, wait: str = "2m", other_routine=None, other_ratio: int = 10, max_cycles=float("inf"), exception_handler=None, logger=None):
        self.routine = lambda: None if routine is None else routine  # main routine to be repeated every `wait` time period
        self.other_routine = lambda: None if other_routine is None else other_routine  # routine to be repeated every `other` time period
        self.wait, self.other_ratio = str2timedelta(wait).total_seconds(), other_ratio  # wait period between routine cycles.
        self.logger, self.exception_handler = logger or Log(name="SchedulerAutoLogger_" + randstr()), exception_handler
        self.sess_start_time, self.records, self.cycle, self.max_cycles = None, List([]), 0, max_cycles
    def run(self, until="2050-01-01", max_cycles=None):
        self.max_cycles, self.cycle, self.sess_start_time = max_cycles or self.max_cycles, 0, datetime.now()
        while datetime.now() < datetime.fromisoformat(until) and self.cycle < self.max_cycles:  # 1- Time before Ops, and Opening Message
            time1 = datetime.now(); self.logger.info(f"Starting Cycle {self.cycle: <5}. Total Run Time = {str(datetime.now() - self.sess_start_time)[:-7]: <10}. UTC Time: {datetime.utcnow().isoformat(timespec='minutes', sep=' ')}")
            Experimental.try_this(self.routine, handle=self._handle_exceptions, during="routine")  # 2- Perform logic
            if self.cycle % self.other_ratio == 0: Experimental.try_this(self.other_routine, handle=self._handle_exceptions, during="occasional")  # 3- Optional logic every while
            time_left = int(self.wait - (datetime.now() - time1).total_seconds())  # 4- Conclude Message
            self.cycle += 1; self.logger.info(f"Finishing Cycle {self.cycle - 1: <4}. Sleeping for {self.wait} seconds. ({time_left} seconds left)\n" + "-" * 50)
            try: __import__("time").sleep(time_left if time_left > 0 else 0.1)  # # 5- Sleep. consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex: self._handle_exceptions(ex, during="sleep")  # that's probably the only kind of exception that can rise during sleep.
        else: self.record_session_end(reason=f"Reached maximum number of cycles ({self.max_cycles})" if self.cycle >= self.max_cycles else f"Reached due stop time ({until})"); return self
    def history(self): return __import__("pandas").DataFrame.from_records(self.records, columns=["start", "finish", "duration", "cycles", "logfile"])
    def record_session_end(self, reason="Unknown"):
        self.records.append([self.sess_start_time, end_time := datetime.now(), duration := end_time-self.sess_start_time, self.cycle, self.logger.file])
        summ = {f"start time": f"{str(self.sess_start_time)}", f"finish time": f"{str(end_time)}.", f"duration": f"{str(duration)} | wait time {self.wait} seconds", f"cycles ran": f"{self.cycle} | Lifetime cycles = {self.history()['cycles'].sum()}", f"termination reason": f"{reason}"}
        self.logger.critical(f"\n--> Scheduler has finished running a session. \n" + Struct(summ).print(config=True, return_str=True, quotes=False) + "\n" + "-" * 100); return self
    def _handle_exceptions(self, ex, during):
        if self.exception_handler is not None: self.exception_handler(ex, during=during, sched=self)  # user decides on handling, terminate, save checkpoint, or continue, etc.  # Use signal library.
        else: self.record_session_end(reason=ex); raise ex


class Experimental:  # Debugging and Meta programming tools"""
    @staticmethod
    def try_this(func, return_=None, raise_=None, run=None, handle=None, **kwargs):
        try: return func()
        except BaseException as ex:  # or Exception
            if raise_ is not None: raise raise_
            if handle is not None: return handle(ex, **kwargs)
            return run() if run is not None else return_
    @staticmethod
    def show_globals(scope, **kwargs): return Struct(scope).filter(lambda k, v: "__" not in k and not k.startswith("_") and k not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}).print(**kwargs)
    @staticmethod
    def run_globally(func, scope=None, args=None, self: str = None): return Experimental.capture_locals(func=func, scope=scope, args=args, self=self, update_scope=True)
    @staticmethod
    def monkey_patch(class_inst, func): setattr(class_inst.__class__, func.__name__, func)
    @staticmethod
    def generate_readme(path, obj=None, meta=None, save_source_code=True, verbose=True):
        """Generates a readme file to contextualize any binary files by mentioning module, class, method or function used to generate the data"""
        text = "# Meta\n" + (meta if meta is not None else '') + (separator := "\n" + "-----" + "\n\n")
        if obj is not None:
            text += f"# Code to generate the result\n" + "```python\n" + (inspect := __import__("inspect")).getsource(obj) + "\n```" + separator
            text += f"# Source code file generated me was located here: \n'{inspect.getfile(obj)}'\n" + separator
        readmepath = (P(path) / f"README.md" if P(path).is_dir() else P(path)).write_text(text); print(f"SAVED README.md @ {readmepath.absolute().as_uri()}") if verbose else None
        if save_source_code: P(__import__("inspect").getmodule(obj).__file__).zip(path=readmepath.with_name("source_code.zip"), verbose=False); print("SAVED source code @ " + readmepath.with_name("source_code.zip").absolute().as_uri())
    @staticmethod
    def load_from_source_code(directory, obj=None, delete=False):
        P(directory).find("source_code*", r=True).unzip(tmpdir := P.tmp() / timestamp(name="tmp_sourcecode"))
        sys.path.insert(0, str(tmpdir)); sourcefile = __import__(tmpdir.find("*").stem); tmpdir.delete(sure=delete, verbose=False)
        return getattr(sourcefile, obj) if obj is not None else sourcefile
    @staticmethod
    def capture_locals(func, scope, args=None, self: str = None, update_scope=False): res = dict(); exec(Experimental.extract_code(func, args=args, self=self, include_args=False, verbose=False), scope, res); scope.update(res) if update_scope else None; return Struct(res)
    @staticmethod
    def extract_code(func, code: str = None, include_args=True, verbose=True, copy2clipboard=False, **kwargs):  # TODO: how to handle decorated functions.  add support for lambda functions.  ==> use dill for powerfull inspection"""
        import inspect; import textwrap  # assumptions: first line could be @classmethod or @staticmethod. second line could be def(...). Then function body must come in subsequent lines, otherwise ignored.
        raw = inspect.getsourcelines(func)[0]; lines = textwrap.dedent("".join(raw[1 + (1 if raw[0].lstrip().startswith("@") else 0):])).split("\n")
        code_string = ''.join([aline + "\n" if not textwrap.dedent(aline).startswith("return ") else aline.replace("return ", "return_ = ") + "\n" for aline in lines])  # remove return statements if there else keep line as is.
        code_string = (Experimental.extract_arguments(func, **kwargs) if include_args else '') + ("\n" + code + "\n" if code is not None else '') + "\n# " + f"BODY OF {func.__name__}".center(80, "=") +"\n" + code_string  # added later so it has more overwrite authority.
        install_n_import("clipboard").copy(code_string) if copy2clipboard else None; print(f"Code extracted from `{func}`: \n" + "=" * 100 + '\n' + code_string, "=" * 100) if verbose else None
        return code_string  # ready to be run with exec()
    @staticmethod
    def extract_arguments(func, copy2clipboard=False, **kwargs):
        ak = Struct(dict((inspect := __import__("inspect")).signature(func).parameters)).values()  # ignores self for methods automatically but also ignores args and kwargs.
        res = Struct.from_keys_values(ak.name, ak.default).update(kwargs).print(config=True, return_str=True, justify=0, quotes=True).replace("<class 'inspect._empty'>", "None").replace("= '", "= rf'")
        ak = inspect.getfullargspec(func); res = res + (f"{ak.varargs} = (,)\n" if ak.varargs else '') + (f"{ak.varkw} = " + "{}\n" if ak.varkw else '')  # add args = () and kwargs = {}
        install_n_import("clipboard").copy(res) if copy2clipboard else None; return res
    @staticmethod
    def run_cell(pointer, module=sys.modules[__name__]):
        for cell in P(module.__file__).read_text().split("#%%"):
            if pointer in cell.split('\n')[0]: break  # bingo
        else: raise KeyError(f"The pointer `{pointer}` was not found in the module `{module}`")
        print(cell); install_n_import("clipboard").copy(cell); return cell


if __name__ == '__main__':
    pass
