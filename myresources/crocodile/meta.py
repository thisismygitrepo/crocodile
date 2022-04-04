

from crocodile.core import timestamp, randstr, str2timedelta, Save, install_n_import, List, Struct
from crocodile.file_management import P, datetime
import logging
import subprocess
import sys


class Null:
    def __init__(self): pass
    def __repr__(self): return "Welcome to the labyrinth!"
    def __getattr__(self, item): _ = item; return self
    def __getitem__(self, item): _ = item; return self
    def __call__(self, *args, **kwargs): return self
    def __len__(self): return 0
    def __bool__(self): return False


class Log(object):
    def __init__(self, dialect=["colorlog", "logging", "coloredlogs"][0], name=None, file: bool = False, file_path=None, stream=True, fmt=None, sep=" | ",
                 s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False, log_colors=None):
        self.specs = dict(name=name, file=file, file_path=file_path, stream=stream, fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level)  # save speces that are essential to re-create the object at
        self.dialect = dialect  # specific to this class
        self.verbose = verbose  # specific to coloredlogs dialect
        self.log_colors = log_colors  # specific kwarg to colorlog dialect
        self.owners = []  # list of objects using this object to log. It won't be pickled anyway, no circularity prob
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
    @staticmethod
    def get_basic_format(): return logging.BASIC_FORMAT
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
        if self.specs["file_path"] is not None: self.specs["file_path"] = P(self.specs["file_path"]).rel2home()
        self._install()

    def __getstate__(self):  # logger can be pickled without this method, but its handlers are lost, so what's the point? no perfect reconstruction.
        state = self.__dict__.copy(); state["specs"] = state["specs"].copy(); del state["logger"]
        if self.specs["file_path"] is not None: state["specs"]["file_path"] = P(self.specs["file_path"]).expanduser()
        return state

    def __repr__(self): return "".join([f"{self.logger} with handlers: \n"] + [repr(h) + "\n" for h in self.logger.handlers])
    @staticmethod  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes
    def get_format(sep): return f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s{sep}%(levelno)s{sep}%(message)s{sep}"

    @staticmethod
    def get_coloredlogs(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False):
        # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        level_styles = {'spam': {'color': 'green', 'faint': True},
                        'debug': {'color': 'white'},
                        'verbose': {'color': 'blue'},
                        'info': {'color': "green"},
                        'notice': {'color': 'magenta'},
                        'warning': {'color': 'yellow'},
                        'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True, "underline": True},
                        'critical': {'color': 'red', 'bold': True, "inverse": False}}
        field_styles = {'asctime': {'color': 'green'},
                        'hostname': {'color': 'magenta'},
                        'levelname': {'color': 'black', 'bold': True},
                        'path': {'color': 'blue'},
                        'programname': {'color': 'cyan'},
                        'username': {'color': 'yellow'}}
        coloredlogs = install_n_import("coloredlogs")
        if verbose:  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
            logger = install_n_import("verboselogs").VerboseLogger(name=name); logger.setLevel(l_level)
        else:
            logger = Log.get_base_logger(logging, name=name, l_level=l_level)
            Log.add_handlers(logger, module=logging, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
        coloredlogs.install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger

    @staticmethod
    def get_colorlog(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, log_colors=None, ):
        log_colors = log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'thin_red', 'CRITICAL': 'fg_bold_red,bg_white', }  # see here for format: https://pypi.org/project/colorlog/
        colorlog = install_n_import("colorlog"); logger = Log.get_base_logger(colorlog, name, l_level)
        fmt = colorlog.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), log_colors=log_colors)
        Log.add_handlers(logger, colorlog, file, f_level, file_path, fmt, stream, s_level)
        return logger

    @staticmethod
    def get_logger(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG):
        """Basic Python logger."""
        logger = Log.get_base_logger(logging, name, l_level)
        Log.add_handlers(logger, logging, file, f_level, file_path, logging.Formatter(fmt or Log.get_format(sep)), stream, s_level)
        return logger

    @staticmethod
    def get_base_logger(module, name, l_level):
        if name is None: print(f"Logger path not passed. It is preferable to pass a path indicates the owner.")
        else: print(f"Logger `{name}` from `{module.__name__}` is instantiated with level {l_level}.")
        logger = module.getLogger(name=name or randstr()); logger.setLevel(level=l_level)  # logs everything, finer level of control is given to its handlers
        return logger

    @staticmethod
    def add_handlers(logger, module, file, f_level, file_path, fmt, stream, s_level):
        if file or file_path:  Log.add_filehandler(logger, file_path=file_path, fmt=fmt, f_level=f_level)  # create file handler for the logger.
        if stream: Log.add_streamhandler(logger, s_level, fmt, module=module)  # ==> create stream handler for the logger.

    @staticmethod
    def add_streamhandler(logger, s_level=logging.DEBUG, fmt=None, module=logging, name="myStream"):
        shandler = module.StreamHandler(); shandler.setLevel(level=s_level); shandler.setFormatter(fmt=fmt); shandler.set_name(name); logger.addHandler(shandler)
        print(f"    Level {s_level} stream handler for Logger `{logger.name}` is created.")

    @staticmethod
    def add_filehandler(logger, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="myFileHandler"):
        if file_path is None: file_path = P.tmpfile(name="logger", suffix=".log", folder="tmp_loggers")
        fhandler = logging.FileHandler(filename=str(file_path), mode=mode)
        fhandler.setFormatter(fmt=fmt); fhandler.setLevel(level=f_level); fhandler.set_name(name); logger.addHandler(fhandler)
        print(f"    Level {f_level} file handler for Logger `{logger.name}` is created @ " + P(file_path).clickable())

    @staticmethod
    def test_logger(logger):
        logger.debug("this is a debugging message"); logger.info("this is an informational message"); logger.warning("this is a warning message")
        logger.error("this is an error message"); logger.critical("this is a critical message"); [logger.log(msg=f"This is a message of level {level}", level=level) for level in range(0, 60, 5)]

    @staticmethod
    def test_all():
        for logger in [Log.get_logger(), Log.get_colorlog(), Log.get_coloredlogs()]: Log.test_logger(logger); print("=" * 100)

    @staticmethod
    def manual_degug(path): sys.stdout = open(path, 'w'); sys.stdout.close(); print(f"Finished ... have a look @ \n {path}")  # all print statements will write to this file.


class Terminal:
    class Response:
        @staticmethod
        def from_completed_process(cp: subprocess.CompletedProcess):
            resp = Terminal.Response(cmd=cp.args)
            resp.output.update(dict(stdout=cp.stdout, stderr=cp.stderr, returncode=cp.returncode))
            return resp

        def __init__(self, stdin=None, stdout=None, stderr=None, cmd=None): self.std, self.output, self.input = dict(stdin=stdin, stdout=stdout, stderr=stderr), dict(stdin="", stdout="", stderr="", returncode=None), cmd  # input command
        def __call__(self, *args, **kwargs): return self.op.rstrip() if type(self.op) is str else None
        op = property(lambda self: self.output["stdout"])
        ip = property(lambda self: self.output["stdin"])
        err = property(lambda self: self.output["stderr"])
        success = property(lambda self: self.output["returncode"] == 0)
        returncode = property(lambda self: self.output["returncode"])
        as_path = property(lambda self: P(self.op.rstrip()) if self.err == "" else None)
        def capture(self): [self.output.__setitem__(key, val.read().decode().rstrip()) for key, val in self.std.items() if val is not None and val.readable()]; return self
        def print(self): self.capture(); print(f"Terminal Response:\nInput Command: {self.input}" + "".join([f"{f' {idx} - {key} '}".center(40, "-") + f"\n{val}" for idx, (key, val) in enumerate(self.output.items())]) + "=" * 50, "\n\n"); return self

    def __init__(self, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, elevated=False):
        """
        * adding `start` to the begining of the command results in launching a new console that will not inherit from the console python was launched from (e.g. conda environment), unlike when console path is ignored.
        * `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job.
        * `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task.
        * `suprocess.call` is an archaic command for pre-Python-3.5.
        * In both `Popen` and `run`, the (shell=True) argument, implies that shell-specific commands are loaded up,
        e.g. `start` or `conda`.
        """
        self.available_consoles = ["cmd", "Command Prompt", "wt", "powershell", "wsl", "ubuntu", "pwsh"]
        self.elevated, self.stdout, self.stderr, self.stdin = elevated, stdout, stderr, stdin
        self.machine = sys.platform  # 'win32', 'linux' OR: import platform; self.platform.system(): Windows, Linux, Darwin

    def set_std_system(self): self.stdout = sys.stdout; self.stderr = sys.stderr; self.stdin = sys.stdin
    def set_std_pipe(self): self.stdout = subprocess.PIPE; self.stderr = subprocess.PIPE; self.stdin = subprocess.PIPE
    def set_std_null(self): self.stdout, self.stderr, self.stdin = subprocess.DEVNULL, subprocess.DEVNULL, subprocess.DEVNULL  # Equivalent to `echo 'foo' &> /dev/null`
    @staticmethod
    def is_admin(): return Experimental.try_this(lambda: __import__("ctypes").windll.shell32.IsUserAnAdmin(), return_=False)  # https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script

    def run(self, *cmds, shell=None, check=False, ip=None):
        """Blocking operation. Thus, if you start a shell via this method, it will run in the main and
        won't stop until you exit manually IF stdin is set to sys.stdin, otherwise it will run and close quickly.
        Other combinations of stdin, stdout can lead to funny behaviour like no output but accept input or opposite.
        * This method is short for:
        res = subprocess.run("powershell command", capture_output=True, shell=True, text=True)
        * Unlike `__import__('os').system(cmd)`, `subprocess.run(cmd)` gives much more control over the output and input.
        * `shell=True` loads up the profile of the shell called so more specific commands can be run.
            Importantly, on Windows, the `start` command becomes availalbe and new windows can be launched.
        * `text=True` converts the bytes objects returned in stdout to text by default.
        :param shell:
        :param ip:
        :param check: throw an exception is the execution of the external command failed (non zero returncode)
        """
        my_list = list(cmds)
        if self.machine == "win32" and shell in {"powershell", "pwsh"}: my_list = [shell, "-Command"] + my_list  # alternatively, one can run "cmd"
        if self.elevated is False or self.is_admin(): resp = subprocess.run(my_list, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout, text=True, shell=True, check=check, input=ip)
        else: resp = __import__("ctypes").windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        """ The advantage of addig `powershell -Command` is to give access to wider range of options. Other wise, command prompt shell doesn't recognize commands like `ls`.
        `capture_output` prevents the stdout to redirect to the stdout of the script automatically, instead it will be stored in the Response object returned.
        # `capture_output=True` same as `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
        """
        return self.Response.from_completed_process(resp)

    def run_async(self, *cmds, new_window=True, shell=None, terminal=None):
        """Opens a new terminal, and let it run asynchronously.
        Maintaining an ongoing conversation with another process is very hard. It is adviseable to run all
        commands in one go without interaction with an ongoing channel. Use this only for the purpose of
        producing a different window and humanly interact with it.
        https://stackoverflow.com/questions/54060274/dynamic-communication-between-main-and-subprocess-in-python
        https://www.youtube.com/watch?v=IynV6Y80vws
        https://www.oreilly.com/library/view/windows-powershell-cookbook/9781449359195/ch01.html
        """
        if terminal is None: terminal = ""  # this means that cmd is the default console. alternative is "wt"
        if shell is None:
            if self.machine == "win32": shell = ""  # other options are "powershell" and "cmd". # if terminal is wt, then it will pick powershell by default anyway.
            else: shell = ""
        new_window = "start" if new_window is True else ""  # start is alias for Start-Process which launches a new window.
        extra, my_list = ("-Command" if shell in {"powershell", "pwsh"} else ""), list(cmds)
        if self.machine == "win32": my_list = [new_window, terminal, shell, extra] + my_list  # having a list is equivalent to: start "ipython -i file.py". Thus, arguments of ipython go to ipython, not start.
        my_list = [item for item in my_list if item != ""]
        print("Meta.Terminal.run_async: Subprocess command: ", my_list)
        return subprocess.Popen(my_list, stdin=subprocess.PIPE, shell=True)  # stdout=self.stdout, stderr=self.stderr, stdin=self.stdin. # returns Popen object, not so useful for communcation with an opened terminal

    @staticmethod
    def run_script(script, wdir=None, interactive=True, ipython=True,
                   shell=None, delete=False, terminal="", new_window=True, header=True):
        """This method is a wrapper on top of `run_async" except that the command passed will launch python
        terminal that will run script passed by user.
        * Regular Python is much lighter than IPython. Consider using it while not debugging.
        """  # TODO: add option whether to add prepend to the script or not.
        header_script = f"""
# ======================== Code prepended by Terminal.run_script =========================
import crocodile.toolbox as tb
tb.sys.path.insert(0, r'{wdir or P.cwd()}')
# ======================== End of header, start of script passed: ========================
"""     # this header is necessary so import statements in the script passed are identified relevant to wdir.
        script = header_script + script if header else script
        if terminal in {"wt", "powershell", "pwsh"}: script += "\ntb.DisplayData.set_pandas_auto_width()\n"
        script = f"""print(r'''{script}''')""" + "\n" + script
        file = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmp_scripts").write_text(script)
        print(f"Script to be executed asyncronously: ", file.absolute().as_uri())
        Terminal().run_async(f"{'ipython' if ipython else 'python'}", f"{'-i' if interactive else ''}", f"{file}", terminal=terminal, shell=shell, new_window=new_window)
        # python will use the same dir as the one from console this method is called.
        # file.delete(sure=delete, verbose=False)
        _ = delete  # command = f'ipython {"-i" if interactive else ""} -c "{script}"'

    @staticmethod
    def replicate_in_new_session(obj, execute=False, cmd=""):
        """Python brachnes off to a new window and run the function passed. context can be either a pickled session or the current file __file__"""
        file = P.tmpfile(tstamp=False, suffix=".pkl")  # step 1: pickle the function # step 2: create a script that unpickles it. # step 3: run the script that runs the function.
        Save.pickle(obj=obj, path=file, verbose=False)
        script = f"""
path = tb.P(r'{file}')
obj = path.readit()
path.delete(sure=True, verbose=False)
obj{'()' if execute else ''}
{cmd}"""
        Terminal.run_script(script)

    @staticmethod
    def replicate_session(cmd=""):
        file = P.tmpfile(suffix=".pkl")
        __import__("dill").dump_session(file, main=sys.modules[__name__])
        script = f"""
path = tb.P(r'{file}')
tb.dill.load_session(str(path)); 
path.delete(sure=True, verbose=False)
{cmd}"""
        Terminal().run_script(script=script)

    @staticmethod
    def is_user_admin():
        """@return: True if the current user is an 'Admin' whatever that means (root on Unix), otherwise False. adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows"""
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
        """Attempt to relaunch the current script as an admin using the same command line parameters.  Pass cmdLine in to override and set a new
        command.  It must be a list of [command, arg1, arg2...] format.
        Set wait to False to avoid waiting for the sub-process to finish. You will not be able to fetch the exit code of the process if wait is False.
        Returns the sub-process return code, unless wait is False in which case it returns None.
        adopted from: https://stackoverflow.com/questions/19672352/how-to-run-script-with-elevated-privilege-on-windows
        """
        if __import__('os').name != 'nt': raise RuntimeError("This function is only implemented on Windows.")
        _ = install_n_import("win32api", name="pypiwin32")
        win32event, win32process = install_n_import("win32event"), install_n_import("win32process")
        win32com = __import__("win32com", fromlist=["shell.shell.ShellExecuteEx"])
        if cmd_line is None: cmd_line = [sys.executable] + sys.argv
        elif type(cmd_line) not in (tuple, list): raise ValueError("cmdLine is not a sequence.")
        cmd = '"%s"' % (cmd_line[0],)   # TODO: isn't there a function or something we can call to massage command line params?
        params = " ".join(['"%s"' % (x,) for x in cmd_line[1:]])
        # ShellExecute() doesn't seem to allow us to fetch the PID or handle of the process, so we can't get anything useful from it. Therefore the more complex ShellExecuteEx() must be used. # procHandle = win32api.ShellExecute(0, lpVerb, cmd, params, cmdDir, showCmd)
        proce_info = win32com.shell.shell.ShellExecuteEx(nShow=__import__("win32con").SW_SHOWNORMAL, fMask=__import__("win32com", fromlist=["shell.shellcon"]).shell.shellcon.SEE_MASK_NOCLOSEPROCESS, lpVerb='runas',  # causes UAC elevation prompt.
                                                         lpFile=cmd, lpParameters=params)
        if wait: proc_handle = proce_info['hProcess']; _ = win32event.WaitForSingleObject(proc_handle, win32event.INFINITE); rc = win32process.GetExitCodeProcess(proc_handle)
        else: rc = None; return rc


class SSH(object):
    def __init__(self, username, hostname, sshkey=None, pwd=None):
        _ = False
        if _: super().__init__()
        self.sshkey = str(sshkey) if sshkey is not None else None  # no need to pass sshkey if it was configured properly already
        import paramiko
        self.ssh = paramiko.SSHClient(); self.ssh.load_system_host_keys(); self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.hostname, self.username = hostname, username
        self.ssh.connect(hostname=hostname, username=username, password=pwd, port=22, key_filename=self.sshkey)
        self.sftp = self.ssh.open_sftp()
        self.target_machine = "Windows" if self.run("$env:OS", verbose=False).output["stdout"] == "Windows_NT" else "Linux"
        # it must uses a python independent way to figure out the machine type to avoid circualrity below:
        self.platform = __import__("platform")
        if self.platform.system() == "Windows": self.local_python_cmd = rf"""~/venvs/ve/Scripts/activate"""  # works for both cmd and pwsh
        else: self.local_python_cmd = rf"""source ~/venvs/ve/bin/activate"""
        if self.target_machine == "Windows": self.remote_python_cmd = rf"""~/venvs/ve/Scripts/activate"""  # works for both cmd and pwsh
        else: self.remote_python_cmd = rf"""source ~/venvs/ve/bin/activate"""

    def get_key(self):
        """In SSH commands you need this: scp -r {self.get_key()} "{str(source.expanduser())}" "{self.username}@{self.hostname}:'{target}' """
        return f"""-i "{str(P(self.sshkey).expanduser())}" """ if self.sshkey is not None else ""

    def copy_sshkeys_to_remote(self, fqdn):
        """Windows Openssh alternative to ssh-copy-id"""
        assert self.platform.system() == "Windows"
        return Terminal().run(fr'type $env:USERPROFILE\.ssh\id_rsa.pub | ssh {fqdn} "cat >> .ssh/authorized_keys"')

    def __repr__(self): return f"{self.local()} [{self.platform.system()}] SSH connection to {self.remote()} [{self.target_machine}] "
    def remote(self): return f"{self.username}@{self.hostname}"
    def local(self): return f"{__import__('os').getlogin()}@{self.platform.node()}"
    def open_console(self, new_window=True): Terminal().run_async(f"""ssh -i {self.sshkey} {self.username}@{self.hostname}""", new_window=new_window)
    def copy_env_var(self, name): assert self.target_machine == "Linux"; self.run(f"{name} = {__import__('os').environ[name]}; export {name}")
    def copy_to_here(self, source, target=None): pass
    def runpy(self, cmd): return self.run(f"""{self.remote_python_cmd}; python -c 'import crocodile.toolbox as tb; {cmd} ' """)
    def run_locally(self, command): print(f"Executing Locally @ {self.platform.node()}:\n{command}"); return Terminal.Response(__import__('os').system(command))

    def run(self, cmd, verbose=True):
        res = self.ssh.exec_command(cmd); res = Terminal.Response(stdin=res[0], stdout=res[1], stderr=res[2], cmd=cmd)
        res.print() if verbose else None; return res

    def copy_from_here(self, source, target=None, zip_n_encrypt=False):
        pwd = randstr(length=10, safe=True)
        if zip_n_encrypt: print(f"ZIPPING & ENCRYPTING".center(80, "=")); source = P(source).expanduser().zip_n_encrypt(pwd=pwd)
        if target is None:
            target = P(source).collapseuser()
            assert target.is_relative_to("~"), f"If target is not specified, source must be relative to home."
            target = target.as_posix()
        print("\n" * 3, f"Creating Target directory {target} @ remote machine.".center(80, "="))
        resp = self.runpy(f'print(tb.P(r"{target}").expanduser().parent.create())')
        remotepath = P(resp.op or "").joinpath(P(target).name).as_posix()
        print(f"SENT `{source}` ==> `{remotepath}`".center(80, "="))
        self.sftp.put(localpath=P(source).expanduser(), remotepath=remotepath)
        if zip_n_encrypt:
            print(f"UNZIPPING & DECRYPTING".center(80, "="))
            resp = self.runpy(f"""tb.P(r"{remotepath}").expanduser().decrypt_n_unzip(pwd="{pwd}", inplace=True)""")
            source.delete(sure=True)
            return resp


class Scheduler:
    def __init__(self, routine=lambda: None, occasional=lambda: None,
                 exception=None, wind_down=None,
                 other: int = 10, wait: str = "2m", runs=float("inf"), logger=None):
        """
        :param wait: repeat the cycle every this many minutes.
        """
        self.routine = routine  # main routine to be repeated every `wait` time_produced.
        self.occasional = occasional  # routine to be repeated every `other` time_produced.
        self.exception_handler = exception if exception is not None else lambda ex: None
        self.wind_down = wind_down
        # routine to be run when an error occurs, e.g. save object.
        self.wait = wait  # wait period between routine cycles.
        self.other = other  # number of routine cycles before `occasional` get executed once.
        self.cycles = runs  # how many times to run the routine. defaults to infinite.
        self.logger = logger or Log(name="SchedulerAutoLogger" + randstr())
        self._start_time = None  # begining of a session (local time_produced)
        self.history, self.count, self.total_count = [], 0, 0

    def run(self, until="2050-01-01", cycles=None):
        self.cycles = cycles or self.cycles
        self.count = 0
        self._start_time = datetime.now()
        wait_time = str2timedelta(self.wait).total_seconds()
        import pandas as pd  # TODO remove this dependency
        until = pd.to_datetime(until)  # (local time_produced)
        while datetime.now() < until and self.count < self.cycles:
            # 1- Opening Message ==============================================================
            time1 = datetime.now()  # time_produced before calcs started.  # use  fstring format {x:<10}
            msg = f"Starting Cycle  {self.count: 4d}. Total Run Time = {str(datetime.now() - self._start_time)}."
            self.logger.info(msg + f" UTC Time: {datetime.utcnow().isoformat(timespec='minutes', sep=' ')}")
            # 2- Perform logic ======================================================
            try: self.routine()
            except Exception as ex: self.handle_exceptions(ex)
            # 3- Optional logic every while =========================================
            if self.count % self.other == 0:
                try: self.occasional()
                except Exception as ex: self.handle_exceptions(ex)
            # 4- Conclude Message ============================================================
            self.count += 1
            time_left = int(wait_time - (datetime.now() - time1).total_seconds())  # take away processing time_produced.
            time_left = time_left if time_left > 0 else 1
            self.logger.info(f"Finishing Cycle {self.count - 1: 4d}. "
                             f"Sleeping for {self.wait} ({time_left} seconds left)\n" + "-" * 50)
            # 5- Sleep ===============================================================
            try: __import__("time").sleep(time_left)  # consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex: self.handle_exceptions(ex)
        else:  # while loop finished due to condition satisfaction (rather than breaking)
            if self.count >= self.cycles: stop_reason = f"Reached maximum number of cycles ({self.cycles})"
            else: stop_reason = f"Reached due stop time_produced ({until})"
            self.record_session_end(reason=stop_reason)

    def record_session_end(self, reason="Unknown"):
        """It is vital to record operation time_produced to retrospectively inspect market status at session time_produced."""
        self.total_count += self.count
        end_time = datetime.now()  # end of a session.
        time_run = end_time - self._start_time
        self.history.append([self._start_time, end_time, time_run, self.count])
        self.logger.critical(f"\nScheduler has finished running a session. \n"
                             f"start  time_produced: {str(self._start_time)}\n"
                             f"finish time_produced: {str(end_time)} .\n"
                             f"time_produced    ran: {str(time_run)} | wait time_produced {self.wait}  \n"
                             f"cycles  ran: {self.count}  |  Lifetime cycles: {self.total_count} \n"
                             f"termination: {reason} \n" + "-" * 100)

    def handle_exceptions(self, ex):
        """One can implement a handler that raises an error, which terminates the program, or handle
        it in some fashion, in which case the cycles continue."""
        self.record_session_end(reason=ex)
        self.exception_handler(ex)
        raise ex
        # import signal
        # def keyboard_interrupt_handler(signum, frame): print(signum, frame); raise KeyboardInterrupt
        # signal.signal(signal.SIGINT, keyboard_interrupt_handler)


class Experimental:
    """Debugging and Meta programming tools"""
    @staticmethod
    def try_this(func, return_=None, raise_=None, run=None, handle=None):
        try: return func()
        except BaseException as e:  # or Exception
            if raise_ is not None: raise raise_
            if handle is not None: return handle(e)
            return run() if run is not None else return_

    @staticmethod
    def show_globals(scope, **kwargs): return Struct(scope).filter(lambda k, v: "__" not in k and not k.startswith("_") and k not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}).print(**kwargs)
    @staticmethod
    def run_globally(func, scope, args=None, self: str = None): return Experimental.capture_locals(func=func, scope=scope, args=args, self=self, update_scope=True)
    @staticmethod
    def monkey_patch(class_inst, func): setattr(class_inst.__class__, func.__name__, func)

    @staticmethod
    def generate_readme(path, obj=None, meta=None, save_source_code=True):
        """Generates a readme file to contextualize any binary files.
        :param path: directory or file path. If directory is passed, README.md will be the filename.
        :param obj: Python module, class, method or function used to generate the result data.
         (dot not pass the data data_only or an instance of any class)
        :param meta:
        :param save_source_code:
        """
        import inspect
        readmepath, separator, text = P(path) / f"README.md" if P(path).is_dir() else P(path), "\n" + "-----" + "\n\n", "# Meta\n"
        if meta is not None: text = text + meta
        text += separator
        if obj is not None:
            lines = inspect.getsource(obj)
            text += f"# Code to generate the result\n" + "```python\n" + lines + "\n```" + separator
            text += f"# Source code file generated me was located here: \n'{inspect.getfile(obj)}'\n" + separator
        readmepath.write_text(text)
        print(f"Successfully generated README.md file. Checkout:\n", readmepath.absolute().as_uri())
        if save_source_code:
            P(inspect.getmodule(obj).__file__).zip(path=readmepath.with_name("source_code.zip"))
            print("Source code saved @ " + readmepath.with_name("source_code.zip").absolute().as_uri())

    @staticmethod
    def load_from_source_code(directory, obj=None, delete=False):
        """Does the following:
        * scope directory passed for ``source_code`` module.
        * Loads the directory to the memroy.
        * Returns either the package or a piece of it as indicated by ``obj``
        """
        tmpdir = P.tmp() / timestamp(name="tmp_sourcecode")
        P(directory).find("source_code*", r=True).unzip(tmpdir)
        sys.path.insert(0, str(tmpdir))
        sourcefile = __import__(tmpdir.find("*").stem)
        tmpdir.delete(sure=delete, verbose=False)
        return getattr(sourcefile, obj) if obj is not None else sourcefile

    @staticmethod
    def capture_locals(func, scope, args=None, self: str = None, update_scope=False):
        """Captures the local variables inside a function.
        :param func:
        :param scope: `globals()` executed in the main scope. This provides the function with scope defined in main.
        :param args: dict of what you would like to pass to the function as arguments.
        :param self: relevant only if the function is a method of a class. self refers to the path of the instance
        :param update_scope: binary flag refers to whether you want the result in a struct or update main."""
        code = Experimental.extract_code(func, args=args, self=self, include_args=False, verbose=False)
        exec(code, scope, res := dict())  # run the function within the scope `res`
        if update_scope: scope.update(res)
        return res

    @staticmethod
    def extract_code(func, code: str = None, include_args=True, modules=None,
                     verbose=True, copy2clipboard=False, **kwargs):
        """Takes in a function path, reads it source code and returns a new version of it that can be run in the main.
        This is useful to debug functions and class methods alike.
        Use: in the main: exec(extract_code(func)) or is used by `run_globally` but you need to pass globals()
        TODO: how to handle decorated functions.
        """
        if type(func) is str:
            assert modules is not None, f"If you pass a string, you must pass globals to contextualize it."
            tmp = func
            first_parenth = func.find("(")
            # last_parenth = -1
            func = eval(tmp[:first_parenth])
            # args_kwargs = tmp[first_parenth + 1: last_parenth]
            # what is self? only for methods:
            # tmp2 = tmp[:first_parenth]
            # idx = -((tmp[-1:0:-1] + tmp[0]).find(".") + 1)
            self = ".".join(func.split(".")[:-1])
            _ = self
            func = eval(func, modules)
        # TODO: add support for lambda functions.  ==> use dill for powerfull inspection
        import inspect
        import textwrap
        codelines = textwrap.dedent(inspect.getsource(func))
        if codelines.startswith("@staticmethod\n"): codelines = codelines[14:]
        assert codelines.startswith("def "), f"extract_code method is expects a function to start with `def `"
        # remove def func_name() line from the list
        idx = codelines.find("):\n")
        codelines = codelines[idx + 3:]
        # remove any indentation (4 for funcs and 8 for classes methods, etc)
        codelines = textwrap.dedent(codelines)
        lines = codelines.split("\n")  # remove return statements
        codelines = []
        for aline in lines:
            if not textwrap.dedent(aline).startswith("return "): codelines.append(aline + "\n")  # keep as is, normal statement
            else: codelines.append(aline.replace("return ", "return_ = ") + "\n")  # a return statement
        code_string = ''.join(codelines)  # convert list to string.
        args_kwargs = ""
        if include_args:  args_kwargs = Experimental.extract_arguments(func, verbose=verbose, **kwargs)
        if code is not None: args_kwargs = args_kwargs + "\n" + code + "\n"  # added later so it has more overwrite authority.
        if include_args or code: code_string = args_kwargs + code_string
        if copy2clipboard: install_n_import("clipboard").copy(code_string)
        if verbose: print(f"code to be run extracted from {func.__name__} \n", code_string, "=" * 100)
        return code_string  # ready to be run with exec()

    @staticmethod
    def extract_arguments(func, modules=None, exclude_args=True, verbose=True, copy2clipboard=False, **kwargs):
        """Get code to define the args and kwargs defined in the main. Works for funcs and methods.
        """
        if type(func) is str:  # will not work because once a string is passed, this method won't be able # to interpret it, at least not without the globals passed.
            self = ".".join(func.split(".")[:-1]); _ = self
            func = eval(func, modules)
        from crocodile.file_management import Struct
        import inspect
        ak = Struct(dict(inspect.signature(func).parameters)).values()  # ignores self for methods.
        ak = Struct.from_keys_values(ak.name, ak.default)
        ak = ak.update(kwargs)
        res = """"""
        for key, val in ak.items():
            if key != "args" and key != "kwargs":
                flag = False
                if val is inspect._empty:  # not passed argument.
                    if exclude_args: flag = True
                    else: val = None; print(f'Experimental Warning: arg {key} has no value. Now replaced with None.')
                if not flag: res += f"{key} = " + (f"'{val}'" if type(val) is str else str(val)) + "\n"
        ak = inspect.getfullargspec(func)
        if ak.varargs: res += f"{ak.varargs} = (,)\n"
        if ak.varkw: res += f"{ak.varkw} = " + "{}\n"
        if copy2clipboard: install_n_import("clipboard").copy(res)
        if verbose: print("Finished. Paste code now.")
        return res

    @staticmethod
    def edit_source(module, *edits):
        sourcelines = P(module.__file__).read_text().split("\n")
        for edit_idx, edit in enumerate(edits):
            line_idx = 0
            for line_idx, line in enumerate(sourcelines):
                if f"here{edit_idx}" in line:
                    new_line = line.replace(edit[0], edit[1])
                    print(f"Old Line: {line}\nNew Line: {new_line}")
                    if new_line == line: raise KeyError(f"Text Not found.")
                    sourcelines[line_idx] = new_line
                    break
            else: raise KeyError(f"No marker found in the text. Place the following: 'here{line_idx}'")
        newsource = "\n".join(sourcelines)
        P(module.__file__).write_text(newsource)
        import importlib
        importlib.reload(module)
        return module

    @staticmethod
    def run_cell(pointer, module=sys.modules[__name__]):
        sourcecells = P(module.__file__).read_text().split("#%%")
        for cell in sourcecells:
            if pointer in cell.split('\n')[0]: break  # bingo
        else: raise KeyError(f"The pointer `{pointer}` was not found in the module `{module}`")
        print(cell)
        install_n_import("clipboard").copy(cell)
        return cell


class Manipulator:
    @staticmethod
    def merge_adjacent_axes(array, ax1, ax2):
        """Multiplies out two axes to generate reduced order array."""
        shape = array.shape
        sz1, sz2 = shape[ax1], shape[ax2]
        new_shape = shape[:ax1] + (sz1 * sz2,)
        if ax2 == -1 or ax2 == len(shape): pass
        else: new_shape = new_shape + shape[ax2 + 1:]
        return array.reshape(new_shape)

    @staticmethod
    def merge_axes(array, ax1, ax2):
        """Brings ax2 next to ax1 first, then combine the two axes into one."""
        array2 = __import__("numpy").moveaxis(array, ax2, ax1 + 1)  # now, previously known as ax2 is located @ ax1 + 1
        return Manipulator.merge_adjacent_axes(array2, ax1, ax1 + 1)

    @staticmethod
    def expand_axis(array, ax_idx, factor, curtail=False):
        """opposite functionality of merge_axes.  While ``numpy.split`` requires the division number, this requies the split size."""
        if curtail:  # if size at ax_idx doesn't divide evenly factor, it will be curtailed.
            size_at_idx = array.shape[ax_idx]
            extra = size_at_idx % factor
            array = array[Manipulator.indexer(axis=ax_idx, myslice=slice(0, -extra))]
        total_shape = list(array.shape)
        for index, item in enumerate((int(total_shape.pop(ax_idx) / factor), factor)): total_shape.insert(ax_idx + index, item)
        return array.reshape(tuple(total_shape))  # should be same as return __import__("numpy)s.plit(array, new_shape, ax_idx)

    @staticmethod
    def slicer(array, a_slice: slice, axis=0):
        """Extends Numpy slicing by allowing rotation if index went beyond size."""
        lower_, upper_ = a_slice.start, a_slice.stop
        n = array.shape[axis]
        lower_ = lower_ % n  # if negative, you get the positive equivalent. If > n, you get principal value.
        roll = lower_
        lower_, upper_ = lower_ - roll, upper_ - roll
        array_ = __import__("numpy").roll(array, -roll, axis=axis)
        upper_ = upper_ % n
        new_slice = slice(lower_, upper_, a_slice.step)
        return array_[Manipulator.indexer(axis=axis, myslice=new_slice, rank=array.ndim)]

    @staticmethod
    def indexer(axis, myslice, rank=None):
        """Allows subseting an array of arbitrary shape, given console index to be subsetted and the range. Returns a tuple of slicers."""
        if rank is None: rank = axis + 1
        indices = [slice(None, None, None)] * rank  # slice(None, None, None) is equivalent to `:` `everything`
        indices[axis] = myslice
        # noinspection PyTypeChecker
        indices.append(Ellipsis)  # never hurts to add this in the end.
        return tuple(indices)


if __name__ == '__main__':
    pass
