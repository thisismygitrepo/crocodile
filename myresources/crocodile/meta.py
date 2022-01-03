
import logging
import subprocess
import time
from crocodile.core import np, os, sys, timestamp, randstr, str2timedelta, datetime, Save, assert_package_installed
from crocodile.file_management import P


class Null:
    def __init__(self):
        pass

    def __repr__(self):
        return "Welcome to the labyrinth!"

    def __getattr__(self, item):
        _ = item
        return self

    def __getitem__(self, item):
        _ = item
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __len__(self):
        return 0


class Cycle:
    def __init__(self, c=None, name=''):
        self.c = c  # a list of values.
        self.index = -1
        self.name = name

    def __str__(self):
        return self.name

    def next(self):
        self.index += 1
        if self.index >= len(self.c):
            self.index = 0
        return self.c[self.index]

    def previous(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.c) - 1
        return self.c[self.index]

    def set(self, value):
        self.index = self.c.index(value)

    def get(self):
        return self.c[self.index]

    def get_index(self):
        return self.index

    def set_index(self, index):
        self.index = index

    def sample(self, size=1):
        return np.random.choice(self.c, size)

    def __add__(self, other):
        pass  # see behviour of matplotlib cyclers.


class DictCycle(Cycle):
    def __init__(self, strct, **kwargs):
        strct = dict(strct)
        super(DictCycle, self).__init__(c=strct.items(), **kwargs)
        self.keys = strct.keys()

    def set_key(self, key):
        self.index = list(self.keys).index(key)


class Experimental:
    """Debugging and Meta programming tools"""

    @staticmethod
    def profile_memory(command):
        psutil = assert_package_installed("psutil")
        before = psutil.virtual_memory()
        exec(command)
        after = psutil.virtual_memory()
        print(f"Memory used = {(after.used - before.used) / 1e6}")

    @staticmethod
    def try_this(func, otherwise=None):
        try:
            return func()
        except BaseException as e:
            _ = e
            return otherwise

    @staticmethod
    def show_globals(scope, **kwargs):
        """Returns a struct with variables that are defined in the globals passed."""
        res = scope.keys()
        res = res.filter(lambda x: "__" not in x).filter(lambda x: not x.startswith("_"))
        res = res.filter(lambda x: x not in {"In", "Out", "get_ipython", "quit", "exit", "sys"})
        res.print(**kwargs)

    @staticmethod
    def generate_readme(path, obj=None, meta=None, save_source_code=True):
        """Generates a readme file to contextualize any binary files.

        :param path: directory or file path. If directory is passed, README.md will be the filename.
        :param obj: Python module, class, method or function used to generate the result data.
         (dot not pass the data itself or an instance of any class)
        :param meta:
        :param save_source_code:
        """
        import inspect
        path = P(path)
        readmepath = path / f"README.md" if path.is_dir() else path

        separator = "\n" + "-----" + "\n\n"
        text = "# Meta\n"
        if meta is not None:
            text = text + meta
        text += separator

        if obj is not None:
            lines = inspect.getsource(obj)
            text += f"# Code to generate the result\n" + "```python\n" + lines + "\n```" + separator
            text += f"# Source code file generated me was located here: \n'{inspect.getfile(obj)}'\n" + separator

        readmepath.write_text(text)
        print(f"Successfully generated README.md file. Checkout:\n", readmepath.absolute().as_uri())

        if save_source_code:
            P(inspect.getmodule(obj).__file__).zip(op_path=readmepath.with_name("source_code.zip"))
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
        if obj is not None:
            loaded = getattr(sourcefile, obj)
            return loaded
        else:
            return sourcefile

    @staticmethod
    def capture_locals(func, scope, args=None, self: str = None, update_scope=False):
        """Captures the local variables inside a function.
        :param func:
        :param scope: `globals()` executed in the main scope. This provides the function with scope defined in main.
        :param args: dict of what you would like to pass to the function as arguments.
        :param self: relevant only if the function is a method of a class. self refers to the name of the instance
        :param update_scope: binary flag refers to whether you want the result in a struct or update main."""
        code = Experimental.extract_code(func, args=args, self=self, include_args=False, verbose=False,
                                         )
        print(code)
        res = dict()
        exec(code, scope, res)  # run the function within the scope `res`
        if update_scope:
            scope.update(res)
        return res

    @staticmethod
    def run_globally(func, scope, args=None, self: str = None):
        return Experimental.capture_locals(func=func, scope=scope, args=args, self=self, update_scope=True)

    @staticmethod
    def extract_code(func, code: str = None, include_args=True, modules=None,
                     verbose=True, copy2clipboard=False, **kwargs):
        """Takes in a function name, reads it source code and returns a new version of it that can be run in the main.
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

        # remove return statements
        lines = codelines.split("\n")
        codelines = []
        for aline in lines:
            if not textwrap.dedent(aline).startswith("return "):  # normal statement
                codelines.append(aline + "\n")  # keep as is
            else:  # a return statement
                codelines.append(aline.replace("return ", "return_ = ") + "\n")

        code_string = ''.join(codelines)  # convert list to string.
        args_kwargs = ""
        if include_args:
            args_kwargs = Experimental.extract_arguments(func, verbose=verbose, **kwargs)
        if code is not None:
            args_kwargs = args_kwargs + "\n" + code + "\n"  # added later so it has more overwrite authority.
        if include_args or code:
            code_string = args_kwargs + code_string

        if copy2clipboard:
            clipboard = assert_package_installed("clipboard")
            clipboard.copy(code_string)
        if verbose: print(f"code to be run extracted from {func.__name__} \n", code_string, "=" * 100)
        return code_string  # ready to be run with exec()

    @staticmethod
    def extract_arguments(func, modules=None, exclude_args=True, verbose=True, copy2clipboard=False, **kwargs):
        """Get code to define the args and kwargs defined in the main. Works for funcs and methods.
        """
        if type(func) is str:  # will not work because once a string is passed, this method won't be able
            # to interpret it, at least not without the globals passed.
            self = ".".join(func.split(".")[:-1])
            _ = self
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
                    if exclude_args:
                        flag = True
                    else:
                        val = None
                        print(f'Experimental Warning: arg {key} has no value. Now replaced with None.')
                if not flag:
                    res += f"{key} = " + (f"'{val}'" if type(val) is str else str(val)) + "\n"

        ak = inspect.getfullargspec(func)
        if ak.varargs:
            res += f"{ak.varargs} = (,)\n"
        if ak.varkw:
            res += f"{ak.varkw} = " + "{}\n"

        if copy2clipboard:
            clipboard = assert_package_installed("clipboard")
            clipboard.copy(res)
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
                    if new_line == line:
                        raise KeyError(f"Text Not found.")
                    sourcelines[line_idx] = new_line
                    break
            else:
                raise KeyError(f"No marker found in the text. Place the following: 'here{line_idx}'")
        newsource = "\n".join(sourcelines)
        P(module.__file__).write_text(newsource)
        import importlib
        importlib.reload(module)
        return module

    @staticmethod
    def monkey_patch(class_inst, func):
        """On the fly, attach a function as a method of an instantiated class."""
        setattr(class_inst.__class__, func.__name__, func)

    @staticmethod
    def run_cell(pointer, module=sys.modules[__name__]):
        # update the module by reading it again.
        # if type(module) is str:
        #     module = __import__(module)
        # importlib.reload(module)
        # if type(module) is str:
        #     sourcecells = P(module).read_text().split("#%%")
        # else:
        sourcecells = P(module.__file__).read_text().split("#%%")

        for cell in sourcecells:
            if pointer in cell.split('\n')[0]:
                break  # bingo
        else:
            raise KeyError(f"The pointer `{pointer}` was not found in the module `{module}`")
        print(cell)
        clipboard = assert_package_installed("clipboard")
        clipboard.copy(cell)
        return cell


class Manipulator:
    @staticmethod
    def merge_adjacent_axes(array, ax1, ax2):
        """Multiplies out two axes to generate reduced order array.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        changed in April 2021
        """
        shape = array.shape
        sz1, sz2 = shape[ax1], shape[ax2]
        new_shape = shape[:ax1] + (sz1 * sz2,)
        if ax2 == -1 or ax2 == len(shape):
            pass
        else:
            new_shape = new_shape + shape[ax2 + 1:]
        return array.reshape(new_shape)

    @staticmethod
    def merge_axes(array, ax1, ax2):
        """Brings ax2 next to ax1 first, then combine the two axes into one.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        """
        array2 = np.moveaxis(array, ax2, ax1 + 1)  # now, previously known as ax2 is located @ ax1 + 1
        return Manipulator.merge_adjacent_axes(array2, ax1, ax1 + 1)

    @staticmethod
    def expand_axis(array, ax_idx, factor, curtail=False):
        """opposite functionality of merge_axes.
        While ``numpy.split`` requires the division number, this requies the split size.
        """
        if curtail:  # if size at ax_idx doesn't divide evenly factor, it will be curtailed.
            size_at_idx = array.shape[ax_idx]
            extra = size_at_idx % factor
            array = array[Manipulator.indexer(axis=ax_idx, myslice=slice(0, -extra))]
        total_shape = list(array.shape)
        size = total_shape.pop(ax_idx)
        new_shape = (int(size / factor), factor)
        for index, item in enumerate(new_shape):
            total_shape.insert(ax_idx + index, item)
        # should be same as return np.split(array, new_shape, ax_idx)
        return array.reshape(tuple(total_shape))

    @staticmethod
    def slicer(array, a_slice: slice, axis=0):
        """Extends Numpy slicing by allowing rotation if index went beyond size."""
        lower_ = a_slice.start
        upper_ = a_slice.stop
        n = array.shape[axis]
        lower_ = lower_ % n  # if negative, you get the positive equivalent. If > n, you get principal value.
        roll = lower_
        lower_ = lower_ - roll
        upper_ = upper_ - roll
        array_ = np.roll(array, -roll, axis=axis)
        upper_ = upper_ % n
        new_slice = slice(lower_, upper_, a_slice.step)
        return array_[Manipulator.indexer(axis=axis, myslice=new_slice, rank=array.ndim)]

    @staticmethod
    def indexer(axis, myslice, rank=None):
        """Allows subseting an array of arbitrary shape, given console index to be subsetted and the range.
        Returns a tuple of slicers.
        changed in April 2021 without testing.
        """
        everything = slice(None, None, None)  # `:`
        if rank is None:
            rank = axis + 1
        indices = [everything] * rank
        indices[axis] = myslice
        # noinspection PyTypeChecker
        indices.append(Ellipsis)  # never hurts to add this in the end.
        return tuple(indices)


M = Manipulator


def batcher(func_type='function'):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, x, *args, per_instance_kwargs=None, **kwargs):
                output = []
                for counter, item in enumerate(x):
                    if per_instance_kwargs is not None:
                        mykwargs = {key: value[counter] for key, value in per_instance_kwargs.items()}
                    else:
                        mykwargs = {}
                    output.append(func(self, item, *args, **mykwargs, **kwargs))
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __init__(self, func):
                self.func = func

            def __call__(self, x, **kwargs):
                output = [self.func(item, **kwargs) for item in x]
                return np.array(output)

        return Batch


def batcherv2(func_type='function', order=1):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, *args, **kwargs):
                output = [func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __int__(self, func):
                self.func = func

            def __call__(self, *args, **kwargs):
                output = [self.func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

        return Batch


class Terminal:
    class Response:
        @staticmethod
        def from_completed_process(cp: subprocess.CompletedProcess):
            tmp = dict(stdout=cp.stdout, stderr=cp.stderr, returncode=cp.returncode)
            # for key, val in tmp:
            #     if val is str:
            #         tmp[key] = val.rstrip()
            resp = Terminal.Response(cmd=cp.args)
            resp.output.update(tmp)
            return resp

        def __init__(self, stdin=None, stdout=None, stderr=None, cmd=None):
            self.std = dict(stdin=stdin, stdout=stdout, stderr=stderr)  # streams go here.
            self.output = dict(stdin=None, stdout=None, stderr=None, returncode=None)
            self.input = cmd  # input command

        def __call__(self, *args, **kwargs):
            return self.op.rstrip() if type(self.op) is str else None

        @property
        def op(self):
            return self.output["stdout"]

        @property
        def ip(self):
            return self.output["stdin"]

        @property
        def err(self):
            return self.output["stderr"]

        @property
        def returncode(self):
            return self.output["returncode"]

        def capture(self):
            for key, val in self.std.items():
                if val is not None and val.readable():
                    string = val.read().decode()
                    self.output[key] = string.rstrip()  # move ending spaces and new lines.

        def print(self):
            self.capture()
            print(f"Terminal Response:\nInput Command: {self.input}")
            for idx, (key, val) in enumerate(self.output.items()):
                print(f"{idx} - {key} {'=' * 30}\n{val}")
            print("-" * 50, "\n\n")
            return self

    def __init__(self, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, elevated=False):
        """
        Console
        Terminal
        Bash
        Shell
        Host
        * adding `start` to the begining of the command results in launching a new console that will not
        inherit from the console python was launched from (e.g. conda enviroment), unlike when console name is ignored.

        * `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job.
        * `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task.
        * `suprocess.call` is an archaic command for pre-Python-3.5.
        * In both `Popen` and `run`, the (shell=True) argument, implies that shell-specific commands are loaded up,
        e.g. `start` or `conda`.
        * To launch a new window, either use
        """
        self.available_consoles = ["cmd", "Command Prompt", "wt", "powershell", "wsl", "ubuntu", "pwsh"]
        self.elevated = elevated
        self.stdout = stdout
        self.stderr = stderr
        self.stdin = stdin
        import platform
        self.machine = platform.system()  # Windows, Linux, Darwin

    def set_std_system(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.stdin = sys.stdin

    def set_std_pipe(self):
        self.stdout = subprocess.PIPE
        self.stderr = subprocess.PIPE
        self.stdin = subprocess.PIPE

    def set_std_null(self):
        """Equivalent to `echo 'foo' &> /dev/null`"""
        self.stdout = subprocess.DEVNULL
        self.stderr = subprocess.DEVNULL
        self.stdin = subprocess.DEVNULL

    @staticmethod
    def is_admin():
        # https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script
        import ctypes
        return Experimental.try_this(lambda: ctypes.windll.shell32.IsUserAnAdmin(), otherwise=False)

    def run(self, *cmds, shell=None, check=False, ip=None):
        """Blocking operation. Thus, if you start a shell via this method, it will run in the main and
        won't stop until you exit manually IF stdin is set to sys.stdin, otherwise it will run and close quickly.
        Other combinations of stdin, stdout can lead to funny behaviour like no output but accept input or opposite.

        * This method is short for:
        res = subprocess.run("powershell command", capture_output=True, shell=True, text=True)
        * Unlike `os.system(cmd)`, `subprocess.run(cmd)` gives much more control over the output and input.
        * `shell=True` loads up the profile of the shell called so more specific commands can be run.
            Importantly, on Windows, the `start` command becomes availalbe and new windows can be launched.
        * `text=True` converts the bytes objects returned in stdout to text by default.

        :param shell:
        :param ip:
        :param check: throw an exception is the execution of the external command failed (non zero returncode)
        """
        my_list = []
        if self.machine == "Windows":
            if shell in {"powershell", "pwsh"}:
                my_list = [shell, "-Command"]  # alternatively, one can run "cmd"
                """ The advantage of addig `powershell -Command` is to give access to wider range of options.
            Other wise, command prompt shell doesn't recognize commands like `ls`."""
            else:
                pass  # that is, use command prompt as a shell, which is the default.
        my_list += list(cmds)
        if self.elevated is False or self.is_admin():
            resp = subprocess.run(my_list, stderr=self.stderr, stdin=self.stdin, stdout=self.stdout,
                                  text=True, shell=True, check=check, input=ip)
            """
            `capture_output` prevents the stdout to redirect to the stdout of the script automatically, instead it will
        be stored in the Response object returned.
            # `capture_output=True` same as `stdout=subprocess.PIPE, stderr=subprocess.PIPE`
            """
        else:
            import ctypes
            resp = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
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
        if terminal is None:
            terminal = ""  # this means that cmd is the default console. alternative is "wt"
        if shell is None:
            if self.machine == "Windows":
                shell = ""  # other options are "powershell" and "cmd"
                # if terminal is wt, then it will pick powershell by default anyway.
            else:
                shell = ""
        if new_window is True:  # start is alias for Start-Process which launches a new window.
            new_window = "start"
        else:
            new_window = ""
        if shell in {"powershell", "pwsh"}:
            extra = "-Command"
        else:
            extra = ""
        my_list = []
        if self.machine == "Windows":
            my_list += [new_window, terminal, shell, extra]
        my_list += cmds
        print("Meta.Terminal.run_async: Subprocess command: ", my_list)
        my_list = [item for item in my_list if item != ""]
        w = subprocess.Popen(my_list, stdin=subprocess.PIPE, shell=True)  # stdout=self.stdout, stderr=self.stderr, stdin=self.stdin
        # returns Popen object, not so useful for communcation with an opened terminal
        return w

    @staticmethod
    def run_script(script, wdir=None, interactive=True, ipython=True,
                   shell=None, delete=False, terminal="", new_window=True):
        """This method is a wrapper on top of `run_async" except that the command passed will launch python
        terminal that will run script passed by user.
        * Regular Python is much lighter than IPython. Consider using it while not debugging.
        """
        wdir = wdir or P.cwd()
        header = f"""
# The following lines of code form a header appended by Terminal.run_script
import crocodile.toolbox as tb
tb.sys.path.insert(0, r'{wdir}')
# End of header, start of script passed:
"""  # this header is necessary so import statements in the script passed are identified relevant to wdir.

        script = header + script
        if terminal in {"wt", "powershell", "pwsh"}:
            script += "\ntb.DisplayData.set_pandas_auto_width()\n"
        script = f"""print(r'''{script}''')""" + "\n" + script
        file = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmpscripts")
        file.write_text(script)
        print(f"Script to be executed asyncronously: ", file.absolute().as_uri())
        Terminal().run_async(f"{'ipython' if ipython else 'python'}",
                             f"{'-i' if interactive else ''}",
                             f"{file}",
                             terminal=terminal, shell=shell, new_window=new_window)
        # python will use the same dir as the one from console this method is called.
        # file.delete(sure=delete, verbose=False)
        _ = delete
        # command = f'ipython {"-i" if interactive else ""} -c "{script}"'

    @staticmethod
    def load_object_in_new_session(obj):
        """Python brachnes off to a new window and run the function passed.
        context can be either a pickled session or the current file __file__"""
        # step 1: pickle the function
        # step 2: create a script that unpickles it.
        # step 3: run the script that runs the function.
        # TODO complete this
        fname = P.tmpfile(tstamp=False, suffix=".pkl")
        Save.pickle(obj=obj, path=fname, verbose=False)
        script = f"""
fname = tb.P(r'{fname}')
obj = fname.readit()
fname.delete(sure=True, verbose=False)
"""
        Terminal.run_script(script)


class SSH(object):
    def __init__(self, hostname, username, ssh_key=None):
        _ = False
        if _:
            super().__init__()
        import paramiko
        self.ssh_key = ssh_key
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.hostname = hostname
        self.username = username
        self.ssh.connect(hostname=hostname,
                         username=username,
                         port=22, key_filename=self.ssh_key.string if self.ssh_key is not None else None)
        self.sftp = self.ssh.open_sftp()
        self.load_python_cmd = rf"""source ~/miniconda3/bin/activate"""  # possible activate an env
        import platform
        self.platform = platform
        self.target_machine = self.ssh.exec_command(self.load_python_cmd +
                                                    "python -c 'import platform; platform.system()'")

    def get_key(self):
        """In SSH commands you need this:
        scp -r {self.get_key()} "{str(source.expanduser())}" "{self.username}@{self.hostname}:'{target}'"
        """
        if self.ssh_key is not None:
            return f"""-i "{str(P(self.ssh_key.expanduser()))}" """
        else:
            return ""

    def __repr__(self):
        return f"{self.local()} SSH connection to {self.remote()}"

    def remote(self):
        return f"{self.username}@{self.hostname}"

    def local(self):
        return f"{os.getlogin()}@{self.platform.node()}"

    def open_console(self, new_window=True):
        cmd = f"""ssh -i {self.ssh_key} {self.username}@{self.hostname}"""
        Terminal().run_async(cmd, new_window=new_window)

    def copy_env_var(self, name):
        assert self.target_machine == "Linux"
        self.run(f"{name} = {os.environ[name]}; export {name}")

    def copy_from_here(self, source, target=None, zip_n_encrypt=True):
        pwd = None
        if zip_n_encrypt:
            pwd = randstr(length=10, safe=True)
            source = P(source).expanduser().zip_n_encrypt(pwd=pwd)
        print(f"Step 1: Creating Target directory {target} @ remote machine.")
        if target is None: target = P(source).collapseuser().as_posix()  # works if source is relative.
        resp = self.runpy(f'print(tb.P(r"{target}").expanduser().parent.create())')
        remotepath = P(resp.op or "").joinpath(P(target).name).as_posix()
        print(f"Send `{source}` ==> `{remotepath}` ")
        self.sftp.put(localpath=P(source).expanduser(), remotepath=remotepath)
        if zip_n_encrypt:
            resp = self.runpy(f"""tb.P(r"{remotepath}").expanduser().decrypt_n_unzip(pwd="{pwd}", delete=True)""")
            return resp

    def copy_to_here(self, source, target=None):
        pass

    def run(self, cmd, printit=True):
        print(f"\nExecuting on remote {self.username}@{self.hostname}:\n{cmd}")
        res = self.ssh.exec_command(cmd)
        res = Terminal.Response(stdin=res[0], stdout=res[1], stderr=res[2], cmd=cmd)
        if printit: res.print()
        return res

    def runpy(self, cmd):
        cmd = f"""{self.load_python_cmd}; python -c 'import crocodile.toolbox as tb; {cmd} ' """
        return self.run(cmd)

    def run_locally(self, command):
        print(f"Executing Locally @ {self.platform.node()}:\n{command}")
        return Terminal.Response(os.system(command))


class Log(object):
    """This class is needed once a project grows beyond simple work. Simple print statements from
    dozens of objects will not be useful as the programmer will not easily recognize who
     is printing this message, in addition to many other concerns.

     Advantages of using instances of this class: You do not need to worry about object pickling process by modifing
     the __getstate__ method of the class that will own the logger. This is the case because loggers lose access
     to the file logger when unpickled, so it is better to instantiate them again.
     Logger can be pickled, but its handlers are lost, so what's the point? no perfect reconstruction.
     Additionally, this class keeps track of log files used, append to them if they still exist.

     Implementation detail: the design favours composition over inheritence. To counter the inconvenience
      of having extra typing to reach the logger, a property `logger` was added to Base class to refer to it."""

    def __init__(self, dialect=["colorlog", "logging", "coloredlogs"][0],
                 name=None, file: bool = False, file_path=None, stream=True, fmt=None, sep=" | ",
                 s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG,
                 verbose=False, log_colors=None):
        # save speces that are essential to re-create the object at
        self.specs = dict(name=name, file=file, file_path=file_path, stream=stream, fmt=fmt, sep=sep,
                          s_level=s_level, f_level=f_level, l_level=l_level)
        self.dialect = dialect  # specific to this class
        self.verbose = verbose  # specific to coloredlogs dialect
        self.log_colors = log_colors  # specific kwarg to colorlog dialect
        if file is False and stream is False:
            self.logger = Null()
        else:
            self.logger = Null()  # to be populated by `_install`
            self._install()
            # update specs after intallation.
            self.specs["name"] = self.logger.name
            if file:  # first handler is a file handler
                self.specs["file_path"] = self.logger.handlers[0].baseFilename

    def __getattr__(self, item):  # makes it twice as slower as direct access 300 ns vs 600 ns
        return getattr(self.logger, item)

    def debug(self, msg):  # to speed up the process and avoid falling back to __getattr__
        return self.logger.debug(msg)

    def info(self, msg):
        return self.logger.info(msg)

    def warn(self, msg):
        return self.logger.warn(msg)

    def error(self, msg):
        return self.logger.error(msg)

    def critical(self, msg):
        return self.logger.critical(msg)

    @property
    def file(self):
        return P(self.specs["file_path"]) if self.specs["file_path"] else None

    def _install(self):  # populates self.logger attribute according to specs and dielect.
        if self.dialect == "colorlog":
            self.logger = Log.get_colorlog(log_colors=self.log_colors, **self.specs)
        elif self.dialect == "logging":
            self.logger = Log.get_logger(**self.specs)
        elif self.dialect == "coloredlogs":
            self.logger = Log.get_coloredlogs(verbose=self.verbose, **self.specs)
        else:  # default
            self.logger = Log.get_colorlog(**self.specs)

    def __setstate__(self, state):
        self.__dict__ = state
        if self.specs["file_path"] is not None:
            self.specs["file_path"] = P.home() / self.specs["file_path"]
        self._install()

    def __getstate__(self):
        # logger can be pickled, but its handlers are lost, so what's the point? no perfect reconstruction.
        state = self.__dict__.copy()
        state["specs"] = state["specs"].copy()
        del state["logger"]
        if self.specs["file_path"] is not None:
            state["specs"]["file_path"] = P(self.specs["file_path"]).rel2home()
        return state

    def __repr__(self):
        tmp = f"{self.logger} with handlers: \n"
        for h in self.logger.handlers:
            tmp += repr(h) + "\n"
        return tmp

    @staticmethod
    def get_format(sep):
        fmt = f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s{sep}%(levelno)s" \
              f"{sep}%(message)s{sep}"
        # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes
        return fmt

    @staticmethod
    def get_basic_format():
        return logging.BASIC_FORMAT

    @staticmethod
    def get_coloredlogs(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ",
                        s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG,
                        verbose=False):
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
                        'name': {'color': 'blue'},
                        'programname': {'color': 'cyan'},
                        'username': {'color': 'yellow'}}
        coloredlogs = assert_package_installed("coloredlogs")
        if verbose:
            verboselogs = assert_package_installed("verboselogs")
            # https://github.com/xolox/python-verboselogs
            # verboselogs.install()  # hooks into logging module.
            logger = verboselogs.VerboseLogger(name=name)
            logger.setLevel(l_level)
        else:
            logger = Log.get_base_logger(logging, name=name, l_level=l_level)
            # new step, not tested:
            Log.add_handlers(logger, module=logging, file=file, f_level=f_level, file_path=file_path,
                             fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)
        coloredlogs.install(logger=logger, name="lol_different_name", level=logging.NOTSET,
                            level_styles=level_styles, field_styles=field_styles,
                            fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger

    @staticmethod
    def get_colorlog(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ",
                     s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG,
                     log_colors=None,
                     ):
        if log_colors is None:
            log_colors = {'DEBUG': 'bold_cyan',
                          'INFO': 'green',
                          'WARNING': 'yellow',
                          'ERROR': 'thin_red',
                          'CRITICAL': 'fg_bold_red,bg_white',
                          }  # see here for format: https://pypi.org/project/colorlog/
        colorlog = assert_package_installed("colorlog")
        logger = Log.get_base_logger(colorlog, name, l_level)
        fmt = colorlog.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), log_colors=log_colors)
        Log.add_handlers(logger, colorlog, file, f_level, file_path, fmt, stream, s_level)
        return logger

    @staticmethod
    def get_logger(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ",
                   s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG):
        """Basic Python logger."""
        logger = Log.get_base_logger(logging, name, l_level)
        fmt = logging.Formatter(fmt or Log.get_format(sep))
        Log.add_handlers(logger, logging, file, f_level, file_path, fmt, stream, s_level)
        return logger

    @staticmethod
    def get_base_logger(module, name, l_level):
        if name is None:
            print(f"Logger name not passed. It is preferable to pass a name indicates the owner.")
        else:
            print(f"Logger `{name}` from `{module.__name__}` is instantiated with level {l_level}.")
        logger = module.getLogger(name=name or randstr())
        logger.setLevel(level=l_level)  # logs everything, finer level of control is given to its handlers
        return logger

    @staticmethod
    def add_handlers(logger, module, file, f_level, file_path, fmt, stream, s_level):
        if file or file_path:  # ==> create file handler for the logger.
            Log.add_filehandler(logger, file_path=file_path, fmt=fmt, f_level=f_level)
        if stream:  # ==> create stream handler for the logger.
            Log.add_streamhandler(logger, s_level, fmt, module=module)

    @staticmethod
    def add_streamhandler(logger, s_level=logging.DEBUG, fmt=None, module=logging):
        shandler = module.StreamHandler()
        shandler.setLevel(level=s_level)
        shandler.setFormatter(fmt=fmt)
        logger.addHandler(shandler)
        print(f"    Level {s_level} stream handler for Logger `{logger.name}` is created.")

    @staticmethod
    def add_filehandler(logger, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="fileHandler"):
        if file_path is None:
            file_path = P.tmpfile(name="logger", suffix=".log", folder="loggers")
        fhandler = logging.FileHandler(filename=str(file_path), mode=mode)
        fhandler.setFormatter(fmt=fmt)
        fhandler.setLevel(level=f_level)
        fhandler.set_name(name)
        logger.addHandler(fhandler)
        print(f"    Level {f_level} file handler for Logger `{logger.name}` is created @ " + P(
            file_path).absolute().as_uri())

    @staticmethod
    def test_logger(logger):
        logger.debug("this is a debugging message")
        logger.info("this is an informational message")
        logger.warning("this is a warning message")
        logger.error("this is an error message")
        logger.critical("this is a critical message")
        for level in range(0, 60, 5):
            logger.log(msg=f"This is a message of level {level}", level=level)

    @staticmethod
    def test_all():
        for logger in [Log.get_logger(), Log.get_colorlog(), Log.get_coloredlogs()]:
            Log.test_logger(logger)
            print("=" * 100)

    # def config_root_logger(self):
    #     logging.basicConfig(filename=None, filemode="w", level=None, format=None)

    @staticmethod
    def manual_degug(path):  # man
        sys.stdout = open(path, 'w')  # all print statements will write to this file.
        sys.stdout.close()
        print(f"Finished ... have a look @ \n {path}")


def accelerate(func, ip):
    """ Conditions for this to work:
    * Must run under __main__ context
    * func must be defined outside that context.

    To accelerate IO-bound process, use multithreading. An example of that is somthing very cheap to process,
    but takes a long time to be obtained like a request from server. For this, multithreading launches all threads
    together, then process them in an interleaved fashion as they arrive, all will line-up for same processor,
    if it happens that they arrived quickly.

    To accelerate processing-bound process use multiprocessing, even better, use Numba.
    Method1 use: multiprocessing / multithreading.
    Method2: using joblib (still based on multiprocessing)
    from joblib import Parallel, delayed
    Fast method using Concurrent module
    """
    split = np.array_split(ip, os.cpu_count())
    # make each thread process multiple inputs to avoid having obscene number of threads with simple fast
    # operations

    # vectorize the function so that it now accepts lists of ips.
    # def my_func(ip):
    #     return [func(tmp) for tmp in ip]

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        op = executor.map(func, split)
        op = list(op)  # convert generator to list
    op = np.concatenate(op, axis=0)
    # op = self.reader.assign_resize(op, f=0.8, nrp=56, ncp=47, interpolation=True)
    return op


class Scheduler:
    def __init__(self, routine=lambda: None, occasional=lambda: None,
                 exception=None, wind_down=None,
                 other: int = 10, wait: str = "2m", runs=float("inf"), logger=None):
        """
        :param wait: repeat the cycle every this many minutes.
        """
        self.routine = routine  # main routine to be repeated every `wait` time.
        self.occasional = occasional  # routine to be repeated every `other` time.
        self.exception_handler = exception if exception is not None else lambda ex: None
        self.wind_down = wind_down
        # routine to be run when an error occurs, e.g. save object.
        self.wait = wait  # wait period between routine cycles.
        self.other = other  # number of routine cycles before `occasional` get executed once.
        self.cycles = runs  # how many times to run the routine. defaults to infinite.
        self.logger = logger or Log(name="SchedulerAutoLogger" + randstr())
        self.history = []
        self._start_time = None  # begining of a session (local time)
        self.total_count = 0
        self.count = 0

    def run(self, until="2050-01-01", cycles=None):
        self.cycles = cycles or self.cycles
        self.count = 0
        self._start_time = datetime.now()
        wait_time = str2timedelta(self.wait).total_seconds()
        import pandas as pd
        until = pd.to_datetime(until)  # (local time)

        while datetime.now() < until and self.count < self.cycles:
            # 1- Opening Message ==============================================================
            time1 = datetime.now()  # time before calcs started.  # use  fstring format {x:<10}
            msg = f"Starting Cycle  {self.count: 4d}. Total Run Time = {str(datetime.now() - self._start_time)}."
            self.logger.info(msg + f" UTC Time: {datetime.utcnow().isoformat(timespec='minutes', sep=' ')}")

            # 2- Perform logic ======================================================
            try:
                self.routine()
            except Exception as ex:
                self.handle_exceptions(ex)

            # 3- Optional logic every while =========================================
            if self.count % self.other == 0:
                try:
                    self.occasional()
                except Exception as ex:
                    self.handle_exceptions(ex)

            # 4- Conclude Message ============================================================
            self.count += 1
            time_left = int(wait_time - (datetime.now() - time1).total_seconds())  # take away processing time.
            time_left = time_left if time_left > 0 else 1
            self.logger.info(f"Finishing Cycle {self.count - 1: 4d}. "
                             f"Sleeping for {self.wait} ({time_left} seconds left)\n" + "-" * 50)

            # 5- Sleep ===============================================================
            try:
                time.sleep(time_left)  # consider replacing by Asyncio.sleep
            except KeyboardInterrupt as ex:
                self.handle_exceptions(ex)

        else:  # while loop finished due to condition satisfaction (rather than breaking)
            if self.count >= self.cycles:
                stop_reason = f"Reached maximum number of cycles ({self.cycles})"
            else:
                stop_reason = f"Reached due stop time ({until})"
            self.record_session_end(reason=stop_reason)

    def record_session_end(self, reason="Unknown"):
        """It is vital to record operation time to retrospectively inspect market status at session time."""
        self.total_count += self.count
        end_time = datetime.now()  # end of a session.
        time_run = end_time - self._start_time
        self.history.append([self._start_time, end_time, time_run, self.count])
        self.logger.critical(f"\nScheduler has finished running a session. \n"
                             f"start  time: {str(self._start_time)}\n"
                             f"finish time: {str(end_time)} .\n"
                             f"time    ran: {str(time_run)} | wait time {self.wait}  \n"
                             f"cycles  ran: {self.count}  |  Lifetime cycles: {self.total_count} \n"
                             f"termination: {reason} \n" + "-" * 100)

    def handle_exceptions(self, ex):
        """One can implement a handler that raises an error, which terminates the program, or handle
        it in some fashion, in which case the cycles continue."""
        self.record_session_end(reason=ex)
        self.exception_handler(ex)
        raise ex
        # import signal
        # def keyboard_interrupt_handler(signum, frame):
        #     print(signum, frame)
        #     raise KeyboardInterrupt
        # signal.signal(signal.SIGINT, keyboard_interrupt_handler)


def qr(txt):
    qrcode = assert_package_installed("qrcode")
    img = qrcode.make(txt)
    file = P.tmpfile(suffix=".png")
    img.save(file.__str__())
    file()
    return file


if __name__ == '__main__':
    # Log.get_colorlog()
    Terminal().run("$profile", shell="pwsh").print()
