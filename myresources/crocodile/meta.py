
import logging
import dill
import subprocess
from crocodile.core import np, os, timestamp, randstr
from crocodile.file_management import sys, P, Struct


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
        strct = Struct(strct)
        super(DictCycle, self).__init__(c=strct.items(), **kwargs)
        self.keys = strct.keys()

    def set_key(self, key):
        self.index = self.keys.list.index(key)


class Experimental:
    """Debugging and Meta programming tools"""

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
        res = Struct(scope).spawn_from_keys(Struct(scope).keys())
        res = res.filter(lambda x: "__" not in x).filter(lambda x: not x.startswith("_"))
        res = res.filter(lambda x: x not in {"In", "Out", "get_ipython", "quit", "exit", "sys"})
        res.print(**kwargs)

    @staticmethod
    def assert_package_installed(package):
        """imports a package and installs it if not."""
        try:
            pkg = __import__(package)
            return pkg
        except ImportError:
            # import pip
            # pip.main(['install', package])
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        pkg = __import__(package)
        return pkg

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
        print(f"Successfully generated README.md file. Checkout:\n", readmepath.as_uri())

        if save_source_code:
            P(inspect.getmodule(obj).__file__).zip(op_path=readmepath.with_name("source_code.zip"))
            print("Source code saved @ " + readmepath.with_name("source_code.zip").as_uri())

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
        tmpdir.delete(are_you_sure=delete, verbose=False)
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
        code = Experimental.extract_code(func, args=args, self=self, include_args=False, verbose=False)

        print(code)
        res = Struct()
        exec(code, scope, res.dict)  # run the function within the scope `res`
        if update_scope:
            scope.update(res.dict)
        return res

    @staticmethod
    def run_globally(func, scope, args=None, self: str = None):
        return Experimental.capture_locals(func=func, scope=scope, args=args, self=self, update_scope=True)

    @staticmethod
    def extract_code(func, code: str = None, include_args=True, modules=None,
                     verbose=True, **kwargs):
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

        clipboard = Experimental.assert_package_installed("clipboard")
        clipboard.copy(code_string)
        if verbose:
            print(f"code to be run extracted from {func.__name__} \n", code_string, "=" * 100)
        return code_string  # ready to be run with exec()

    @staticmethod
    def extract_arguments(func, modules=None, exclude_args=True, verbose=True, **kwargs):
        """Get code to define the args and kwargs defined in the main. Works for funcs and methods.
        """
        if type(func) is str:  # will not work because once a string is passed, this method won't be able
            # to interpret it, at least not without the globals passed.
            self = ".".join(func.split(".")[:-1])
            _ = self
            func = eval(func, modules)

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
                        print(f'tb.Experimental Warning: arg {key} has no value. Now replaced with None.')
                if not flag:
                    res += f"{key} = " + (f"'{val}'" if type(val) is str else str(val)) + "\n"

        ak = inspect.getfullargspec(func)
        if ak.varargs:
            res += f"{ak.varargs} = (,)\n"
        if ak.varkw:
            res += f"{ak.varkw} = " + "{}\n"

        clipboard = Experimental.assert_package_installed("clipboard")
        clipboard.copy(res)
        if verbose:
            print("Finished. Paste code now.")
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
        # import importlib
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
        clipboard = Experimental.assert_package_installed("clipboard")
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
    def __init__(self, stdout=None, stderr=None, elevated=False):
        """
        Console
        Terminal
        Bash
        Shell
        Host
        * adding console to the begining of the command results in launching a new console that will not
        inherent from the console python was launched from (e.g. conda enviroment), unlike when console name is ignored.

        * `subprocess.Popen` (process open) is the most general command. Used here to create asynchronous job.
        * `subprocess.run` is a thin wrapper around Popen that makes it wait until it finishes the task.
        * `suprocess.call` is an archaic command for pre-Python-3.5.
        * In both `Popen` and `run`, the (shell=True) argument, implies that shell-specific commands are loaded up,
        e.g. `start` or `conda`.
        * To launch a new window, either use
        """
        self.available_consoles = ["cmd", "Command Prompt", "wt", "powershell", "wsl", "ubuntu", "pwsh"]
        self.stdout = subprocess.DEVNULL if stdout is None else stdout
        self.stderr = subprocess.DEVNULL if stderr is None else stderr
        self.elevated = elevated
        # https://stackoverflow.com/questions/130763/request-uac-elevation-from-within-a-python-script

    @staticmethod
    def is_admin():
        import ctypes
        return Experimental.try_this(lambda: ctypes.windll.shell32.IsUserAnAdmin(), otherwise=False)

    def run_command(self, command, console="powershell"):
        # alternative: res = subprocess.run("powershell -ls; dir", capture_output=True, shell=True, text=True)
        my_list = [console, "-Command"] if console is not None else []
        my_list.append(command)
        if self.elevated is False or self.is_admin():
            resp = subprocess.run(my_list, capture_output=True, text=True, shell=True)
        else:
            import ctypes
            resp = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        print(f"Crocodile: meta: Terminal: command execution response: {resp}")
        return resp

    def run_command_async(self, command, console=None):
        my_list = [console, "-Command"] if console is not None else []
        my_list.append(command)
        w = subprocess.Popen(my_list, stdout=self.stdout, stderr=self.stderr, shell=True)
        return w

    @staticmethod
    def open_console(console="", command="", shell=True, new_window=True, new_context=True):
        """
        :param console: default is same as the launching console. `cmd` doesn't recieve command, spoiler: not easy.
         `powershell` doesn't inherit venv. `wt` and `` works.
        :param command:
        :param shell:
        :param new_window:
        :param new_context:
        :return:
        """
        # This does not inherit from the from the shell launched python.
        if new_context:
            return subprocess.Popen(f'{"start" if new_window else ""} {console} {command}', shell=shell,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # this way, the new console inherits from its current context.
            return os.system(fr'{"start" if new_window else ""} {console} {command} \K')
            # /K remains the window, /C executes and dies (popup)

    def run_script(self, script, wdir=None, interactive=True, shell=True, delete=False, console="", new_window=True):
        wdir = wdir or P.cwd()
        header = f"""
import crocodile.toolbox as tb
tb.sys.path.insert(0, r'{wdir}')
"""  # header is necessary so import statements in the script passed are identified relevant to wdir.
        script = header + script
        script = f"""print(r'''{script}''')""" + "\n" + script
        file = P.tmpfile(name="tmp_python_script", suffix=".py", folder="tmpscripts")
        file.write_text(script)
        print(f"Script to be executed asyncronously: ", file.as_uri())
        self.open_console(console=console, command=f"ipython {'-i' if interactive else ''} {file}",
                          shell=shell, new_window=new_window)
        # python will use the same dir as the one from console this method is called.
        # file.delete(are_you_sure=delete, verbose=False)
        _ = delete
        # TODO: add return option (asynchronous programming)
        # command = f'ipython {"-i" if interactive else ""} -c "{script}"'

    def run_function(self, func):
        """Python brachnes off to a new window and execute the function passed.
        context can be either a pickled session or the current file __file__"""
        # step 1: pickle the function
        # step 2: create a script that unpickles it.
        # step 3: run the script that runs the function.
        # TODO complete this
        fname = P.tmpfile()
        dill.dump(obj=func, file=fname)
        script = f"""
import crocodile.toolbox as tb
func = tb.dill.unpickle({fname})
func()
"""
        # TODO: make sure that pickling function that serialize to tmp location delete this location afterwards
        self.run_script(script)


class Log:
    """This class is needed once a project grows beyond simple work. Simple print statements from
    dozens of objects will not be useful as the programmer will not easily recognize who
     is printing this message, in addition to many other concerns."""

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
    def get_coloredlogs(name=None, l_level=0, fmt=None, sep=" | "):
        coloredlogs = Experimental.assert_package_installed("coloredlogs")
        # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        level_styles = {'spam': {'color': 'green', 'faint': True},
                        'debug': {'color': 'white'},
                        'verbose': {'color': 'blue'},
                        'info': {'color': "green"},
                        'notice': {'color': 'magenta'},
                        'warning': {'color': 'yellow'},
                        'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True},
                        'critical': {'color': 'red', 'bold': True, "inverse": True}}
        field_styles = {'asctime': {'color': 'green'},
                        'hostname': {'color': 'magenta'},
                        'levelname': {'color': 'black', 'bold': True},
                        'name': {'color': 'blue'},
                        'programname': {'color': 'cyan'},
                        'username': {'color': 'yellow'}}
        logger = Log.get_base_logger(logging, name=name, l_level=l_level)
        coloredlogs.install(logger=logger, name="lol_different_name", level=logging.NOTSET,
                            level_styles=level_styles, field_styles=field_styles,
                            fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger

    @staticmethod
    def get_colorlog(file_path=None, file=False, stream=True, name=None, fmt=None, sep=" | ",
                     s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG,
                     log_colors=None,
                     ):
        if log_colors is None:
            log_colors = {'DEBUG': 'bold_cyan',
                          'INFO': 'green',
                          'WARNING': 'yellow',
                          'ERROR': 'thin_red',
                          'CRITICAL': 'bold_red,bg_white',
                          }  # see here for format: https://pypi.org/project/colorlog/
        colorlog = Experimental.assert_package_installed("colorlog")
        logger = Log.get_base_logger(colorlog, name, l_level)
        fmt = colorlog.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), log_colors=log_colors)
        Log.add_handlers(logger, colorlog, file, f_level, file_path, fmt, stream, s_level)
        return logger

    @staticmethod
    def get_logger(file_path=None, file=False, stream=True, name=None, fmt=None, sep=" | ",
                   s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG):
        """Basic Python logger."""
        logger = Log.get_base_logger(logging, name, l_level)
        fmt = logging.Formatter(fmt or Log.get_format(sep))
        Log.add_handlers(logger, logging, file, f_level, file_path, fmt, stream, s_level)
        return logger

    @staticmethod
    def get_base_logger(module, name, l_level):
        if name is None: print(f"Logger name not passed. It is preferable to pass a name indicates the owner.")
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
        print(f"Stream handler for Logger ({module})`{logger.name}` is created.")

    @staticmethod
    def add_filehandler(logger, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="fileHandler"):
        if file_path is None:
            file_path = P.tmpfile(name="logger", suffix=".log", folder="loggers")
        fhandler = logging.FileHandler(filename=str(file_path), mode=mode)
        fhandler.setFormatter(fmt=fmt)
        fhandler.setLevel(level=f_level)
        fhandler.set_name(name)
        logger.addHandler(fhandler)
        print(f"File handler for Logger `{logger.name}` is created @ " + file_path.as_uri())

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


if __name__ == '__main__':
    # Log.get_colorlog()
    pass
