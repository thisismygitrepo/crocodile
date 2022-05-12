

from crocodile.toolbox import *
from glob import glob


class Browse(object):
    def __init__(self, path, directory=True):
        # Create an attribute in __dict__ for each child
        self.__path__ = path
        if directory:
            sub_paths = glob(os.path.join(path, '*'))
            names = [os.path.basename(i) for i in sub_paths]
            # this is better than listdir, gives consistent results with glob
            for file, full in zip(names, sub_paths):
                key = P(file).make_valid_filename()
                setattr(self, 'FDR_' + key if os.path.isdir(full) else 'FLE_' + key,
                        full if os.path.isdir(full) else Browse(full, False))

    def __getattribute__(self, name):
        if name == '__path__':
            return super().__getattribute__(name)
        d = super().__getattribute__('__dict__')
        if name in d:
            child = d[name]
            if isinstance(child, str):
                child = Browse(child)
                setattr(self, name, child)
            return child
        return super().__getattribute__(name)

    def __repr__(self):
        return self.__path__

    def __str__(self):
        return self.__path__


def browse(path, depth=2, width=20):
    """
    :param width: if there are more than this items in a directory, dont' parse the rest.
    :param depth: to prevent crash, limit how deep recursive call can happen.
    :param path: absolute path
    :return: constructs a class dynamically by using object method.
    """
    if depth > 0:
        my_dict = {'z_path': P(path)}  # prepare _path attribute which returns current path from the browser object
        val_paths = glob(os.path.join(path, '*'))  # prepare other methods that refer to the contents.
        temp = [os.path.basename(i) for i in val_paths]
        # this is better than listdir, gives consistent results with glob (no hidden files)
        key_contents = []  # keys cannot be folders/file names immediately, there are caveats.
        for akey in temp:
            # if not akey[0].isalpha():  # cannot start with digit or +-/?.,<>{}\|/[]()*&^%$#@!~`
            #     akey = '_' + akey
            for i in string.punctuation.replace('_', ' '):  # disallow punctuation and space except for _
                akey = akey.replace(i, '_')
            key_contents.append(akey)  # now we have valid attribute path
        for i, (akey, avalue) in enumerate(zip(key_contents, val_paths)):
            if i < width:
                if os.path.isfile(avalue):
                    my_dict['FLE_' + akey] = P(avalue)
                else:
                    my_dict['FDR_' + akey] = browse(avalue, depth=depth - 1)

        def repr_func(self):
            if self.z_path.is_file():
                return 'Explorer object. File: \n' + str(self.z_path)
            else:
                return 'Explorer object. Folder: \n' + str(self.z_path)

        def str_func(self):
            return str(self.z_path)

        my_dict["__repr__"] = repr_func
        my_dict["__str__"] = str_func
        my_class = type(os.path.basename(path), (), dict(zip(my_dict.keys(), my_dict.values())))
        return my_class()
    else:
        return path


def accelerate(func, ip):
    """ Conditions for this to work:
    * Must run under __main__ context
    * func must be defined outside that context.
    To accelerate IO-bound process, use multithreading. An example of that is somthing very cheap to process,
    but takes a long time_produced to be obtained like a request from server. For this, multithreading launches all threads
    together, then process them in an interleaved fashion as they arrive, all will line-up for same processor,
    if it happens that they arrived quickly.
    To accelerate processing-bound process use multiprocessing, even better, use Numba.
    Method1 use: multiprocessing / multithreading.
    Method2: using joblib (still based on multiprocessing)
    from joblib import Parallel, delayed
    Fast method using Concurrent module
    """
    split = np.array_split(ip, os.cpu_count())
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor: op = list(executor.map(func, split))
    return np.concatenate(op, axis=0)


class SaveDecorator(object):
    def __init__(self, func, ext=""): self.func, self.ext = func, ext

    @classmethod
    def init(cls, func=None, **kwargs):
        """Always use this method for construction."""
        if func is None:  # User instantiated the class with no func argument and specified kwargs.
            def wrapper(func_): return cls(func_, **kwargs)
            return wrapper  # a function ready to be used by Python (pass func to it to instantiate it)
        else: return cls(func)  # return instance of the class. # called by Python with func passed and user did not specify non-default kwargs:

    def __call__(self, path=None, obj=None, **kwargs):
        path = Path(__import__("tempfile") .mkdtemp() + "-" + timestamp() + self.ext) if path is None else Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self.func(path, obj, **kwargs)
        print(f"File {obj} saved @ ", path.absolute().as_uri(), ". Directory: ", path.parent.absolute().as_uri())
        return path


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
    if wait: proc_handle = proce_info['hProcess']; _ = win32event.WaitForSingleObject(proc_handle, win32event.INFINITE); return win32process.GetExitCodeProcess(proc_handle)
    else: return None


#
# class Base:
#     def save_code(self, path):  # a usecase for including code in the save is when the source code is continously changing and still you want to reload an old version."""
#         module = __import__("inspect").getmodule(self)
#         if hasattr(module, "__file__"): file = Path(module.__file__)
#         else: raise FileNotFoundError(f"Attempted to save code from a script running in interactive session! module should be imported instead.")
#         Path(path).expanduser().write_text(file.read_text()); return Path(path) if type(path) is str else path  # path could be tb.P, better than Path
#
#     def save(self, path=None, add_suffix=True, data_only=False, save_code=False):
#         """Pickles the object.
#         * `data_only` is True: the object (self) will be pickled altogether.
#             * Ups: * convenient since Dill can, by data_only import all the required classes to __main__ IF the object is restored while directory is @ the same location object was created, thus,
#             * Downs: * It requires (in case of composed objects) that every sub-object is well-behaved and has the appropriate state methods implemented and the libraries imported must be present at load time_produced.
#         * `data_only` is False: means to save __getstate__.
#             * Ups: * very safe and unlikely to cause import errors at load time_produced. * User is responsible for loading up the class.
#             * Downs: the class data_only is required later and the `from_pickled_state` method should be used to reload the instance again.  __init__ method will be used again at reconstruction time_produced of the object before the attributes are monkey-patched. It is very arduous to design __init__ method that is convenient (uses plethora of default arguments) and works at the same time_produced with no input at reconstruction time_produced.
#         * add_suffix: if True, the suffixes `.pkl` and `.py` will be added to the file name.
#         * Tip: whether pickling the class or its data alone, always implement __getstate__ appropriately to avoid security risk involved in pickling objects that reference sensitive information like tokens and passwords.
#         """
#         path = str(path or Path.home().joinpath(f"tmp_results/tmp_files/{randstr()}"))
#         if add_suffix: path += ("" if self.__class__.__name__ in path else ("." + self.__class__.__name__)) + ("" if (not data_only or ".dat" in path) else ".dat")  # Fruthermore, .zip or .pkl will be added later depending on `save_code` value, warning will be raised.
#         path = Path(path).expanduser().resolve()
#         if not data_only: obj = self  # Choosing what object to pickle:
#         else:
#             obj = self.__getstate__(); obj = obj.copy()  # do not mess with original __dict__
#             for key, val in obj.items():
#                 if Base in val.__class__.__mro__:  # a class instance rather than pure data
#                     val.save(path=path, data_only=data_only, save_code=save_code)
#                     obj[key] = None  # this tough object is finished, the rest should be easy.
#         if save_code is True:
#             temp_path = Path().home().joinpath(f"tmp_results/tmp_zipping/{randstr()}"); temp_path.mkdir(parents=True, exist_ok=True)
#             self.save_code(path=temp_path.joinpath(f"source_code_{randstr()}.py"))
#             Save.pickle(path=temp_path.joinpath("class_data" + path.name), obj=obj, verbose=False, add_suffix=add_suffix)
#             result_path = Path(__import__("shutil").make_archive(base_name=str(path), format="zip", root_dir=str(temp_path), base_dir="."))
#             print(f"Code and data for the object ({repr(obj)}) saved @ `{result_path.as_uri()}`, Directory: `{result_path.parent.as_uri()}`")
#         else: result_path = Save.pickle(obj=obj, path=path, verbose=False, add_suffix=add_suffix); print(f"{'Data of' if data_only else ''} Object ({Display.f(repr(obj), 50)}) saved @ `{result_path.absolute().as_uri()}`, Directory: `{result_path.parent.absolute().as_uri()}`")
#         return result_path
#
#     @classmethod
#     def from_saved(cls, path, *args, r=False, scope=None, **kwargs):
#         """methodology: 1- load state, 2- load code. 3- initialize from __init__, 4- populate __dict__
#         :param path: points to where the `data` of this class is saved
#         :param r: recursive flag. If set to True, then, the directory containing the file `path` will be searched for
#         files of certain extension (.pkl or .zip) and will be loaded up in similar fashion and added as attributes
#         (to be implemented).
#         :param scope: dict of classes that are themselves attributes of the object to be loaded (will be loaded in as similar fashion to this method, if `r` is set to True). If scope are not passed and `r` is True, then, the classes will be assumed in [global scope, loaded from code].
#         """
#         assert ".dat." in str(path), f"Are you sure the path {path} is pointing to pickeld state of {cls}?"
#         data = dill.loads(Path(path).read_bytes())  # ============================= step 1: unpickle the data
#         inst = cls(*args, **kwargs)  # ============================= step 2: initialize the class
#         inst.__setstate__(dict(data))  # ===========step 3: update / populate instance attributes with data.
#         if r:  # ============================= step 4: check for saved attributes.
#             for key, _ in data.items():  # val is probably None (if init was written properly)
#                 for item in Path(path).parent.glob("*.zip"):  # if key is not an attribute of the object, skip it.
#                     setattr(inst, key, Base.from_zipped_code_state(path=Path(path).parent.joinpath(item.stem + ".zip"), r=True, scope=scope, **kwargs))
#         return inst
#
#     @staticmethod
#     def from_code_and_state(code_path, data_path=None, class_name=None, r=False, *args, **kwargs):
#         sys.path.insert(0, str(Path(code_path).parent)); return getattr(__import__("importlib").import_module(Path(code_path).stem), class_name).from_state(data_path, *args, r=r, **kwargs)
#
#     @staticmethod
#     def from_zipped_code_state(path, *args, class_name=None, r=False, scope=None, **kwargs):
#         """A simple wrapper on top of `from_code_and_state` where file passed is zip archive holding both source code and data. Additionally, the method gives the option to ignore the source code
#         saved and instead use code passed through `scope` which is a dictionary, e.g. globals(), in which the class of interest can be found [className -> module]."""
#         temp_path = Path.home().joinpath(f"tmp_results/tmp_unzipped/{Path(path).name.split('.zip')[1]}_{randstr()}")
#         with __import__("ZipFile").ZipFile(str(path), 'r') as zipObj: zipObj.extractall(temp_path)
#         code_path, data_path = list(temp_path.glob("source_code*"))[0], list(temp_path.glob("class_data*"))[0]
#         if ".dat." in str(data_path):  # loading the state and initializing the class
#             class_name = class_name or str(data_path).split(".")[1]
#             return Base.from_code_and_state(*args, code_path=code_path, data_path=data_path, class_name=class_name, r=r, **kwargs) if scope is None else scope[class_name].from_saved()
#         if scope: print(f"Warning: global scope has been contaminated by loaded scope {code_path} !!"); scope.update(__import__("runpy").run_path(str(code_path)))  # Dill will no longer complain.
#         return dill.loads(data_path.read_bytes())

    #
    # @staticmethod
    # def edit_source(module, *edits):
    #     sourcelines, line_idx = P(module.__file__).read_text().split("\n"), 0
    #     for edit_idx, edit in enumerate(edits):
    #         for line_idx, line in enumerate(sourcelines):
    #             if f"here{edit_idx}" in line:
    #                 new_line = line.replace(edit[0], edit[1])
    #                 print(f"Old Line: {line}\nNew Line: {new_line}")
    #                 if new_line == line: raise KeyError(f"Text Not found.")
    #                 sourcelines[line_idx] = new_line
    #                 break
    #         else: raise KeyError(f"No marker found in the text. Place the following: 'here{line_idx}'")
    #     P(module.__file__).write_text("\n".join(sourcelines))
    #     return __import__("importlib").reload(module)
    #

# class Log(object):  #
#     def __init__(self, dialect=["colorlog", "logging", "coloredlogs"][0], name=None, file: bool = False, file_path=None, stream=True, fmt=None, sep=" | ",
#                  s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False, log_colors=None):
#         self.specs = dict(name=name, file=file, file_path=file_path, stream=stream, fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level)  # save speces that are essential to re-create the object at
#         self.dialect = dialect  # specific to this class
#         self.verbose = verbose  # specific to coloredlogs dialect
#         self.log_colors = log_colors  # specific kwarg to colorlog dialect
#         self._install()
#         super().__init__(name=name)
#         self.specs["path"] = self.logger.name
#         if file: self.specs["file_path"] = self.logger.handlers[0].baseFilename  # first handler is a file handler
#
#     file = property(lambda self: P(self.specs["file_path"]) if self.specs["file_path"] else None)
#     def close(self): raise NotImplementedError
#     def get_shandler(self, first=True): shandlers = List(handler for handler in self.logger.handlers if "StreamHandler" in str(handler)); return shandlers[0] if first else shandlers
#     def get_fhandler(self, first=True): fhandlers = List(handler for handler in self.logger.handlers if "FileHandler" in str(handler)); return fhandlers[0] if first else fhandlers
#     def set_level(self, level, which=["logger", "stream", "file", "all"][0]): self.logger.setLevel(level) if which in {"logger", "all"} else None; self.get_shandler().setLevel(level) if which in {"stream", "all"} else None; self.get_fhandler().setLevel(level) if which in {"file", "all"} else None
#     def _install(self):  # populates self.logger attribute according to specs and dielect.
#         if self.specs["file"] is False and self.specs["stream"] is False: self.logger = Null()
#         elif self.dialect == "colorlog": self.logger = Log.get_colorlog(log_colors=self.log_colors, **self.specs)
#         elif self.dialect == "logging": self.logger = Log.get_logger(**self.specs)
#         elif self.dialect == "coloredlogs": self.logger = Log.get_coloredlogs(verbose=self.verbose, **self.specs)
#         else: self.logger = Log.get_colorlog(**self.specs)
#     def __setstate__(self, state):
#         self.__dict__ = state   # this way of creating relative path makes transferrable across machines.
#         self.specs["file_path"] = P(self.specs["file_path"]).rel2home() if self.specs["file_path"] is not None else None; self._install()
#     def __getstate__(self):  # logger can be pickled without this method, but its handlers are lost, so what's the point? no perfect reconstruction.
#         state = self.__dict__.copy(); state["specs"] = state["specs"].copy(); del state["logger"]
#         state["specs"]["file_path"] = P(self.specs["file_path"]).expanduser() if self.specs["file_path"] is not None else None; return state
#     def __repr__(self): return "".join([f"{self.logger} with handlers: \n"] + [repr(h) + "\n" for h in self.logger.handlers])
#     get_basic_format = staticmethod(lambda: logging.BASIC_FORMAT)
#     get_format = staticmethod(lambda sep: f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s{sep}%(levelno)s{sep}%(message)s{sep}")  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes
#     test_all = staticmethod(lambda: [Log.test_logger(logger) if not print("=" * 100) else None for logger in [Log.get_logger(), Log.get_colorlog(), Log.get_coloredlogs()]])
#     @staticmethod
#     def manual_degug(path): sys.stdout = open(path, 'w'); sys.stdout.close(); print(f"Finished ... have a look @ \n {path}")  # all print statements will write to this file.
#     @staticmethod
#     def get_coloredlogs(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, verbose=False):
#         level_styles = {'spam': {'color': 'green', 'faint': True}, 'debug': {'color': 'white'}, 'verbose': {'color': 'blue'}, 'info': {'color': "green"}, 'notice': {'color': 'magenta'}, 'warning': {'color': 'yellow'}, 'success': {'color': 'green', 'bold': True},
#                         'error': {'color': 'red', "faint": True, "underline": True}, 'critical': {'color': 'red', 'bold': True, "inverse": False}}  # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
#         field_styles = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'color': 'black', 'bold': True}, 'path': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}}
#         coloredlogs = install_n_import("coloredlogs")
#         if verbose: logger = install_n_import("verboselogs").VerboseLogger(name=name); logger.setLevel(l_level)  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
#         else: logger = Log.get_base_logger(logging, name=name, l_level=l_level); Log.add_handlers(logger, module=logging, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
#         coloredlogs.install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
#         return logger
#     @staticmethod
#     def get_colorlog(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG, log_colors=None, ):
#         log_colors = log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'thin_red', 'CRITICAL': 'fg_bold_red', }  # see here for format: https://pypi.org/project/colorlog/
#         colorlog = install_n_import("colorlog"); logger = Log.get_base_logger(colorlog, name, l_level)
#         Log.add_handlers(logger, colorlog, file, f_level, file_path, colorlog.ColoredFormatter(fmt or (rf"%(log_color)s" + Log.get_format(sep)), log_colors=log_colors), stream, s_level); return logger
#     @staticmethod
#     def get_logger(name=None, file=False, file_path=None, stream=True, fmt=None, sep=" | ", s_level=logging.DEBUG, f_level=logging.DEBUG, l_level=logging.DEBUG):  # Basic Python logger."""
#         logger = Log.get_base_logger(logging, name, l_level); Log.add_handlers(logger, logging, file, f_level, file_path, logging.Formatter(fmt or Log.get_format(sep)), stream, s_level); return logger
#     @staticmethod
#     def get_base_logger(module, name, l_level):
#         if name is None: print(f"Logger name not passed. It is preferable to pass a name indicating the owner.")
#         else: print(f"Logger `{name}` from `{module.__name__}` is instantiated with level {l_level}.")
#         logger = module.getLogger(name=name or randstr()); logger.setLevel(level=l_level)  # logs everything, finer level of control is given to its handlers
#         return logger  # https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
#     @staticmethod
#     def add_handlers(logger, module, file, f_level, file_path, fmt, stream, s_level):
#         if file or file_path:  Log.add_filehandler(logger, file_path=file_path, fmt=fmt, f_level=f_level)  # create file handler for the logger.
#         if stream: Log.add_streamhandler(logger, s_level, fmt, module=module)  # ==> create stream handler for the logger.
#     @staticmethod
#     def add_streamhandler(logger, s_level=logging.DEBUG, fmt=None, module=logging, name="myStream"):
#         shandler = module.StreamHandler(); shandler.setLevel(level=s_level); shandler.setFormatter(fmt=fmt); shandler.set_name(name); logger.addHandler(shandler); print(f"    Level {s_level} stream handler for Logger `{logger.name}` is created.")
#     @staticmethod
#     def add_filehandler(logger, file_path=None, fmt=None, f_level=logging.DEBUG, mode="a", name="myFileHandler"):
#         if file_path is None: file_path = P.tmpfile(name="logger", suffix=".log", folder="tmp_loggers")
#         fhandler = logging.FileHandler(filename=str(file_path), mode=mode); fhandler.setFormatter(fmt=fmt); fhandler.setLevel(level=f_level); fhandler.set_name(name); logger.addHandler(fhandler)
#         print(f"    Level {f_level} file handler for Logger `{logger.name}` is created @ " + P(file_path).clickable())
#     @staticmethod
#     def test_logger(logger):
#         logger.debug("this is a debugging message"); logger.info("this is an informational message"); logger.warning("this is a warning message")
#         logger.error("this is an error message"); logger.critical("this is a critical message"); [logger.log(msg=f"This is a message of level {level}", level=level) for level in range(0, 60, 5)]


"""This is based on `dill` package. While very flexible, it comes at the cost of assuming so many packages are loaded up and it happens implicitly. It often fails at load time_produced and requires same packages to be reloaded first.
         Compared to vanilla pickle, the former always raises an error when cannot pickle an object due to dependency. Dill however, stores all the required packages for any attribute object, but not the class data_only, or the classes that it inherits (at least at with this version)."""



if __name__ == '__main__':
    pass
