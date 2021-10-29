
from crocodile.core import np, os, get_time_stamp
from crocodile.file_management import sys, P, Struct


class Cycle:
    def __init__(self, c, name=''):
        self.c = c
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


class Experimental:
    """Debugging and Meta programming tools"""

    class Log:
        """Saves console output to a file."""

        def __init__(self, path=None):
            if path is None:
                path = P('console_output')
            self.path = path + '.log'
            sys.stdout = open(self.path, 'w')

        def finish(self):
            sys.stdout.close()
            print(f"Finished ... have a look @ \n {self.path}")

    @staticmethod
    def show_globals(globs):
        """Returns a struct with variables that are defined in the globals passed."""
        res = Struct(globs).spawn_from_keys(
            Struct(globs).keys().
                filter(lambda x: "__" not in x).
                filter(lambda x: not x.startswith("_")).
                filter(lambda x: x not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}))
        res.print()

    @staticmethod
    def assert_package_installed(package):
        """imports a package and installs it if not."""
        try:
            pkg = __import__(package)
            return pkg
        except ImportError:
            # import pip
            # pip.main(['install', package])
            import subprocess
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
            print(readmepath.with_name("source_code.zip").as_uri())

    @staticmethod
    def load_from_source_code(directory, obj=None):
        """Does the following:

        * Globs directory passed for ``source_code`` module.
        * Loads the directory to the memroy.
        * Returns either the package or a piece of it as indicated by ``obj``
        """
        tmpdir = P.tmp() / get_time_stamp(name="tmp_sourcecode")
        P(directory).find("source_code*", r=True).unzip(tmpdir)
        sys.path.insert(0, str(tmpdir))
        sourcefile = __import__(tmpdir.find("*").stem)
        if obj is not None:
            loaded = getattr(sourcefile, obj)
            return loaded
        else:
            return sourcefile

    @staticmethod
    def capture_locals(func, globs, args=None, self: str = None, update_globs=False):
        """Captures the local variables inside a function.
        :param func:
        :param globs: `globals()` executed in the main scope. This provides the function with modules defined in main.
        :param args: dict of what you would like to pass to the function as arguments.
        :param self: relevant only if the function is a method of a class. self refers to the name of the instance
        :param update_globs: binary flag refers to whether you want the result in a struct or update main."""
        code = Experimental.extract_code(func, args=args, self=self, include_args=False, verbose=False)

        print(code)
        res = Struct()
        exec(code, globs, res.dict)  # run the function within the scope `res`
        if update_globs:
            globs.update(res.dict)
        return res

    @staticmethod
    def run_globaly(func, globs, args=None, self: str = None):
        return Experimental.capture_locals(func=func, globs=globs, args=args, self=self, update_globs=True)

    @staticmethod
    def extract_code(func, args=None, self=None, include_args=True, verbose=True):
        """Takes in a function name, reads it source code and returns a new version of it that can be run in the main.
        This is useful to debug functions and class methods alike.
        Use: in the main: exec(extract_code(func)) or is used by `run_globally` but you need to pass globals()
        """
        if type(func) is str:
            self = ".".join(func.split(".")[:-1])
            func = eval(func)

        import inspect
        import textwrap

        codelines = textwrap.dedent(inspect.getsource(func))
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
        if include_args:
            code_string = Experimental.extract_arguments(func, args=args, self=self, verbose=verbose) + code_string

        clipboard = Experimental.assert_package_installed("clipboard")
        clipboard.copy(code_string)
        if verbose:
            print(f"code to be run extracted from {func.__name__} \n", code_string, "=" * 100)
        return code_string  # ready to be run with exec()

    @staticmethod
    def extract_arguments(func, self=None, args=None, globs=None, verbose=True):
        """Get code to define the args and kwargs defined in the main. Works for funcs and methods.
        """
        if type(func) is str:
            self = ".".join(func.split(".")[:-1])
            func = eval(func, globs)

        import inspect
        ak = Struct(dict(inspect.signature(func).parameters)).values()  # ignores self for methods.
        ak = Struct.from_keys_values(ak.name, ak.default)
        if args is not None:
            ak = ak.update(args)

        res = ""
        for key, val in ak.items():
            if key != "args" and key != "kwargs":
                if val is inspect._empty:  # not passed argument.
                    val = None
                    print(f'tb.Experimental Warning: arg {key} has no value. Now replaced with None.')
                res += f"{key} = " + (f"'{val}'" if type(val) is str else str(val)) + "\n"
        if self and inspect.ismethod(func):
            res += f"self = {self}\n"  # don't put string version of it.

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
        """Allows subseting an array of arbitrary shape, given which index to be subsetted and the range.
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
    def __init__(self, stdout=None, stderr=None):
        import subprocess
        self.subp = subprocess
        self.stdout = self.subp.DEVNULL if stdout is None else stdout
        self.stderr = self.subp.DEVNULL if stderr is None else stderr

    def run(self, command):
        resp = self.subp.run(["powershell", "-Command", command], capture_output=True, text=True)
        print(resp.stdout)
        print(resp.stderr)
        return resp

    def run_async(self, command):
        w = self.subp.Popen(["powershell", "-Command", f"{command}"], stdout=self.stdout, stderr=self.stderr)
        return w


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

