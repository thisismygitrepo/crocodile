"""
A collection of classes extending the functionality of Python's builtins.
email programmer@usa.com
"""

# Typing
# Path
import os
import sys
from pathlib import Path
import string
import random

# Numerical
import numpy as np
# import pandas as pd  # heavy weight, avoid unless necessary.
# Meta
import dill
import copy
from datetime import datetime
import datetime as dt  # useful for deltatime and timezones.

_ = dt


# ============================== Accessories ============================================


def timestamp(fmt=None, name=None):
    """isoformat is not compatible with file naming convention, this function provides compatible fmt
    tip: do not use this to create random addresses as it fails at high speed runs. Random string is better."""
    if fmt is None:
        fmt = '%Y-%m-%d-%I-%M-%S-%p-%f'
    _ = datetime.now().strftime(fmt)
    if name:
        name = name + '_' + _
    else:
        name = _
    return name


def str2timedelta(past):
    """Converts a human readable string like '1m' or '1d' to a timedate object.
    In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    sc = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks",
          "M": "months", "y": "years"}
    key, val = sc[past[-1]], eval(past[:-1])
    if key == "months":
        key = "days"
        val = val * 30
    elif key == "years":
        key = "weeks"
        val = val * 52
    return dt.timedelta(**{key: val})


def randstr(length=10, lower=True, upper=True, digits=True, punctuation=False, safe=False):
    if safe:
        import secrets  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
        return secrets.token_urlsafe(length)
    pool = "" + (string.ascii_lowercase if lower else "")
    pool = pool + (string.ascii_uppercase if upper else "")
    pool = pool + (string.digits if digits else "")
    pool = pool + (string.punctuation if punctuation else "")
    result_str = ''.join(random.choice(pool) for _ in range(length))
    return result_str


def assert_package_installed(package):
    """imports a package and installs it if not."""
    try:
        pkg = __import__(package)
        return pkg
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    pkg = __import__(package)
    return pkg


# ====================================== Classes ====================================


class SaveDecorator(object):
    def __init__(self, func, ext=""):
        # TODO: migrate from save_decorator to SaveDecorator
        # Called with func argumen when constructing the decorated function.
        # func argument is passed implicitly by Python.
        self.func = func
        self.ext = ext

    @classmethod
    def init(cls, func=None, **kwargs):
        """Always use this method for construction."""
        if func is None:  # User instantiated the class with no func argument and specified kwargs.
            def wrapper(func_):
                return cls(func_, **kwargs)

            return wrapper  # a function ready to be used by Python (pass func to it to instantiate it)
        else:  # called by Python with func passed and user did not specify non-default kwargs:
            return cls(func)  # return instance of the class.

    def __call__(self, path=None, obj=None, **kwargs):
        # Called when calling the decorated function (instance of this called).
        if path is None:
            import tempfile
            path = Path(tempfile.mkdtemp() + "-" + timestamp() + self.ext)
            # raise ValueError
        else:
            if not str(path).endswith(self.ext):
                # path = P(str(path) + self.ext)
                raise ValueError
            else:
                # path = P(path)
                raise ValueError

        # noinspection PyUnreachableCode
        path.parent.mkdir(exist_ok=True, parents=True)
        self.func(path, obj, **kwargs)
        print(f"File {obj} saved @ ", path.absolute().as_uri(), ". Directory: ", path.parent.absolute().as_uri())
        return path


def save_decorator(ext=""):
    """Apply default paths, add extension to path, print the saved file path"""

    def decorator(func):
        def wrapper(obj, path=None, verbose=True, **kwargs):
            if path is None:
                path = Path.home().joinpath("tmp_results").joinpath(randstr() + ext)
                print(f"tb.core: Warning: Path not passed to {func}. "
                      f"A default path has been chosen: {path.absolute().as_uri()}")
                # raise ValueError
            else:
                if not str(path).endswith(ext):
                    path = Path(str(path) + ext)
                    print(f"tb.core: Warning: suffix {ext} is added to path passed {path.as_uri()}")
                else:
                    path = Path(path)

            path.parent.mkdir(exist_ok=True, parents=True)
            func(path=path, obj=obj, **kwargs)
            if verbose:
                rep = repr(obj)
                rep = rep if len(rep) < 50 else rep[:10] + "... "
                print(f"SAVED {rep}  @ `{path.absolute().as_uri()}` |  Directory: `{path.parent.absolute().as_uri()}`")
            return path

        return wrapper

    return decorator


class Save:
    @staticmethod
    @save_decorator(".csv")
    def csv(obj, path=None):
        # obj.to_frame('dtypes').reset_index().to_csv(P(path).append(".dtypes").string)
        obj.to_frame('dtypes').reset_index().to_csv(path + ".dtypes")

    @staticmethod
    @save_decorator(".npy")
    def npy(obj, path, **kwargs):
        np.save(path, obj, **kwargs)

    @staticmethod
    @save_decorator(".mat")
    def mat(mdict, path=None, **kwargs):
        """
        .. note::
            Avoid using mat for saving results because of incompatiblity:

            * `None` type is not accepted.
            * Scalars are conveteed to [1 x 1] arrays.
            * etc. As such, there is no gaurantee that you restore what you saved.

            Unless you want to pass the results to Matlab animals, avoid this format.
        """
        from scipy.io import savemat
        for key, value in mdict.items():
            if value is None:
                mdict[key] = []
        savemat(str(path), mdict, **kwargs)

    @staticmethod
    @save_decorator(".json")
    def json(obj, path=None, **kwargs):
        """This format is **compatible** with simple dictionaries that hold strings or numbers
         but nothing more than that.
        E.g. arrays or any other structure. An example of that is settings dictionary.
        It is useful for to generate human-readable file."""
        import json

        with open(str(path), "w") as file:
            json.dump(obj, file, default=lambda x: x.__dict__, **kwargs)

    @staticmethod
    @save_decorator
    def yaml(obj, path, **kwargs):
        import yaml
        with open(str(path), "w") as file:
            yaml.dump(obj, file, **kwargs)

    @staticmethod
    @save_decorator(".pkl")
    def vanilla_pickle(obj, path, **kwargs):
        import pickle
        with open(str(path), 'wb') as file:
            pickle.dump(obj, file, **kwargs)

    @staticmethod
    @save_decorator(".pkl")
    def pickle(obj=None, path=None, r=False, **kwargs):
        """This is based on `dill` package. While very flexible, it comes at the cost of assuming so many packages are
        loaded up and it happens implicitly. It often fails at load time and requires same packages to be reloaded first
        . Compared to vanilla pickle, the former always raises an error when cannot pickle an object due to
        dependency. Dill however, stores all the required packages for any attribute object, but not the class itself,
        or the classes that it inherits (at least at with this version)."""
        with open(str(path), 'wb') as file:
            dill.dump(obj, file, recurse=r, **kwargs)

    @staticmethod
    def pickle_s(obj):
        binary = dill.dumps(obj)
        return binary


class Base(object):
    def __init__(self, *args, **kwargs):
        pass

    def __getstate__(self):
        """This method is used by Python internally when an instance of the class is pickled.
        attributes that you do not want to be pickled for one reason or the other, should be omitted from the
        returned dictionary (as opposed to setting them to None)."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """The solution to recover missing objects from state, is dependent on where it came from.
        If the attribute was instantiated by the class itself, then, similar thing should happen here.
        If the object was passed to the __init__ by the caller, it should be passed again. For the meanwhile it should
        be set to None."""
        self.__dict__.update(state)

    def save_code(self, path):
        """a usecase for including code in the save is when the source code is continously
         changing and still you want to reload an old version."""
        import inspect
        module = inspect.getmodule(self)
        if hasattr(module, "__file__"): file = Path(module.__file__)
        else: raise FileNotFoundError(f"Attempted to save code from a script running in interactive session! "
                                      f"module should be imported instead.")
        Path(path).write_text(file.read_text())
        return Path(path) if type(path) is str else path  # path could be tb.P, better than Path

    def save(self, path=None, itself=True, r=False, include_code=False):
        """Pickles the object.
        :param path: destination file.
        :param itself: `itself` means the object (self) will be pickled straight away. This is the default behaviour,
        however, it requires (in case of composed objects) that every sub-object is well-behaved and has the appropriate
        state methods implemented. The alternative to this option (itself=False) is to save __dict__ only
         (assuming it is pure data rather than code, otherwise recusive flag must be set), then the class itself is
          required later and the `from_pickled_state` method should be used to reload the instance again.
          The disadvantage of this method is that __init__ method will be used again at reconstruction time
           of the object before the attributes are monkey-patched.
           It is very arduous to design __init__ method that is convenient (uses plethora of
          default arguments) and works at the same time with no input at reconstruction time.
          # Use only for classes with whacky behaviours or too expensive to redesign
            # methodology: 1- Save state, 2- save code. 3- initialize from __init__, 4- populate __dict__
        :param include_code: `save_code` will be called.
        :param r: recursive flag.

        * Dill package manages to resconstruct the object by loading up all the appropriate libraries again
        IF the object is restored while directory is @ the same location object was created, thus,
        no need for saving code or even reloading it.
        * Beware of the security risk involved in pickling objects that reference sensitive information like tokens and
        passwords. The best practice is to pass them again at load time.
        """
        path = str(path or Path.home().joinpath(f"tmp_results/tmpfiles/{randstr()}"))
        path = Path(path + "." + self.__class__.__name__ + ("" if itself else ".dat"))
        # Fruthermore, .zip or .pkl will be added later depending on `include_code` value, warning will be raised.

        # Choosing what object to pickle:
        if itself: obj = self
        else:
            obj = self.__getstate__()
            if r:
                obj = obj.copy()  # do not mess with original __dict__
                for key, val in obj.items():
                    if Base in val.__class__.__mro__:  # a class instance rather than pure data
                        val.save(itself=itself, include_code=include_code, path=path, r=r)
                        obj[key] = None  # this tough object is finished, the rest should be easy.
                    else:
                        pass  # leave this object as is.

        if include_code is True:
            temp_path = Path().home().joinpath(f"tmp_results/zipping/{randstr()}")
            temp_path.mkdir(parents=True, exist_ok=True)
            self.save_code(path=temp_path.joinpath(f"source_code_{randstr()}.py"))
            Save.pickle(path=temp_path.joinpath("class_data"), obj=obj, r=r, verbose=False)
            import shutil
            result_path = shutil.make_archive(base_name=str(path), format="zip",
                                              root_dir=str(temp_path), base_dir=".")
            result_path = Path(result_path)
            print(f"Code and data for the object ({repr(obj)}) saved @ "
                  f"{result_path.as_uri()}, Directory: {result_path.parent.as_uri()}")
        else:
            result_path = Save.pickle(obj=obj, path=path, r=r, verbose=False)
            print(f"{'Data of' if itself else ''} Object ({repr(obj)}) saved @ "
                  f"{result_path.absolute().as_uri()}, Directory: {result_path.parent.absolute().as_uri()}")
        return result_path

    @classmethod
    def from_saved(cls, path, *args, r=False, scope=None, **kwargs):
        """Works in conjuction with save_pickle when `itself`=False, i.e. only state of object is pickled.

        :param path: points to where the `data` of this class is saved
        :param r: recursive flag. If set to True, then, the directory containing the file `path` will be searched for
        files of certain extension (.pkl or .zip) and will be loaded up in similar fashion and added as attributes
        (to be implemented).
        :param scope: dict of classes that are themselves attributes of the object to be loaded (will be loaded
        in as similar fashion to this method, if `r` is set to True). If scope are not passed and `r` is True,
        then, the classes will be assumed in [global scope, loaded from code].

        It is vital that __init__ method of the class is well behaved.  That is, class instance can be initialized
        with no or only fake inputs (use default args to achieve this behaviour), so that a skeleton instance
        can be easily made then attributes are updated from the data loaded from disc. A good practice is to add
        a flag (e.g. from_saved) to init method to require the special behaviour indicated above when it is raised,
         e.g. do NOT create some expensive attribute is this flag is raised because it will be obtained later.
        """

        # ============================= step 1: load up the data
        assert ".dat." in str(path), f"Are you sure the path {path} is pointing to pickeld state of {cls}?"
        data = dill.loads(Path(path).read_bytes())
        # ============================= step 2: initialize the class
        inst = cls(*args, **kwargs)
        # ============================= step 3: update / populate instance attributes with data.
        inst.__setstate__(dict(data))
        # ============================= step 4: check for saved attributes.
        if r:  # add further attributes to `inst`, if any.
            contents = [item.stem for item in Path(path).parent.glob("*.zip")]
            for key, _ in data.items():  # val is probably None (if init was written properly)
                if key in contents:  # if key is not an attribute of the object, skip it.
                    setattr(inst, key, Base.from_zipped_code_state(path=Path(path).parent.joinpath(key + ".zip"),
                                                                   r=True, scope=scope, **kwargs))
        return inst

    @staticmethod
    def from_code_and_state(code_path, data_path=None, class_name=None, r=False, *args, **kwargs):
        """A simple wrapper on tops of the class method `from_state`, where the class is to be loaded from
        a source code file.
        :param code_path:
        :param data_path:
        :param class_name: Assumed to inherit from `Base`, so it has mthod `from_state`.
        :param r:
        :param args:
        :param kwargs:
        :return:
        """
        code_path = Path(code_path)
        sys.path.insert(0, str(code_path.parent))
        import importlib
        sourcefile = importlib.import_module(code_path.stem)
        return getattr(sourcefile, class_name).from_state(data_path, *args, r=r, **kwargs)

    @staticmethod
    def from_zipped_code_state(path, *args, class_name=None, r=False, scope=None, **kwargs):
        """A simple wrapper on top of `from_code_and_state` where file passed is zip archive holding
        both source code and data. Additionally, the method gives the option to ignore the source code
        saved and instead use code passed through `scope` which is a dictionary, e.g. globals(), in which
        the class of interest can be found [className -> module].
        """
        fname = Path(path).name.split(".zip")[1]
        temp_path = Path.home().joinpath(f"tmp_results/unzipped/{fname}_{randstr()}")
        from zipfile import ZipFile
        with ZipFile(str(path), 'r') as zipObj:
            zipObj.extractall(temp_path)
        code_path = list(temp_path.glob("source_code*"))[0]
        data_path = list(temp_path.glob("class_data*"))[0]
        if ".dat." in str(data_path):  # loading the state and initializing the class
            class_name = class_name or str(data_path).split(".")[1]
            if scope is None:  # load from source code
                return Base.from_code_and_state(*args, code_path=code_path, data_path=data_path, class_name=class_name,
                                                r=r, **kwargs)
            else:  # use fresh scope passed.
                return scope[class_name].from_saved()

        else:  # file points to pickled object:
            if scope:
                import runpy
                mods = runpy.run_path(str(code_path))
                print(f"Warning: global scope has been contaminated by loaded scope {code_path} !!")
                scope.update(mods)  # Dill will no longer complain.
            obj = dill.loads(data_path.read_bytes())
            return obj

    def get_attributes(self, check_ownership=False, remove_base_attrs=True, return_objects=False,
                       fields=True, methods=True):
        attrs = list(filter(lambda x: ('__' not in x) and not x.startswith("_"), dir(self)))
        _ = check_ownership
        if remove_base_attrs:
            [attrs.remove(x) for x in Base().get_attributes(remove_base_attrs=False)]
        # if exclude is not None:
        #     [attrs.remove(x) for x in exlcude]
        import inspect
        if not fields:  # logic (questionable): anything that is not a method is a field
            attrs = list(filter(lambda x: inspect.ismethod(getattr(self, x)), attrs))
        
        if not methods:
            attrs = list(filter(lambda x: not inspect.ismethod(getattr(self, x)), attrs))

        if return_objects:
            # attrs = attrs.apply(lambda x: getattr(self, x))
            attrs = [getattr(self, x) for x in attrs]
        return List(attrs)

    def __deepcopy__(self, *args, **kwargs):
        """Literally creates a new copy of values of old object, rather than referencing them.
        similar to copy.deepcopy()"""
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(copy.deepcopy(self.__dict__))
        return obj

    def __copy__(self, *args, **kwargs):
        """Shallow copy. New object, but the keys of which are referencing the values from the old object.
        Does similar functionality to copy.copy"""
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(self.__dict__.copy())
        return obj

    def evalstr(self, string_, expected='self'):
        """This method allows other methods to parse strings that refer to the object themselves via `self`.
        This is a next level walrus operator where you can always refer to the object on the fly visa `self`
        string. It comes particularly handy during chaining in one-liners. There is no need to break chaining to
        get a handle of the latest object to reference it in a subsequent method."""
        # be wary of unintended behaciour if a string had `self` in it by coincidence.
        _ = self
        if type(string_) is str:
            if expected == 'func':
                return eval("lambda x: " + string_)
            elif expected == 'self':
                if "self" in string_:
                    return eval(string_)
                else:
                    return string_
        else:
            return string_

    def print(self, typeinfo=False):
        Struct(self.__dict__).print(dtype=typeinfo)

    def viz_heirarchy(self, depth=3, obj=None, filt=None):
        import objgraph
        import tempfile
        filename = Path(tempfile.gettempdir()).joinpath("graph_viz_" + randstr() + ".png")
        objgraph.show_refs([self] if obj is None else [obj], max_depth=depth, filename=str(filename), filter=filt)
        import sys
        if sys.platform == "win32":
            os.startfile(str(filename.absolute()))  # works for files and folders alike
        return filename


class List(list, Base):
    """Use this class to keep items of the same type.
    """

    # =============================== Constructor Methods ====================
    def __init__(self, obj_list=None):
        super().__init__()
        self.list = list(obj_list) if obj_list is not None else []

    def insert(self, __index: int, __object):
        self.list.insert(__index, __object)
        return self

    def __bool__(self):
        return bool(self.list)

    @classmethod
    def from_copies(cls, obj, count):
        return cls([copy.deepcopy(obj) for _ in range(count)])

    @classmethod
    def from_replicating(cls, func, *args, replicas=None, **kwargs):
        """
        :param args: could be one item repeated for all instances, or iterable. If iterable, it can by a Cycle object.
        :param kwargs: those could be structures:
        :param replicas:
        :param func:

        """
        if not args and not kwargs:  # empty args list and kwargs list
            return cls([func() for _ in range(replicas)])
        else:
            result = []
            for params in zip(*(args + tuple(kwargs.values()))):
                an_arg = params[:len(args)]
                a_val = params[len(args):]
                a_kwarg = dict(zip(kwargs.keys(), a_val))
                result.append(func(*an_arg, **a_kwarg))
            return cls(result)

    def save_items(self, directory, names=None, saver=None):
        if saver is None:
            saver = Save.pickle
        if names is None:
            names = range(len(self))
        for name, item in zip(names, self.list):
            saver(path=directory / name, obj=item)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
            _ = memodict
        return List([copy.deepcopy(i) for i in self.list])

    def __copy__(self):
        return List(self.list.copy())

    def __getstate__(self):
        return self.list

    def __setstate__(self, state):
        self.list = state

    # ================= call methods =====================================
    def method(self, name, *args, **kwargs):
        return List([getattr(i, name)(*args, **kwargs) for i in self.list])

    def attr(self, name):
        return List([getattr(i, name) for i in self.list])

    # def __getattribute__(self, item):
    #     # you can dispense with this method. Its only purpose is to make eaisr experience qwith the linter
    #     # obj = object.__getattribute__(self, "list")[0]
    #     # try:
    #     #     attr = object.__getattribute__(self, item)
    #     #     if hasattr(obj, item):
    #     #         return self.__getattr__(item)
    #     #     else:
    #     #         return attr
    #     # except AttributeError:
    #     #     return self.__getattr__(item)
    #     if item == "list":  # grant special access to this attribute.
    #         return object.__getattribute__(self, "list")
    #     if item in object.__getattribute__(self, "__dict__").keys():
    #         return self.__getattr__(item)
    #     else:
    #         return object.__getattribute__(self, item)

    def __getattr__(self, name):  # fallback position when normal mechanism fails.
        # this is called when __getattribute__ raises an error or call this explicitly.
        result = List([getattr(i, name) for i in self.list])
        return result

    def __call__(self, *args, lest=True, **kwargs):
        if lest:
            return List([i(*args, **kwargs) for i in self.list])
        else:
            return [i(*args, **kwargs) for i in self.list]

    # ======================== Access Methods ==========================================
    def __getitem__(self, key):
        if type(key) is list or type(key) is np.ndarray:  # to allow fancy indexing like List[1, 5, 6]
            return List([self[item] for item in key])
        elif type(key) is str:  # access keys like dictionaries.
            return List(item[key] for item in self.list)
        else:  # must be an integer or slice
            # behaves similarly to Numpy A[1] vs A[1:2]
            result = self.list[key]  # return the required item only (not a List)
            if type(key) is not slice:
                return result  # choose one item
            else:
                return List(result)

    def __setitem__(self, key, value):
        self.list[key] = value

    def sample(self, size=1, replace=False, p=None):
        """Select at random"""
        return self[np.random.choice(len(self), size, replace=replace, p=p)]

    def to_struct(self, key_val=None):
        """
        :param key_val: function that returns (key, value) pair.
        :return:
        """
        if key_val is None:
            def key_val(x):
                return str(x), x
        else:
            key_val = self.evalstr(key_val)
        # return Struct.from_keys_values_pairs(self.apply(key_val))
        # removed for disentanglement
        return dict(self.apply(key_val))

    # def find(self, patt, match="fnmatch"):
    #     """Looks up the string representation of all items in the list and finds the one that partially matches
    #     the argument passed. This method is a short for ``self.filter(lambda x: string_ in str(x))`` If you need more
    #     complicated logic in the search, revert to filter method.
    #     """
    #

    # if match == "string" or None:
    #     for idx, item in enumerate(self.list):
    #         if patt in str(item):
    #             return item
    # elif match == "fnmatch":
    #     import fnmatch
    #     for idx, item in enumerate(self.list):
    #         if fnmatch.fnmatch(str(item), patt):
    #             return item
    # else:  # "regex"
    #     # escaped = re.escape(string_)
    #     compiled = re.compile(patt)
    #     for idx, item in enumerate(self.list):
    #         if compiled.search(str(item)) is not None:
    #             return item
    # return None

    def index_items(self, idx):
        return List([item[idx] for item in self.list])

    def index(self, func, *args, **kwargs) -> list:
        """ A generalization of the `.index` method of `list`. It takes in a function rather than an
         item to find its index. Additionally, it returns full list of results, not just the first result.
        If you wanted the original index method, refer to .list attribute to use it.

        :param func:
        :return: List of indices of items where the function returns `True`.
        """
        func = self.evalstr(func, expected='func')
        res = []
        for idx, x in enumerate(self.list):
            if func(x):
                res.append(idx)
        return res

    # ======================= Modify Methods ===============================
    def flatten(self):
        res = self.list[0]
        for item in self.list[1:]:
            res = res + item
        return res

    def append(self, item):  # add one item to the list object
        self.list.append(item)
        return self

    def __add__(self, other):
        # implement coersion
        return List(self.list + list(other))

    def __radd__(self, other):
        return List(self.list + list(other))

    def __iadd__(self, other):  # inplace add.
        self.list = self.list + list(other)
        return self

    def __repr__(self):
        if len(self.list) > 0:
            tmp1 = f"List object with {len(self.list)} elements. One example of those elements: \n"
            tmp2 = f"{self.list[0].__repr__()}"
            return tmp1 + tmp2
        else:
            return f"An Empty List []"

    def __len__(self):
        return len(self.list)

    @property
    def len(self):
        return self.list.__len__()

    def __iter__(self):
        return iter(self.list)

    def apply(self, func, *args, other=None, jobs=None, depth=1, verbose=False, desc=None, **kwargs):
        """
        :param jobs:
        :param func: func has to be a function, possibly a lambda function. At any rate, it should return something.
        :param args:
        :param other: other list
        :param verbose:
        :param desc:
        :param depth: apply the function to inner Lists
        :param kwargs: a list of outputs each time the function is called on elements of the list.
        :return:
        """
        if depth > 1:
            depth -= 1
            # assert type(self.list[0]) == List, "items are not Lists".
            self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth, **kwargs))

        func = self.evalstr(func, expected='func')
        tqdm = assert_package_installed("tqdm").tqdm
        if other is None:
            iterator = self.list if not verbose else tqdm(self.list, desc=desc)
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(i, *args, **kwargs) for i in iterator))
            else:
                return List([func(x, *args, **kwargs) for x in iterator])
        else:
            iterator = zip(self.list, other) if not verbose else tqdm(zip(self.list, other), desc=desc)
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(x, y) for x, y in iterator))
            else:
                return List([func(x, y) for x, y in iterator])

    def modify(self, func, lest=None):
        """Modifies objects rather than returning new list of objects, hence the name of the method.
        :param func: a string that will be executed, assuming idx, x and y are given.
        :param lest:
        :return:
        """
        if lest is None:
            for x in self.list:
                _ = x
                exec(func)
        else:
            for idx, (x, y) in enumerate(zip(self.list, lest)):
                _ = idx, x, y
                exec(func)
        return self

    def sort(self, *args, **kwargs):
        self.list.sort(*args, **kwargs)
        return self

    def sorted(self, *args, **kwargs):
        return List(sorted(self.list, *args, **kwargs))

    def filter(self, func):
        if type(func) is str:
            func = eval("lambda x: " + func)
        result = List()
        for item in self.list:
            if func(item):
                result.append(item)
        return result

    def print(self, nl=1, sep=False, style=repr):
        for idx, item in enumerate(self.list):
            print(f"{idx:2}- {style(item)}", end=' ')
            for _ in range(nl):
                print('', end='\n')
            if sep:
                print(sep * 100)

    def to_series(self):
        import pandas as pd
        return pd.Series(self.list)

    def to_dataframe(self, names=None, minimal=False, obj_included=True):
        """

        :param names: name of each object.
        :param minimal: Return Dataframe structure without contents.
        :param obj_included: Include a colum for objects themselves.
        :return:
        """
        # DisplayData.set_pandas_display()  # removed for disentanglement
        columns = list(self.list[0].__dict__.keys())
        if obj_included or names:
            columns = ['object'] + columns
        import pandas as pd
        df = pd.DataFrame(columns=columns)
        if minimal:
            return df

        # Populate the dataframe:
        for i, obj in enumerate(self.list):
            if obj_included or names:
                if names is None:
                    name = [obj]
                else:
                    name = [names[i]]
                df.loc[i] = name + list(self.list[i].__dict__.values())
            else:
                df.loc[i] = list(self.list[i].__dict__.values())
        return df

    def to_numpy(self):
        return self.np

    @property
    def np(self):
        return np.array(self.list)


class Struct(dict):  # inheriting from dict gives `get` method.
    """Use this class to keep bits and sundry items.
    Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience
    """

    def save_json(self, path=None):
        path = Save.json(obj=self.__dict__, path=path)
        return path

    def save_yaml(self, path=None):
        return Save.yaml(obj=self.__dict__, path=path)

    def __len__(self):
        return len(self.keys())

    def __init__(self, dictionary=None, **kwargs):
        """
        :param dictionary: a dict, a Struct, None or an object with __dict__ attribute.
        """
        super(Struct, self).__init__()
        if type(dictionary) is Struct:
            dictionary = dictionary.dict
        if dictionary is None:  # only kwargs were passed
            final_dict = kwargs
        elif not kwargs:  # only dictionary was passed
            if type(dictionary) is dict:
                final_dict = dictionary
            elif type(dictionary) == "mappingproxy":
                final_dict = dict(dictionary)
            else:
                final_dict = dictionary.__dict__
        else:  # both were passed
            final_dict = dictionary if type(dictionary) is dict else dictionary.__dict__
            final_dict.update(kwargs)
        self.__dict__ = final_dict

    def to_default(self, default=lambda: None):
        from collections import defaultdict
        tmp2 = defaultdict(default)
        tmp2.update(self.__dict__)
        self.__dict__ = tmp2
        return self

    def __bool__(self):
        return bool(self.__dict__)

    @staticmethod
    def recursive_struct(mydict):
        struct = Struct(mydict)
        for key, val in struct.items():
            if type(val) is dict:
                struct[key] = Struct.recursive_struct(val)
        return struct

    @staticmethod
    def recursive_dict(struct):
        mydict = struct.dict
        for key, val in mydict.items():
            if type(val) is Struct:
                mydict[key] = Struct.recursive_dict(val)
        return mydict

    @classmethod
    def from_keys_values(cls, keys, values):
        """
        :rtype: Struct
        """
        return cls(dict(zip(keys, values)))

    @classmethod
    def from_keys_values_pairs(cls, my_list):
        res = dict()
        for k, v in my_list:
            res[k] = v
        return cls(res)

    @classmethod
    def from_names(cls, names, default_=None):  # Mimick NamedTuple and defaultdict
        if default_ is None:
            default_ = [None] * len(names)
        return cls.from_keys_values(names, values=default_)

    def get_values(self, keys):
        return List([self[key] for key in keys])

    @property
    def clean_view(self):

        class Temp:
            pass

        temp = Temp()
        temp.__dict__ = self.__dict__
        return temp

    def __repr__(self):
        repr_string = ""
        for key in self.keys().list:
            repr_string += str(key) + ", "
        return "Struct: [" + repr_string + "]"

    def print(self, sep=None, yaml=False, dtype=True, logger=False, limit=50):
        if bool(self) is False:
            print(f"Empty Struct.")
            return None  # break out of the function.
        if yaml:
            # removed for disentanglement
            # self.save_yaml(P.tmp(file="__tmp.yaml"))
            # txt = P.tmp(file="__tmp.yaml").read_text()
            # print(txt)
            return None
        if sep is None:
            sep = 5 + max(self.keys().apply(str).apply(len).list)
        repr_string = ""
        repr_string += "Structure, with following entries:\n"
        repr_string += "Key" + " " * sep + (("Item Type" + " " * sep) if dtype else "") + "Item Details\n"
        repr_string += "---" + " " * sep + (("---------" + " " * sep) if dtype else "") + "------------\n"
        for key in self.keys().list:
            key_str = str(key)
            type_str = str(type(self[key])).split("'")[1]
            val_str = DisplayData.get_repr(self[key], limit=limit).replace("\n", " ")
            repr_string += key_str + " " * abs(sep - len(key_str)) + " " * len("Key")
            if dtype:
                repr_string += type_str + " " * abs(sep - len(type_str)) + " " * len("Item Type")
            repr_string += val_str + "\n"
        if logger:
            return repr_string
        else:
            print(repr_string)
            return self

    def __str__(self, sep=",", newline="\n", breaklines=None):
        mystr = str(self.__dict__)
        mystr = mystr[1:-1].replace(":", " =").replace("'", "").replace(",", sep)
        if breaklines:
            res = np.array(mystr.split(sep))
            res = List(np.array_split(res, int(np.ceil((len(res) / breaklines))))).apply(lambda x: sep.join(x))
            import functools
            mystr = functools.reduce(lambda a, b: a + newline + b, res) if len(res) > 1 else res[0]
        return mystr

    def __getitem__(self, item):  # allows indexing into entries of __dict__ attribute
        return self.__dict__[item]  # thus, gives both dot notation and string access to elements.

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, item):  # this works better with the linter.
        try:
            return self.__dict__[item]
        except KeyError:
            # try:
            # super(Struct, self).__getattribute__(item)
            # object.__getattribute__(self, item)
            # except AttributeError:
            raise AttributeError(f"Could not find the attribute `{item}` in this Struct object.")

    def __getstate__(self):  # serialize
        return self.__dict__

    def __setstate__(self, state):  # deserialize
        self.__dict__ = state

    def __iter__(self):  # used when list(~) is called or it is iterated over.
        return iter(self.dict.items())

    @staticmethod
    def save_yaml(path):
        Save.yaml(path)

    @property
    def dict(self):  # allows getting dictionary version without accessing private memebers explicitly.
        return self.__dict__

    @dict.setter
    def dict(self, adict):
        self.__dict__ = adict

    def update(self, *args, **kwargs):
        """Accepts dicts and keyworded args
        """
        new_struct = Struct(*args, **kwargs)
        self.__dict__.update(new_struct.__dict__)
        return self

    def apply(self, func):
        func = self.evalstr(func)
        for key, val in self.items():
            self[key] = func(val)
        return self

    def inverse(self):
        return Struct({v: k for k, v in self.dict.items()})

    # def append_values(self, *others, **kwargs):
    #     """ """
    #     return Struct(self.concat_dicts(*((self.dict,) + others), **kwargs))

    @staticmethod
    def concat_values(*dicts, method=None, lenient=True, collect_items=False, clone=True):
        if method is None:
            method = list.__add__
        if not lenient:
            keys = dicts[0].keys()
            for i in dicts[1:]:
                assert i.keys() == keys
        # else if lenient, take the union
        if clone:
            total_dict = copy.deepcopy(dicts[0])  # take first dict in the tuple
        else:
            total_dict = dicts[0]  # take first dict in the tuple
        if collect_items:
            for key, val in total_dict.item():
                total_dict[key] = [val]

            def method(tmp1, tmp2):
                return tmp1 + [tmp2]

        if len(dicts) > 1:  # are there more dicts?
            for adict in dicts[1:]:
                for key in adict.keys():  # get everything from this dict
                    try:  # may be the key exists in the total dict already.
                        total_dict[key] = method(total_dict[key], adict[key])
                    except KeyError:  # key does not exist in total dict
                        if collect_items:
                            total_dict[key] = [adict[key]]
                        else:
                            total_dict[key] = adict[key]
        return Struct(total_dict)

    def keys(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.keys())

    def values(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.values())

    def items(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.items())

    def to_dataframe(self, *args, **kwargs):
        # return self.values().to_dataframe(names=self.keys())
        import pandas as pd
        return pd.DataFrame(self.__dict__, *args, **kwargs)

    def spawn_from_values(self, values):
        """From the same keys, generate a new Struct with different values passed."""
        return self.from_keys_values(self.keys(), self.evalstr(values, expected='self'))

    def spawn_from_keys(self, keys):
        """From the same values, generate a new Struct with different keys passed."""
        return self.from_keys_values(self.evalstr(keys, expected="self"), self.values())

    def plot(self, artist=None):
        if artist is None:
            # artist = Artist(figname='Structure Plot')
            # removed for disentanglement
            import matplotlib.pyplot as plt
            fig, artist = plt.subplots()
        for key, val in self:
            # if xdata is None:
            #     xdata = np.arange(len(val))
            artist.plot(val, label=key)
        try:
            artist.fig.legend()
        except AttributeError:
            pass
        return artist


class DisplayData:
    @staticmethod
    def set_pandas_display(rows=1000, columns=1000, width=1000, colwidth=40):
        import pandas as pd
        pd.set_option('display.max_colwidth', colwidth)
        pd.set_option('display.max_columns', columns)  # to avoid replacing them with ...
        pd.set_option('display.width', width)  # to avoid wrapping the table.
        pd.set_option('display.max_rows', rows)  # to avoid replacing rows with ...

    @staticmethod
    def set_pandas_auto_width():
        """For fixed width host windows, this is recommended to avoid chaos due to line-wrapping."""
        import pandas as pd
        pd.options.display.width = 0  # this way, pandas is told to detect window length and act appropriately.

    @staticmethod
    def eng():
        import pandas as pd
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        pd.options.display.float_format = '{:, .5f}'.format
        pd.set_option('precision', 7)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    @staticmethod
    def get_repr(data, limit=50):
        """A well-behaved repr function for all data types."""
        if type(data) is np.ndarray:
            string_ = f"shape = {data.shape}, dtype = {data.dtype}."
        elif type(data) is str:
            string_ = data
        elif type(data) is list:
            example = ("1st item type: " + str(type(data[0]))) if len(data) > 0 else " "
            string_ = f"length = {len(data)}. " + example
        else:
            string_ = repr(data)
        if len(string_) > limit: string_ = string_[:limit]
        return string_

    @staticmethod
    def outline(array, name="Array", imprint=True):
        str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"
        if imprint:
            print(str_)
        return str_

    @staticmethod
    def print_string_list(mylist, char_per_row=125, sep=" "):
        counter = 0
        index = 0
        while index < len(mylist):
            item = mylist[index]
            print(item, end=sep)
            counter += len(item)
            if counter <= char_per_row:
                pass
            else:
                counter = 0
                print("\n")
            index += 1


if __name__ == '__main__':
    pass
