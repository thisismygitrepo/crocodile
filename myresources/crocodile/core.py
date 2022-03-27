"""
A collection of classes extending the functionality of Python's builtins.
email programmer@usa.com

Crocodile Philosophy:
Make Python even friendlier, by making available the common functionality for everyday use. Specifically,
path management, file management, and other common tasks.
A the risk of vandalizing the concept, to make Python more MATLAB-like, in that more libraries are loaded up at
start time_produced than basic arithmetic, but just enought to make it more useful for everyday errands.
Thus, the terseness of Crocodile makes Python REPL a proper shell

The focus is on ease of use, not efficiency.
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
    tmp_res = datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')
    return (name + '_' + tmp_res) if name is not None else tmp_res


def str2timedelta(past):
    """Converts a human readable string like '1m' or '1d' to a timedate object.
    In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    sc = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}
    key, val = sc[past[-1]], eval(past[:-1])
    if key == "months": key, val = "days", val * 30
    elif key == "years": key, val = "weeks", val * 52
    return dt.timedelta(**{key: val})


def randstr(length=10, lower=True, upper=True, digits=True, punctuation=False, safe=False):
    if safe:  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
        import secrets; return secrets.token_urlsafe(length)
    pool = (string.ascii_lowercase if lower else "") + (string.ascii_uppercase if upper else "")
    pool = pool + (string.digits if digits else "") + (string.punctuation if punctuation else "")
    return ''.join(random.choices(pool, k=length))


def validate_name(astring, replace='_'): import re; return re.sub(r'^(?=\d)|\W', replace, str(astring))


def install_n_import(package, name=None):
    """imports a package and installs it first if not."""
    try: return __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", name or package])
    return __import__(package)


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
            if not str(path).endswith(self.ext): raise ValueError
            else: raise ValueError

        # noinspection PyUnreachableCode
        path.parent.mkdir(exist_ok=True, parents=True)
        self.func(path, obj, **kwargs)
        print(f"File {obj} saved @ ", path.absolute().as_uri(), ". Directory: ", path.parent.absolute().as_uri())
        return path


def save_decorator(ext=""):
    """Apply default paths, add extension to path, print the saved file path"""

    def decorator(func):
        def wrapper(obj, path=None, verbose=True, add_suffix=True, **kwargs):
            if path is None:
                path = Path.home().joinpath("tmp_results").joinpath(randstr() + ext)
                print(f"tb.core: Warning: Path not passed to {func}. "
                      f"A default path has been chosen: {path.absolute().as_uri()}")
                # raise ValueError
            else:
                if add_suffix and not str(path).endswith(ext):
                    path = Path(str(path) + ext)
                    print(f"tb.core: Warning: suffix {ext} is added to path passed {path.as_uri()}")
                else: path = Path(path)
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
    def csv(obj, path=None): obj.to_frame('dtypes').reset_index().to_csv(path + ".dtypes")
    @staticmethod
    @save_decorator(".npy")
    def npy(obj, path, **kwargs): np.save(path, obj, **kwargs)
    @staticmethod
    def pickles(obj): return dill.dumps(obj)

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
            if value is None: mdict[key] = []
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
        loaded up and it happens implicitly. It often fails at load time_produced and requires same packages to be reloaded first
        . Compared to vanilla pickle, the former always raises an error when cannot pickle an object due to
        dependency. Dill however, stores all the required packages for any attribute object, but not the class itself,
        or the classes that it inherits (at least at with this version)."""
        with open(str(path), 'wb') as file: dill.dump(obj, file, recurse=r, **kwargs)


class Base(object):
    def __init__(self, *args, **kwargs): pass
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)
    def print(self, typeinfo=False): Struct(self.__dict__).print(dtype=typeinfo)

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

    def save(self, path=None, itself=True, r=False, include_code=False, add_suffix=True):
        """Pickles the object.
        :param path: destination file.
        :param itself:
        * `itself` is True: the object (self) will be pickled altogether.
            * Ups:
                * convenient.
                * Dill can, by itself import all the required classes to __main__ IF the object is restored
                 while directory is @ the same location object was created, thus,
            * Downs:
                * It requires (in case of composed objects) that every sub-object is well-behaved and has the appropriate
                  state methods implemented.
                * The libraries imported must be present at load time_produced.
        * `itself` is False: means to save __getstate__.
            * Ups:
                * very safe and unlikely to cause import errors at load time_produced.
                * User is responsible for loading up the class.
            * Downs: the class itself is required later and the `from_pickled_state` method should be used to reload the instance again.
            * __init__ method will be used again at reconstruction time_produced of the object before the attributes are monkey-patched.
            * It is very arduous to design __init__ method that is convenient (uses plethora of
              default arguments) and works at the same time_produced with no input at reconstruction time_produced.

        :param include_code: `save_code` will be called.
        :param r: recursive flag.
            * If attributes are not data, but rather objects, then, this flag should be set to True.
            * The recusive flag is particularly relevant when `itself` is False and __dict__ is composed of objects.
            * In pickling the object (itself=True), the recursive flag is the default.
        :param add_suffix: if True, the suffixes `.pkl` and `.py` will be added to the file name.

        * Tip: whether pickling the class or its data alone, always implement __getstate__ appropriately to
        avoid security risk involved in pickling objects that reference sensitive information like tokens and
        passwords.
        """
        path = path or Path.home().joinpath(f"tmp_results/tmpfiles/{randstr()}")
        if add_suffix:
            path = str(path)
            path += "" if self.__class__.__name__ in path else ("." + self.__class__.__name__)
            path += "" if (itself or ".dat" in path) else ".dat"
        path = Path(path)
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
                    else: pass  # leave this object as is.

        if include_code is True:
            temp_path = Path().home().joinpath(f"tmp_results/zipping/{randstr()}")
            temp_path.mkdir(parents=True, exist_ok=True)
            self.save_code(path=temp_path.joinpath(f"source_code_{randstr()}.py"))
            Save.pickle(path=temp_path.joinpath("class_data"), obj=obj, r=r, verbose=False, add_suffix=add_suffix)
            import shutil
            result_path = shutil.make_archive(base_name=str(path), format="zip", root_dir=str(temp_path), base_dir=".")
            result_path = Path(result_path)
            print(f"Code and data for the object ({repr(obj)}) saved @ "
                  f"`{result_path.as_uri()}`, Directory: `{result_path.parent.as_uri()}`")
        else:
            result_path = Save.pickle(obj=obj, path=path, r=r, verbose=False, add_suffix=add_suffix)
            rep = repr(obj)
            rep = rep if len(rep) < 50 else rep[:10] + "... "
            print(f"{'Data of' if itself else ''} Object ({rep}) saved @ "
                  f"`{result_path.absolute().as_uri()}`, Directory: `{result_path.parent.absolute().as_uri()}`")
        return result_path

    @classmethod
    def from_saved(cls, path, *args, r=False, scope=None, **kwargs):
        """Works in conjuction with save_pickle when `itself`=False, i.e. only state of object is pickled.
        # methodology: 1- Save state, 2- save code. 3- initialize from __init__, 4- populate __dict__

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
        # ============================= step 1: unpickle the data
        assert ".dat." in str(path), f"Are you sure the path {path} is pointing to pickeld state of {cls}?"
        data = dill.loads(Path(path).read_bytes())
        inst = cls(*args, **kwargs)  # ============================= step 2: initialize the class
        inst.__setstate__(dict(data))  # ===========step 3: update / populate instance attributes with data.
        # ============================= step 4: check for saved attributes.
        if r:  # add further attributes to `inst`, if any.
            contents = [item.stem for item in Path(path).parent.glob("*.zip")]
            for key, _ in data.items():  # val is probably None (if init was written properly)
                if key in contents:  # if key is not an attribute of the object, skip it.
                    setattr(inst, key, Base.from_zipped_code_state(path=Path(path).parent.joinpath(key + ".zip"), r=True, scope=scope, **kwargs))
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
                print(f"Warning: global scope has been contaminated by loaded scope {code_path} !!")
                scope.update(runpy.run_path(str(code_path)))  # Dill will no longer complain.
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
        
        if not methods: attrs = list(filter(lambda x: not inspect.ismethod(getattr(self, x)), attrs))
        if return_objects: attrs = [getattr(self, x) for x in attrs]
        return List(attrs)

    def __deepcopy__(self, *args, **kwargs):
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(copy.deepcopy(self.__dict__))
        return obj

    def __copy__(self, *args, **kwargs):
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
            if expected == 'func': return eval("lambda x: " + string_)
            elif expected == 'self':
                if "self" in string_: return eval(string_)
                else: return string_
        else: return string_

    def viz_composition_heirarchy(self, depth=3, obj=None, filt=None):
        import tempfile
        filename = Path(tempfile.gettempdir()).joinpath("graph_viz_" + randstr() + ".png")
        install_n_import("objgraph").show_refs([self] if obj is None else [obj], max_depth=depth, filename=str(filename), filter=filt)
        import sys
        if sys.platform == "win32": os.startfile(str(filename.absolute()))  # works for files and folders alike
        return filename


class List(Base, list):  # Inheriting from Base gives save method.
    """Use this class to keep items of the same type."""
    # =============================== Constructor Methods ====================
    def __init__(self, obj_list=None):
        super().__init__()
        self.list = list(obj_list) if obj_list is not None else []

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

    @classmethod
    def from_copies(cls, obj, count): return cls([copy.deepcopy(obj) for _ in range(count)])

    def save_items(self, directory, names=None, saver=None):
        if saver is None: saver = Save.pickle
        if names is None: names = range(len(self))
        for name, item in zip(names, self.list): saver(path=directory / name, obj=item)

    def __repr__(self): return f"List object with {len(self.list)} elements. First item of those is: \n" + f"{repr(self.list[0])}" if len(self.list) > 0 else f"An Empty List []"
    def __deepcopy__(self): return List([copy.deepcopy(i) for i in self.list])
    def __bool__(self): return bool(self.list)
    def __contains__(self, key): return key in self.list
    def __copy__(self): return List(self.list.copy())
    def __getstate__(self): return self.list
    def __setstate__(self, state): self.list = state
    def __len__(self): return len(self.list)
    def __iter__(self): return iter(self.list)
    @property
    def len(self): return self.list.__len__()
    # ================= call methods =====================================
    def method(self, name, *args, **kwargs): return List(getattr(i, name)(*args, **kwargs) for i in self.list)
    def attr(self, name): return List(getattr(i, name) for i in self.list)
    def __getattr__(self, name): return List(getattr(i, name) for i in self.list)  # fallback position when __getattribute__ mechanism fails.
    def __call__(self, *args, **kwargs): return List(i(*args, **kwargs) for i in self.list)
    # ======================== Access Methods ==========================================

    def __getitem__(self, key):
        if type(key) is list or type(key) is np.ndarray: return List(self[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
        elif type(key) is str: return List(item[key] for item in self.list)  # access keys like dictionaries.
        else: return self.list[key] if type(key) is not slice else List(self.list[key])  # must be an integer or slice: behaves similarly to Numpy A[1] vs A[1:2]

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
    def __setitem__(self, key, value): self.list[key] = value
    def sample(self, size=1, replace=False, p=None): return self[np.random.choice(len(self), size, replace=replace, p=p)]
    def index_items(self, idx): return List([item[idx] for item in self.list])
    def find_index(self, func) -> list: return List([idx for idx, x in enumerate(self.list) if self.evalstr(func, expected='func')(x)])
    def filter(self, func): return List([item for item in self.list if self.evalstr(func, expected='func')(item)])
    # ======================= Modify Methods ===============================
    def reduce(self, func): from functools import reduce; return reduce(func, self.list)
    def append(self, item): self.list.append(item); return self
    def __add__(self, other): return List(self.list + list(other))  # implement coersion
    def __radd__(self, other): return List(self.list + list(other))
    def __iadd__(self, other): self.list = self.list + list(other); return self  # inplace add.
    def sort(self, key=None, reverse=False): self.list.sort(key=key, reverse=reverse); return self
    def sorted(self, *args, **kwargs): return List(sorted(self.list, *args, **kwargs))
    def insert(self, __index: int, __object): self.list.insert(__index, __object); return self

    def remove(self, value=None, values=None):
        if value is not None: self.list.remove(value)
        if values is not None: [self.list.remove(value) for value in values]
        return self

    def apply(self, func, *args, other=None, jobs=None, depth=1, verbose=False, desc=None, **kwargs):
        """
        :param jobs:
        :param func: func has to be a function, possibly a lambda function. At any rate, it should return something.
        :param args:
        :param other: other list
        :param verbose:
        :param desc:
        :param depth: apply the function to inner Lists
        :param kwargs: a list of outputs each time_produced the function is called on elements of the list.
        :return:
        """
        if depth > 1: self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth-1, **kwargs))
        func = self.evalstr(func, expected='func')
        tqdm = install_n_import("tqdm").tqdm
        if other is None:
            iterator = self.list if not verbose else tqdm(self.list, desc=desc)
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(i, *args, **kwargs) for i in iterator))
            else: return List([func(x, *args, **kwargs) for x in iterator])
        else:
            iterator = zip(self.list, other) if not verbose else tqdm(zip(self.list, other), desc=desc)
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(x, y) for x, y in iterator))
            else: return List([func(x, y) for x, y in iterator])

    def modify(self, func: str, other=None):
        """Modifies objects rather than returning new list of objects
        :param func: a string that will be executed, assuming idx, x and y are given.
        :param other:
        :return:
        """
        if other is None:
            for idx, x in enumerate(self.list): _ = x, idx; exec(func)
        else:
            for idx, (x, y) in enumerate(zip(self.list, other)): _ = idx, x, y; exec(func)
        return self

    def print(self, nl=1, sep=False, style=repr):
        for idx, item in enumerate(self.list):
            print(f"{idx:2}- {style(item)}", end=' ')
            for _ in range(nl): print('', end='\n')
            if sep: print(sep * 100)

    def to_series(self): import pandas as pd; return pd.Series(self.list)
    def to_list(self): return self.list
    def to_numpy(self): return self.np
    @property
    def np(self): return np.array(self.list)
    def to_struct(self, key_val=None): return Struct.from_keys_values_pairs(self.apply(self.evalstr(key_val) if key_val else lambda x: (str(x), x)))

    def to_dataframe(self, names=None, minimal=False, obj_included=True):
        """
        :param names: path of each object.
        :param minimal: Return Dataframe structure without contents.
        :param obj_included: Include a colum for objects themselves.
        :return:
        """
        import pandas as pd
        columns = list(self.list[0].__dict__.keys())
        if obj_included or names: columns = ['object'] + columns
        df = pd.DataFrame(columns=columns)
        if minimal: return df
        # Populate the dataframe:
        for i, obj in enumerate(self.list):
            if obj_included or names:
                if names is None: name = [obj]
                else: name = [names[i]]
                df.loc[i] = name + list(self.list[i].__dict__.values())
            else: df.loc[i] = list(self.list[i].__dict__.values())
        return df


class Struct(Base, dict):
    """Use this class to keep bits and sundry items.
    Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience
    # inheriting from dict gives `get` method, should give `__contains__` but not working.
    # Inheriting from Base gives `save` method.
    """

    def __init__(self, dictionary=None, **kwargs):
        """
        :param dictionary: a dict, a Struct, None or an object with __dict__ attribute.
        """
        super(Struct, self).__init__()
        if type(dictionary) is Struct: dictionary = dictionary.dict
        if dictionary is None: final_dict = kwargs  # only kwargs were passed
        elif not kwargs:  # only dictionary was passed
            if type(dictionary) is dict: final_dict = dictionary
            elif dictionary.__class__.__name__ == "mappingproxy": final_dict = dict(dictionary)
            else: final_dict = dictionary.__dict__
        else:  # both were passed
            final_dict = dictionary if type(dictionary) is dict else dictionary.__dict__
            final_dict.update(kwargs)
        self.__dict__ = final_dict

    @staticmethod
    def recursive_struct(mydict):
        struct = Struct(mydict)
        for key, val in struct.items():
            if type(val) is dict: struct[key] = Struct.recursive_struct(val)
        return struct

    @staticmethod
    def recursive_dict(struct):
        mydict = struct.dict
        for key, val in mydict.items():
            if type(val) is Struct: mydict[key] = Struct.recursive_dict(val)
        return mydict

    def save_json(self, path=None): return Save.json(obj=self.__dict__, path=path)
    @classmethod
    def from_keys_values(cls, keys, values): return cls(dict(zip(keys, values)))
    @classmethod
    def from_keys_values_pairs(cls, my_list): return cls({k: v for k, v in my_list})
    @classmethod
    def from_names(cls, names, default_=None): return cls.from_keys_values(names, values=default_ or [None] * len(names))  # Mimick NamedTuple and defaultdict
    def spawn_from_values(self, values): return self.from_keys_values(self.keys(), self.evalstr(values, expected='self'))
    def spawn_from_keys(self, keys): return self.from_keys_values(self.evalstr(keys, expected="self"), self.values())

    def to_default(self, default=lambda: None):
        from collections import defaultdict
        tmp2 = defaultdict(default)
        tmp2.update(self.__dict__)
        self.__dict__ = tmp2
        return self

    # =========================== print ===========================
    @property
    def clean_view(self):

        class Temp:
            pass

        temp = Temp()
        temp.__dict__ = self.__dict__
        return temp

    def print(self, sep=None, yaml=False, dtype=True, return_str=False, limit=50, config=False, newline=True):
        if config:
            repr_str = Display.config(self.__dict__, newline=newline)
            if return_str: return repr_str
            else:
                print(repr_str)
                return self

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
            val_str = Display.get_repr(self[key], limit=limit).replace("\n", " ")
            repr_string += key_str + " " * abs(sep - len(key_str)) + " " * len("Key")
            if dtype:
                repr_string += type_str + " " * abs(sep - len(type_str)) + " " * len("Item Type")
            repr_string += val_str + "\n"
        if return_str: return repr_string
        else: print(repr_string)
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

    def __getattr__(self, item):  # this works better with the linter.
        try: return self.__dict__[item]
        except KeyError: raise AttributeError(f"Could not find the attribute `{item}` in this Struct object.")

    def __repr__(self): return "Struct: [" + "".join([str(key) + ", " for key in self.keys().to_list()]) + "]"
    def __getitem__(self, item): return self.__dict__[item]  # thus, gives both dot notation and string access to elements.
    def __setitem__(self, key, value): self.__dict__[key] = value
    def __bool__(self): return bool(self.__dict__)
    def __contains__(self, key): return key in self.__dict__
    def __len__(self): return len(self.keys())
    def __getstate__(self): return self.__dict__  # serialization
    def __setstate__(self, state): self.__dict__ = state
    def __iter__(self): return iter(self.dict.items())
    def __delitem__(self, key): del self.__dict__[key]
    @property
    def dict(self): return self.__dict__  # allows getting dictionary version without accessing private memebers explicitly.
    @dict.setter
    def dict(self, adict): self.__dict__ = adict
    def keys(self) -> List: return List(list(self.dict.keys()))
    def values(self) -> List: return List(list(self.dict.values()))
    def items(self) -> List: return List(self.dict.items())
    def get_values(self, keys) -> List: return List([self[key] for key in keys])
    def to_dataframe(self, *args, **kwargs): import pandas as pd; return pd.DataFrame(self.__dict__, *args, **kwargs)
    def apply_to_keys(self, key_val_func): return Struct({key_val_func(key, val): val for key, val in self.items()})
    def filter(self, key_val_func=None): return Struct({key: self[key] for key, val in self.items() if key_val_func(key, val)})
    def inverse(self): return Struct({v: k for k, v in self.dict.items()})
    def update(self, *args, **kwargs): self.__dict__.update(Struct(*args, **kwargs).__dict__); return self

    def delete(self, key=None, keys=None, criterion=None):
        if key is not None: del self.__dict__[key]
        if keys is not None:
            for key in keys: del self.__dict__[key]
        if criterion is not None:
            for key in self.keys().list:
                if criterion(self[key]): self.__dict__.__delitem__(key)
        return self

    def apply_to_values(self, key_val_func):
        for key, val in self.items(): self[key] = key_val_func(val)
        return self

    @staticmethod
    def concat_values(*dicts, method=None, lenient=True, collect_items=False, clone=True):
        if method is None: method = list.__add__
        if not lenient:
            keys = dicts[0].keys()
            for i in dicts[1:]: assert i.keys() == keys
        # else if lenient, take the union
        if clone: total_dict = copy.deepcopy(dicts[0])  # take first dict in the tuple
        else: total_dict = dicts[0]  # take first dict in the tuple
        if collect_items:
            for key, val in total_dict.item(): total_dict[key] = [val]

            def method(tmp1, tmp2):
                return tmp1 + [tmp2]

        if len(dicts) > 1:  # are there more dicts?
            for adict in dicts[1:]:
                for key in adict.keys():  # get everything from this dict
                    try:  # may be the key exists in the total dict already.
                        total_dict[key] = method(total_dict[key], adict[key])
                    except KeyError:  # key does not exist in total dict
                        if collect_items: total_dict[key] = [adict[key]]
                        else: total_dict[key] = adict[key]
        return Struct(total_dict)

    def plot(self, artist=None):
        if artist is None:
            # artist = Artist(figname='Structure Plot')  # removed for disentanglement
            import matplotlib.pyplot as plt
            fig, artist = plt.subplots()
        for key, val in self: artist.plot(val, label=key)
        try: artist.fig.legend()
        except AttributeError: pass
        return artist


class Display:
    @staticmethod
    def set_pandas_display(rows=1000, columns=1000, width=5000, colwidth=40):
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
        if type(data) is np.ndarray: string_ = f"shape = {data.shape}, dtype = {data.dtype}."
        elif type(data) is str: string_ = data
        elif type(data) is list:
            example = ("1st item type: " + str(type(data[0]))) if len(data) > 0 else " "
            string_ = f"length = {len(data)}. " + example
        else: string_ = repr(data)
        if len(string_) > limit: string_ = string_[:limit]
        return string_

    @staticmethod
    def outline(array, name="Array", printit=True):
        str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"
        if printit: print(str_)
        return str_

    @staticmethod
    def config(mydict, newline=True):
        rep = ""
        for key, val in mydict.items(): rep += f"{key} = {val}" + ("\n" if newline else ", ")
        return rep

    @staticmethod
    def print_string_list(mylist, char_per_row=125, sep=" "):
        counter, index = 0, 0
        while index < len(mylist):
            item = mylist[index]
            print(item, end=sep)
            counter += len(item)
            if not counter <= char_per_row: counter = 0; print("\n")
            index += 1


if __name__ == '__main__':
    pass
