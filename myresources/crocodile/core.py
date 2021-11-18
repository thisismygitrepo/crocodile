
"""
A collection of classes extending the functionality of Python's builtins.
email programmer@usa.com
"""

# Typing
# Path
import os
from pathlib import Path
import string
import random

# Numerical
import numpy as np
import pandas as pd
# Meta
import dill
import copy
from datetime import datetime
import datetime as dt  # useful for deltatime and timezones.


_ = dt


def get_time_stamp(fmt=None, name=None):
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
    sc = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks",
          "M": "months", "y": "years"}
    key, val = sc[past[-1]], eval(past[:-1])
    if key == "months":
        key = "days"
        val = val * 30
    elif key == "years":
        key = "weeks"
        val = val * 52
    return dt.timedelta(**{key: val})


def get_random_string(length=10, pool=None):
    if pool is None:
        pool = string.ascii_letters
    result_str = ''.join(random.choice(pool) for _ in range(length))
    return result_str


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
            # path = P.tmp(fn=P.random() + "-" + get_time_stamp()) + self.ext
            raise ValueError
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
        def wrapper(path=None, obj=None, **kwargs):
            if path is None:
                path = Path.home().joinpath("tmp_results").joinpath(get_time_stamp() + ext)
                # raise ValueError
            else:
                if not str(path).endswith(ext):
                    path = Path(str(path) + ext)
                else:
                    path = Path(path)

            path.parent.mkdir(exist_ok=True, parents=True)
            func(path, obj, **kwargs)
            print(f"File saved @ ", path.absolute().as_uri(), ". Directory: ", path.parent.absolute().as_uri())
            return path
        return wrapper
    return decorator


class Save:
    @staticmethod
    @save_decorator(".csv")
    def csv(path=None, obj=None):
        # obj.to_frame('dtypes').reset_index().to_csv(P(path).append(".dtypes").string)
        obj.to_frame('dtypes').reset_index().to_csv(path + ".dtypes")

    @staticmethod
    @save_decorator(".npy")
    def npy(path, obj, **kwargs):
        np.save(path, obj, **kwargs)

    @staticmethod
    @save_decorator(".mat")
    def mat(path=None, mdict=None, **kwargs):
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
    def json(path=None, obj=None, **kwargs):
        """This format is **compatible** with simple dictionaries that hold strings or numbers
         but nothing more than that.
        E.g. arrays or any other structure. An example of that is settings dictionary. It is useful because it can be
        inspected using any text editor."""
        import json

        with open(str(path), "w") as file:
            json.dump(obj, file, default=lambda x: x.__dict__, **kwargs)

    @staticmethod
    @save_decorator
    def yaml(path, obj, **kwargs):
        import yaml
        with open(str(path), "w") as file:
            yaml.dump(obj, file, **kwargs)

    @staticmethod
    @save_decorator(".pkl")
    def vanilla_pickle(path, obj, **kwargs):
        import pickle
        with open(str(path), 'wb') as file:
            pickle.dump(obj, file, **kwargs)

    @staticmethod
    @save_decorator(".pkl")
    def pickle(path=None, obj=None, recurse=False, **kwargs):
        """This is based on `dill` package. While very flexible, it comes at the cost of assuming so many packages are
        loaded up and it happens implicitly. It often fails at load time and requires same packages to be reloaded first
        . Compared to vanilla pickle, the former always raises an error when cannot pickle an object due to
        dependency. Dill however, stores all the required packages for any attribute object, but not the class itself,
        or the classes that it inherits (at least at with this version)."""
        with open(str(path), 'wb') as file:
            dill.dump(obj, file, recurse=recurse, **kwargs)

    @staticmethod
    def pickle_s(obj):
        binary = dill.dumps(obj)
        return binary


class Base(object):
    def __init__(self, *args, **kwargs):
        pass

    def __getstate__(self):
        """This method is used by Python internally when an instance of the class is pickled. (itself=True)
        Additionally, it is used by `save_pickle` to determine which attributes should be saved.
        Best practice here is to delete attributes that are not pickleable, as opposed to setting them to None.
        Setting them to None means that at load time, they will be set to None which is not required.
        Those attributes will be passed from user at construction time."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_saved(cls, path=None, *args, reader=None, r=False, modules=None, **kwargs):
        """Works in conjuction with save_pickle.
        The method thinks of class as a combination of data and functionality. Thus, to load up and instance
         of a class, this method, obviously, requires the class to be loadded up first then this method is used.

        If, at save time, the class itself was saved, then, the path to it is sufficient to load it up.
        It is the responsibility of the user to determine whether the path is pointing to a saved instance
        or saved attributes which require this method to load up the instance.
        A naming protocal for distinguishing is due.

        It is vital that __init__ method of class is well behaved.  That is, class instance can be initialized
        with no or only fake inputs (use default args to achieve this behaviour), so that a skeleton instance
        can be easily made then attributes are updated from the data loaded from disc. A good practice is to add
        a flag (e.g. from_saved) to init method to require the special behaviour indicated above when it is raised, e.g. do NOT
        create some expensive attribute is this flag is raised because it will be obtained later.
        """
        # ============================ step 1: load up the code (methods) (why not data first then class?)
        if args == () and kwargs == {}:  # user did not feed any init args.
            # Either because his/her design does not require them, or, they expect them to be loaded up.
            # uninit_inst = cls.__new__(object)  # avoid the init method because it demands arguments.
            # print("new method is used.")
            inst = cls(*args, **kwargs)
        else:  # some classes are written in a way that you must pass some args at init time.
            # print(args, kwargs)
            inst = cls(*args, **kwargs)
        # ============================= step 2: load up the data
        if path is not None:
            with open(str(path), 'rb') as file:
                data = dill.load(file)
        else:
            data = dict()
        # ============================= step 3: update / populate instance attributes with data.
        inst.__setstate__(dict(data))
        # ============================= step 4: check for saved attributes.
        if r:
            contents = [item.stem for item in Path(path).parent.glob("*.zip")]
            for key, _ in data.items():  # val is probably None (if init was written properly)
                if key in contents:
                    setattr(inst, key, Base.from_zipped_codata(path=Path(path).parent.joinpath(key + ".zip"),
                                                                     r=True, modules=modules, **kwargs))
        # =========================== Return a ready instance the way it was saved.
        return inst

    def save_npy(self, path=None, **kwargs):
        Save.npy(path, self.__dict__, **kwargs)
        return self

    def save_codata(self, path, r=False):
        """In contrast difference to save_pickle, this method saves code and data and produce a zip file.
        save_pikle only saves data, or the entire object (almost always not possible).
        To recover data from this method, treat the zip files with from_zipped_code_and_class.
        Internally, the method uses save_pickle and from_saved methods to handle pickled data."""
        path = Path(path)
        temp_path = Path().home().joinpath(f"tmp_results/zipping/{get_random_string()}")
        temp_path.mkdir(parents=True, exist_ok=True)
        self.save_code(temp_path.joinpath(f"source_code_{get_random_string()}.py"))
        self.save_pickle(path=temp_path.joinpath("class_data"), r=r)
        import shutil
        result_path = shutil.make_archive(base_name=path, format="zip",
                                          root_dir=str(temp_path), base_dir=".")
        result_path = Path(result_path)
        print(f"Class and data saved @ {result_path.as_uri()}")
        return result_path

    def save_code(self, file_path):
        import inspect
        file = Path(inspect.getmodule(self).__file__)
        class_name = self.__class__.__name__
        source_code = file.read_text()
        # injected_code = f"""\n\ndef get_instance(path=None, *args, **kwargs):
        # instance = {class_name}.from_saved(path=tb.P(path), *args, **kwargs)
        # return instance"""
        # source_code += injected_code
        file_path = Path(file_path)
        file_path.write_text(data=source_code)
        return file_path

    # @classmethod
    # def from_class_source_code_and_data(cls, source_code_path, *args, data_path=None, r=False, **kwargs):
    #     return cls.from_source_code_and_data(source_code_path, data_path, *args, class_name=cls.__name__,
    #                                          r=r, **kwargs)

    # @classmethod
    # def from_zipped_code_and_class(cls, path, *args, r=False, **kwargs):
    #     return cls.from_zipped_code_and_class_auto(path, *args, class_name=cls.__name__, r=r, **kwargs)

    @staticmethod
    def from_codata(source_code_path, data_path=None, class_name=None, r=False, *args, **kwargs):
        import sys
        source_code_path = Path(source_code_path)
        sys.path.insert(0, str(source_code_path.parent))
        import importlib
        sourcefile = importlib.import_module(source_code_path.stem)
        # importlib.invalidate_caches()
        # sys.path.remove(str(source_code_path.parent))  # otherwise, the first one will mask the subsequents
        # sys.modules.__delitem__("source_code")
        # removing the source_code means you can no longer save objects as their module is missing.
        # if class_name is None:
        #     instance = getattr(sourcefile, "get_instance")(data_path, *args, r=r, **kwargs)
        #     return instance
        return getattr(sourcefile, class_name).from_saved(data_path, *args, r=r, **kwargs)

    @staticmethod
    def from_zipped_codata(path, *args, class_name=None, r=False, modules=None, **kwargs):
        # TODO make this a class method.
        fname = Path(path).name.split(".zip")[1]
        temp_path = Path.home().joinpath(f"temp_results/unzipped/{fname}_{get_random_string()}")
        from zipfile import ZipFile
        with ZipFile(str(path), 'r') as zipObj:
            zipObj.extractall(temp_path)
        source_code_path = list(temp_path.glob("source_code*"))[0]
        data_path = list(temp_path.glob("class_data*"))[0]
        class_name = class_name or str(data_path).split(".")[1]
        if modules is None:  # load from source code
            return Base.from_codata(*args, source_code_path=source_code_path,
                                                       data_path=data_path,
                                                       class_name=class_name, r=r, **kwargs)
        else:
            return modulse[class_name].from_saved(data_path, *args, r=r, modules=modules, **kwargs)

    def save_pickle(self, path=None, itself=False, r=False, **kwargs):
        """works in conjunction with from_saved.
        :param path:
        :param itself: determiens whether to save the __dict__ only or the entire class instance (code + data).
        If __dict__ is only to be saved (assuming it is pure data rather than code),
        then the class itself is required later and the `from_saved` method should be used to reload the instance again.

        Alternatively, if the `itself` flag is raised, then, at load time, no reference to the class is needed as it is
        stored already. In a word, pickling requirements must be present in mind while writing __init__ method.

        A usecase for the former is when the source code is continously changed and still you want to reload an old
        version.

        Depending on complexity of the class written, it might not be possible to pickle the entire instance.

        A caveat arises when design of class favours composition over inheritence. Then, even with attempting to save
        the data itself rather than class, then it fails because __dict__ contains other classes (code) rather than data
        i.e. the first problem with (itself = True) catches up.

        Beware of the security risk involved in pickling objects that reference sensitive information like tokens and
        passwords. The best practice is to pass them again at load time.
        """

        ext = self.__class__.__name__
        path = Path(str(path) + "." + ext)
        if not itself:  # default, works in all cases.
            obj = self.__getstate__()
            if r:
                obj = obj.copy()  # do not mess with original __dict__
                for key, val in obj.items():
                    if Base in val.__class__.__mro__:  # a class instance rather than pure data
                        val.save_codata(path=path.parent.joinpath(key), r=True)
                        obj[key] = None  # this tough object is finished, the rest should be easy.
                    else:
                        pass  # leave this object as is.
        else:  # does not work for all classes with whacky behaviours, no gaurantee.
            obj = self
        Save.pickle(path, obj, **kwargs)
        return self

    def save_json(self, path=None, *args, **kwargs):
        """Use case: json is good for simple dicts, e.g. settings.
        Advantage: human-readable from file explorer."""
        _ = args
        Save.json(path, self.__dict__, **kwargs)
        return self

    def save_mat(self, path=None, *args, **kwargs):
        """for Matlab compatibility."""
        _ = args
        Save.mat(path, self.__dict__, **kwargs)
        return self

    def get_attributes(self, check_ownership=False, remove_base_attrs=True, return_objects=False):
        attrs = list(filter(lambda x: ('__' not in x) and not x.startswith("_"), dir(self)))
        _ = check_ownership
        if remove_base_attrs:
            pass
            # [attrs.remove(x) for x in Base().get_attributes()]
        # if exclude is not None:
        #     [attrs.remove(x) for x in exlcude]
        if return_objects:
            # attrs = attrs.apply(lambda x: getattr(self, x))
            attrs = [getattr(self, x) for x in attrs]
        return attrs

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
        Struct(self.__dict__).print(typeinfo=typeinfo)

    def viz(self, depth=3, obj=None, filter=None):
        import objgraph
        import tempfile
        filename = Path(tempfile.gettempdir()).joinpath("graph_viz_" + get_random_string() + ".png")
        objgraph.show_refs(self if obj is None else obj, max_depth=depth, filename=str(filename), filter=filter)
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

    def index(self, func, *args, **kwargs):
        """ A generalization of the `.index` method of `list`. It takes in a function rather than an
         item to find its index. Additionally, it returns full list of results, not just the first result.

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

        tqdm = 0
        if verbose or jobs:
            # Experimental.assert_package_installed("tqdm")  # removed for disentanglement
            from tqdm import tqdm
            # print(f"Applying {func} to elements in {self}")

        if other is None:
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(i, *args, **kwargs) for i in
                                                  tqdm(self.list, desc=desc)))
            else:
                iterator = self.list if not verbose else tqdm(self.list)
                return List([func(x, *args, **kwargs) for x in iterator])
        else:
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(x, y) for x, y in
                                                  tqdm(zip(self.list, other), desc=desc)))
            else:
                iterator = zip(self.list, other) if not verbose else \
                    tqdm(zip(self.list, other), desc=desc)
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


class Struct(Base, dict):  # inheriting from dict gives `get` method.
    """Use this class to keep bits and sundry items.
    Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience
    """

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

    def print(self, sep=None, yaml=False, typeinfo=True, logger=False):
        if bool(self) is False:
            print(f"Empty Struct.")
            return None  # break out of the function.
        if yaml:
            # removed for disentanglement
            # self.save_yaml(P.tmp(fn="__tmp.yaml"))
            # txt = P.tmp(fn="__tmp.yaml").read_text()
            # print(txt)
            return None
        if sep is None:
            sep = 5 + max(self.keys().apply(str).apply(len).list)
        repr_string = ""
        repr_string += "Structure, with following entries:\n"
        repr_string += "Key" + " " * sep + (("Item Type" + " " * sep) if typeinfo else "") + "Item Details\n"
        repr_string += "---" + " " * sep + (("---------" + " " * sep) if typeinfo else "") + "------------\n"
        for key in self.keys().list:
            key_str = str(key)
            type_str = str(type(self[key])).split("'")[1]
            val_str = DisplayData.get_repr(self[key]).replace("\n", " ")
            repr_string += key_str + " " * abs(sep - len(key_str)) + " " * len("Key")
            if typeinfo:
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
    def __init__(self, x):
        self.x = pd.DataFrame(x)

    @staticmethod
    def set_pandas_display(rows=1000, columns=1000, width=1000, colwidth=40):
        pd.set_option('display.max_colwidth', colwidth)
        pd.set_option('display.max_columns', columns)  # to avoid replacing them with ...
        pd.set_option('display.width', width)  # to avoid wrapping the table.
        pd.set_option('display.max_rows', rows)  # to avoid replacing rows with ...

    @staticmethod
    def eng():
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        pd.options.display.float_format = '{:, .5f}'.format
        pd.set_option('precision', 7)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    @staticmethod
    def get_repr(data):
        """A well-behaved repr function for all data types."""
        if type(data) is np.ndarray:
            string_ = f"shape = {data.shape}, dtype = {data.dtype}."
            return string_
        elif type(data) is str:
            return data
        elif type(data) is list:
            example = ("1st item type: " + str(type(data[0]))) if len(data) > 0 else " "
            return f"length = {len(data)}. " + example
        else:
            return repr(data)

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
