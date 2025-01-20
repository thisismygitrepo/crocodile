
"""
Core
"""

from pathlib import Path
from typing import Optional, Union, Generic, TypeVar, Type, Literal, List as ListType, Any, Iterator, Callable, Iterable, Hashable, Protocol, ParamSpec, Concatenate, TypedDict

_ = Concatenate
__ = TypedDict


_Slice = TypeVar('_Slice', bound='Slicable')
class Slicable(Protocol):
    def __getitem__(self: _Slice, i: slice) -> _Slice: ...


T = TypeVar('T')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
PLike = Union[str, Path]
PS = ParamSpec('PS')


# ============================== Accessories ============================================
def validate_name(astring: str, replace: str = '_') -> str:
    import re
    return re.sub(r'[^-a-zA-Z0-9_.()]+', replace, str(astring))
def timestamp(fmt: Optional[str] = None, name: Optional[str] = None) -> str:
    import datetime
    return ((name + '_') if name is not None else '') + datetime.datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')  # isoformat is not compatible with file naming convention, fmt here is.
def str2timedelta(shift: str):  # Converts a human readable string like '1m' or '1d' to a timedate object. In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    import datetime
    key, val = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}[shift[-1]], float(shift[:-1])
    key, val = ("days", val * 30) if key == "months" else (("weeks", val * 52) if key == "years" else (key, val))
    return datetime.timedelta(**{key: val})
def install_n_import(library: str, package: Optional[str] = None, fromlist: Optional[list[str]] = None):  # sometimes package name is different from import, e.g. skimage.
    try: return __import__(library, fromlist=fromlist if fromlist is not None else ())
    except (ImportError, ModuleNotFoundError):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package or library])
        return __import__(library, fromlist=fromlist if fromlist is not None else ())
def randstr(length: int = 10, lower: bool = True, upper: bool = True, digits: bool = True, punctuation: bool = False, safe: bool = False, noun: bool = False) -> str:
    if safe:
        import secrets
        return secrets.token_urlsafe(length)  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
    if noun:
        import randomname
        return randomname.get_name()
    import string
    import random
    population = (string.ascii_lowercase if lower else "") + (string.ascii_uppercase if upper else "") + (string.digits if digits else "") + (string.punctuation if punctuation else "")
    return ''.join(random.choices(population, k=length))

def run_in_isolated_ve(packages: list[str], pyscript: str) -> str:
    ve_name = randstr()
    packages_space_separated = " ".join(packages)
    packages_space_separated = "pip setuptools " + packages_space_separated
    ve_creation_cmd = f"""
uv venv $HOME/venvs/tmp/{ve_name} --python 3.11
. $HOME/venvs/tmp/{ve_name}/bin/activate
uv pip install {packages_space_separated}"""
    import time
    t0 = time.time()
    import subprocess
    subprocess.run(ve_creation_cmd, shell=True, check=True, executable='/bin/bash')
    t1 = time.time()
    print(f"✅ Finished creating venv @ $HOME/venvs/tmp/{ve_name} in {t1 - t0:0.2f} seconds.")
    script_path = Path.home().joinpath("tmp_results/tmp_scripts/python").joinpath(ve_name + f"_script_{randstr()}.py")
    script_path.write_text(pyscript, encoding='utf-8')
    fire_script = f"source $HOME/venvs/tmp/{ve_name}/bin/activate; python {script_path}"
    subprocess.run(fire_script, shell=True, check=True, executable='/bin/bash')
    return fire_script


def save_decorator(ext: str = ""):  # apply default paths, add extension to path, print the saved file path
    def decorator(func: Callable[..., Any]):
        def wrapper(obj: Any, path: Union[str, Path, None] = None, verbose: bool = True, add_suffix: bool = False, desc: str = "", class_name: str = "",
                    **kwargs: Any):
            if path is None:
                path = Path.home().joinpath("tmp_results/tmp_files").joinpath(randstr(noun=True))
                _ = print(f"tb.core: Warning: Path not passed to {func}. A default path has been chosen: {Path(path).absolute().as_uri()}") if verbose else None
            if add_suffix:
                _ = [(print(f"tb.core: Warning: suffix `{a_suffix}` is added to path passed {path}") if verbose else None) for a_suffix in [ext, class_name] if a_suffix not in str(path)]
                path = str(path).replace(ext, "").replace(class_name, "") + class_name + ext
                path = Path(path).expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
            else: path = Path(path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            func(path=path, obj=obj, **kwargs)
            if verbose:
                try: print(f"💽 SAVED {desc or path.name} {obj.__class__.__name__}: {Display.f(repr(obj), justify=0, limit=50)}  @ `{path.absolute().as_uri()}`. Size = {path.stat().st_size / 1024**2:0.2f} MB")  # |  Directory: `{path.parent.absolute().as_uri()}`
                except UnicodeEncodeError as err: print(f"crocodile.core: Warning: UnicodeEncodeError: {err}")
            return path
        return wrapper
    return decorator


class Save:
    @staticmethod
    @save_decorator(".json")
    def json(obj: Any, path: PLike, indent: Union[str, int, None] = 4, encoding: str = 'utf-8', **kwargs: Any):
        import json as jsonlib
        return Path(path).write_text(jsonlib.dumps(obj, indent=indent, default=lambda x: x.__dict__, **kwargs), encoding=encoding)
    @staticmethod
    @save_decorator(".yml")
    def yaml(obj: dict[Any, Any], path: PLike, **kwargs: Any):
        import yaml  # type: ignore
        with open(Path(path), 'w', encoding="utf-8") as file:
            yaml.dump(obj, file, **kwargs)
    @staticmethod
    @save_decorator(".toml")
    def toml(obj: dict[Any, Any], path: PLike, encoding: str = 'utf-8'): return Path(path).write_text(install_n_import("toml").dumps(obj), encoding=encoding)
    @staticmethod
    @save_decorator(".ini")
    def ini(obj: dict[Any, Any], path: PLike, **kwargs: Any):
        # conf = install_n_import("configparser").ConfigParser()
        import configparser
        conf = configparser.ConfigParser()
        conf.read_dict(obj)
        with open(path, 'w', encoding="utf-8") as configfile: conf.write(configfile, **kwargs)
    # @staticmethod
    # @save_decorator(".csv")
    # def csv(obj: Any, path: PLike):
    #     return obj.to_frame('dtypes').reset_index().to_csv(str(path) + ".dtypes")
    @staticmethod
    @save_decorator(".npy")
    def npy(obj: Any, path: PLike, **kwargs: Any):
        import numpy as np
        return np.save(path, obj, **kwargs)
    # @save_decorator(".mat")
    # def mat(mdict, path=None, **kwargs): _ = [mdict.__setitem(key, []) for key, value in mdict.items() if value is None]; from scipy.io import savemat; savemat(str(path), mdict, **kwargs)  # Avoid using mat as it lacks perfect restoration: * `None` type is not accepted. Scalars are conveteed to [1 x 1] arrays.
    @staticmethod
    @save_decorator(".pkl")
    def pickle(obj: Any, path: PLike, **kwargs: Any):
        import pickle
        data = pickle.dumps(obj=obj, **kwargs)
        return Path(path).write_bytes(data=data)
    @staticmethod
    @save_decorator(".pkl")
    def dill(obj: Any, path: PLike, **kwargs: Any):
        import dill
        data = dill.dumps(obj=obj, **kwargs)
        return Path(path).write_bytes(data=data)
    # @staticmethod
    # def h5(obj: Any, path: PLike, **kwargs: Any):
    #     import h5py
    #     with h5py


# ====================================== Object Management ====================================
class Base(object):
    def __init__(self, *args: Any, **kwargs: Any): _ = args, kwargs
    def __getstate__(self) -> dict[str, Any]: return self.__dict__.copy()
    def __setstate__(self, state: dict[str, Any]): self.__dict__.update(state)
    def __deepcopy__(self, *args: Any, **kwargs: Any):
        obj = self.__class__(*args, **kwargs)
        import copy
        obj.__dict__.update(copy.deepcopy(self.__dict__))
        return obj
    def __copy__(self, *args: Any, **kwargs: Any):
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(self.__dict__.copy())
        return obj
    # def eval(self, string_, func=False, other=False): return string_ if type(string_) is not str else eval((("lambda x, y: " if other else "lambda x:") if not str(string_).startswith("lambda") and func else "") + string_ + (self if False else ''))
    # def exec(self, expr: str) -> 'Base': exec(expr); return self  # exec returns None.
    def save(self, path: Union[str, Path, None] = None, add_suffix: bool = True, save_code: bool = False, verbose: bool = True, data_only: bool = True, desc: str = ""):  # + (".dat" if data_only else "")
        obj = self.__getstate__() if data_only else self
        saved_file = Save.pickle(obj=obj, path=path, verbose=verbose, add_suffix=add_suffix, class_name="." + self.__class__.__name__, desc=desc or (f"Data of {self.__class__}" if data_only else desc))
        if save_code: self.save_code(path=saved_file.parent.joinpath(saved_file.name + "_saved_code.py"))
        return self
    @classmethod
    def from_saved_data(cls, path: PLike, *args: Any, **kwargs: Any):
        obj = cls(*args, **kwargs)
        import dill
        obj.__setstate__(dict(dill.loads(Path(path).read_bytes())))
        return obj
    def save_code(self, path: Union[str, Path]):
        import inspect
        module = inspect.getmodule(self)
        if module is not None and hasattr(module, "__file__"):
            file = Path(module.__file__)  # type: ignore
        else: raise FileNotFoundError("Attempted to save code from a script running in interactive session! module should be imported instead.")
        _ = Path(path).expanduser().write_text(encoding='utf-8', data=file.read_text(encoding='utf-8'))
        return Path(path) if type(path) is str else path  # path could be P, better than Path
    def get_attributes(self, remove_base_attrs: bool = True, return_objects: bool = False, fields: bool = True, methods: bool = True):
        import inspect
        remove_vals = Base().get_attributes(remove_base_attrs=False) if remove_base_attrs else []
        attrs: List[Any] = List(dir(self)).filter(lambda x: '__' not in x and not x.startswith('_')).remove(values=remove_vals)
        attrs = attrs.filter(lambda x: (inspect.ismethod(getattr(self, x)) if not fields else True) and ((not inspect.ismethod(getattr(self, x))) if not methods else True))  # logic (questionable): anything that is not a method is a field
        return List([getattr(self, x) for x in attrs]) if return_objects else List(attrs)
    def print(self, dtype: bool = False, attrs: bool = False, **kwargs: Any): return Struct(self.__dict__).update(attrs=self.get_attributes() if attrs else None).print(dtype=dtype, **kwargs)
    @staticmethod
    def get_state(obj: Any, repr_func: Callable[[Any], dict[str, Any]] = lambda x: x, exclude: Optional[list[str]] = None) -> dict[str, Any]:
        if not any([hasattr(obj, "__getstate__"), hasattr(obj, "__dict__")]): return repr_func(obj)
        return (tmp if type(tmp := obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__) is not dict else Struct(tmp).filter(lambda k, v: k not in (exclude or [])).apply2values(lambda k, v: Base.get_state(v, exclude=exclude, repr_func=repr_func)).__dict__)
    @staticmethod
    def viz_composition_heirarchy(obj: Any, depth: int = 3, filt: Optional[Callable[[Any], bool]] = None):
        import tempfile
        filename = Path(tempfile.gettempdir()).joinpath("graph_viz_" + randstr(noun=True) + ".png")
        install_n_import("objgraph").show_refs([obj], max_depth=depth, filename=str(filename), filter=filt)
        return filename


class List(Generic[T]):  # Inheriting from Base gives save method.  # Use this class to keep items of the same type."""
    def __init__(self, obj_list: Union[ListType[T], None, Iterator[T], Iterable[T]] = None) -> None:
        super().__init__()
        self.list = list(obj_list) if obj_list is not None else []
    def __repr__(self): return f"List [{len(self.list)} elements]. First Item: " + f"{Display.get_repr(self.list[0], justify=0, limit=100)}" if len(self.list) > 0 else "An Empty List []"
    def print(self, sep: str = '\n', styler: Callable[[Any], str] = repr, return_str: bool = False, **kwargs: dict[str, Any]):
        res = sep.join([f"{idx:2}- {styler(item)}" for idx, item in enumerate(self.list)])
        _ = print(res) if not return_str else None; _ = kwargs
        return res if return_str else None
    def __deepcopy__(self, arg: Any) -> "List[T]":
        _ = arg
        import copy
        return List([copy.deepcopy(i) for i in self.list])
    def __bool__(self) -> bool: return bool(self.list)
    def __contains__(self, key: str) -> bool: return key in self.list
    def __copy__(self) -> 'List[T]': return List(self.list.copy())
    def __getstate__(self) -> list[T]: return self.list
    def __setstate__(self, state: list[T]): self.list = state
    def __len__(self) -> int: return len(self.list)
    def __iter__(self) -> Iterator[T]: return iter(self.list)
    def __array__(self):
        import numpy as np
        return np.array(self.list)  # compatibility with numpy
    # def __next__(self) -> T: return next(self.list)
    @property
    def len(self) -> int: return len(self.list)
    # ================= call methods =====================================
    def __getattr__(self, name: str) -> 'List[T]': return List(getattr(i, name) for i in self.list)  # fallback position when __getattribute__ mechanism fails.
    def __call__(self, *args: Any, **kwargs: Any) -> 'List[Any]':
        items = self.list
        return List([ii.__call__(*args, **kwargs) for ii in items])  # type: ignore
    # ======================== Access Methods ==========================================
    def __setitem__(self, key: int, value: T) -> None: self.list[key] = value
    def sample(self, size: int = 1, replace: bool = False, p: Optional[list[float]] = None) -> 'List[T]':
        import numpy as np
        tmp = np.random.choice(len(self), size, replace=replace, p=p)
        return List([self.list[item] for item in tmp.tolist()])
    def split(self, every: int = 1, to: Optional[int] = None) -> 'List[List[T]]':
        import math
        every = every if to is None else math.ceil(len(self) / to)
        res: list[List[T]] = []
        for ix in range(0, len(self), every):
            if ix + every < len(self):
                tmp = self.list[ix:ix + every]
            else:
                tmp = self.list[ix:len(self)]
            res.append(List(tmp))
        return List(res)
    def filter(self, func: Callable[[T], bool], which: Callable[[int, T], Union[T, T2]] = lambda _idx, _x: _x) -> 'List[Union[T2, T]]':
        return List([which(idx, x) for idx, x in enumerate(self.list) if func(x)])
    # ======================= Modify Methods ===============================
    def reduce(self, func: Callable[[T, T], T], default: Optional[T] = None) -> 'List[T]':
        from functools import reduce
        if default is None:
            tmp = reduce(func, self.list)
            return List(tmp)  # type: ignore
        res = reduce(func, self.list, default)
        return List(res)  # type: ignore
    def append(self, item: T) -> 'List[T]': self.list.append(item); return self
    def insert(self, __index: int, __object: T): self.list.insert(__index, __object); return self
    def __add__(self, other: 'List[T]') -> 'List[T]': return List(self.list + list(other))  # implement coersion
    def __radd__(self, other: 'List[T]') -> 'List[T]': return List(list(other) + self.list)
    def __iadd__(self, other: 'List[T]') -> 'List[T]': self.list = self.list + list(other); return self  # inplace add.
    def sort(self, key: Callable[[T], float], reverse: bool = False) -> 'List[T]': self.list.sort(key=key, reverse=reverse); return self
    def sorted(self, *args: list[Any], **kwargs: Any) -> 'List[T]': return List(sorted(self.list, *args, **kwargs))
    # def modify(self, expr: str, other: Optional['List[T]'] = None) -> 'List[T]': _ = [exec(expr) for idx, x in enumerate(self.list)] if other is None else [exec(expr) for idx, (x, y) in enumerate(zip(self.list, other))]; return self
    def remove(self, value: Optional[T] = None, values: Optional[list[T]] = None, strict: bool = True) -> 'List[T]':
        for a_val in ((values or []) + ([value] if value else [])):
            if strict or value in self.list: self.list.remove(a_val)
        return self
    def to_series(self):
        import pandas as pd
        return pd.Series(self.list)
    def to_list(self) -> list[T]: return self.list
    def to_numpy(self, **kwargs: Any) -> 'Any': import numpy as np; return np.array(self.list, **kwargs)
    def to_struct(self, key_val: Optional[Callable[[T], tuple[Any, Any]]] = None) -> 'Struct':
        return Struct.from_keys_values_pairs(self.apply(func=key_val if key_val else lambda x: (str(x), x)).list)
    # def index(self, val: int) -> int: return self.list.index(val)
    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> 'List[T]': return List(self.list[start:stop:step])
    def __getitem__(self, key: Union[int, list[int], '_Slice']) -> Union[T, 'List[T]']:
        if isinstance(key, (list, Iterable, Iterator)): return List(self.list[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
        # elif isinstance(key, str): return List(item[key] for item in self.list)  # access keys like dictionaries.
        elif isinstance(key, int): return self.list[key]
        # assert isinstance(key, slice)
        return List(self.list[key])  # type: ignore # noqa: call-overload  # slices
    def apply(self, func: Union[Callable[[T], T3], Callable[[T, T2], T3]], *args: Any, other: Optional['List[T]'] = None, filt: Callable[[T], bool] = lambda _x: True,
              jobs: Optional[int] = None, prefer: Optional[Literal['processes', 'threads']] = None, verbose: bool = False, desc: Optional[str] = None, **kwargs: Any,
              ) -> 'Union[List[Union[T2, T3]], List[T3]]':
        # if depth > 1: self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth - 1, **kwargs))
        from tqdm import tqdm
        iterator: Iterable[Any]
        if other is None:
            if not verbose: iterator = self.list
            else: iterator = tqdm(self.list, desc=desc)
        else:
            if not verbose:
                iterator = zip(self.list, other)
            else:
                iterator =  tqdm(zip(self.list, other), desc=desc)
        if jobs is None or jobs ==1:
            if other is None:
                return List([func(x, *args, **kwargs) for x in iterator if filt(x)])
            return List([func(x, y) for x, y in iterator])  # type: ignore
        from joblib import Parallel, delayed
        if other is None: return List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, *args, **kwargs) for x in iterator))  # type: ignore
        return List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, y) for x, y in iterator))  # type: ignore

    def to_dataframe(self, names: Optional[list[str]] = None, minimal: bool = False, obj_included: bool = True):
        import pandas as pd
        df = pd.DataFrame(columns=(['object'] if obj_included or names else []) + list(self.list[0].__dict__.keys()))
        if minimal: return df
        for i, obj in enumerate(self.list):  # Populate the dataframe:
            if obj_included or names:
                tmp: list[Any] = list(self.list[i].__dict__.values())
                data: list[Any] = [obj] if names is None else [names[i]]
                df.iloc[i] = pd.Series(data + tmp)
            else: df.iloc[i] = pd.Series(self.list[i].__dict__.values())  # type: ignore
        return df


class Struct(Base):  # inheriting from dict gives `get` method, should give `__contains__` but not working. # Inheriting from Base gives `save` method.
    """Use this class to keep bits and sundry items. Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience"""
    def __init__(self, dictionary: Union[dict[Any, Any], Type[object], None] = None, **kwargs: Any):
        if dictionary is None or isinstance(dictionary, dict): final_dict: dict[str, Any] = {} if dictionary is None else dictionary
        else:
            final_dict = (dict(dictionary) if dictionary.__class__.__name__ == "mappingproxy" else dictionary.__dict__)  # type: ignore
        final_dict.update(kwargs)  # type ignore
        super(Struct, self).__init__()
        self.__dict__ = final_dict  # type: ignore
    # @staticmethod
    # def recursive_struct(mydict: dict[Any, Any]) -> 'Struct': struct = Struct(mydict); [struct.__setitem__(key, Struct.recursive_struct(val) if type(val) is dict else val) for key, val in struct.items()]; return struct
    # @staticmethod
    # def recursive_dict(struct) -> 'Struct': _ = [struct.__dict__.__setitem__(key, Struct.recursive_dict(val) if type(val) is Struct else val) for key, val in struct.__dict__.items()]; return struct.__dict__
    # def save_json(self, path: Optional[PLike] = None, indent: Optional[str] = None): return Save.json(obj=self.__dict__, path=path, indent=indent)
    @staticmethod
    def from_keys_values(k: Iterable[str], v: Iterable[Any]) -> 'Struct': return Struct(dict(zip(k, v)))
    @staticmethod
    def from_keys_values_pairs(my_list: list[tuple[Any, Any]]) -> "Struct": return Struct({k: v for k, v in my_list})
    @staticmethod
    def from_names(names: list[str], default_: Optional[Any] = None) -> 'Struct': return Struct.from_keys_values(k=names, v=default_ or [None] * len(names))  # Mimick NamedTuple and defaultdict
    def spawn_from_values(self, values: Union[list[Any], List[Any]]) -> 'Struct': return self.from_keys_values(list(self.keys()), values)
    def spawn_from_keys(self, keys: Union[list[str], List[str]]) -> 'Struct': return self.from_keys_values(keys, list(self.values()))
    def to_default(self, default: Optional[Callable[[], Any]] = lambda: None):
        import collections
        tmp2: dict = collections.defaultdict(default)  # type: ignore
        tmp2.update(self.__dict__)
        self.__dict__ = tmp2
        return self
    def __str__(self, sep: str = "\n"): return Display.config(self.__dict__, sep=sep)
    def __getattr__(self, item: str) -> 'Struct':
        try: return self.__dict__[item]
        except KeyError as ke: raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from ke  # this works better with the linter. replacing Key error with Attribute error makes class work nicely with hasattr() by returning False.
    # clean_view = property(lambda self: type("TempClass", (object,), self.__dict__))
    def __repr__(self, limit: int = 150): return "Struct: " + Display.get_repr(self.keys().list.__repr__(), limit=limit, justify=0)
    def __getitem__(self, item: str): return self.__dict__[item]  # thus, gives both dot notation and string access to elements.
    def __setitem__(self, key: str, value: Any): self.__dict__[key] = value
    def __bool__(self): return bool(self.__dict__)
    def __contains__(self, key: Hashable): return key in self.__dict__
    def __len__(self): return len(self.keys())
    def __getstate__(self): return self.__dict__  # serialization
    def __setstate__(self, state: dict[Any, Any]): self.__dict__ = state
    def __iter__(self): return iter(self.__dict__.items())
    def __delitem__(self, key: str): del self.__dict__[key]
    def copy(self) -> 'Struct': return Struct(self.__dict__.copy())
    def to_dataframe(self, *args: Any, **kwargs: Any):
        import pandas as pd
        return pd.DataFrame(self.__dict__, *args, **kwargs)
    def keys(self, verbose: bool = False) -> 'List[Any]':
        from tqdm import tqdm
        return List(list(self.__dict__.keys()) if not verbose else tqdm(self.__dict__.keys()))
    def values(self, verbose: bool = False) -> 'List[Any]':
        from tqdm import tqdm
        return List(list(self.__dict__.values()) if not verbose else tqdm(self.__dict__.values()))
    def items(self, verbose: bool = False, desc: str = "") -> 'List[Any]':
        from tqdm import tqdm
        return List(self.__dict__.items() if not verbose else tqdm(self.__dict__.items(), desc=desc))
    def get(self, key: Optional[str] = None, default: Optional[Any] = None, strict: bool = False, keys: Union[None, list[str]] = None) -> 'Union[Any, List[Any]]':
        if keys is not None: return List([self.__dict__.get(key, default) if not strict else self[key] for key in keys])
        if key is not None: return (self.__dict__.get(key, default) if not strict else self[key])
        else: raise ValueError("Either key or keys should be passed.")
    def apply2keys(self, kv_func: Callable[[Any, Any], Any], verbose: bool = False, desc: str = "") -> 'Struct': return Struct({kv_func(key, val): val for key, val in self.items(verbose=verbose, desc=desc)})
    def apply2values(self, kv_func: Callable[[Any, Any], Any], verbose: bool = False, desc: str = "") -> 'Struct':
        _ = [self.__setitem__(key, kv_func(key, val)) for key, val in self.items(verbose=verbose, desc=desc)]
        return self
    def apply(self, kv_func: Callable[[Any, Any], Any]) -> 'List[Any]': return self.items().apply(lambda item: kv_func(item[0], item[1]))
    def filter(self, kv_func: Callable[[Any, Any], Any]) -> 'Struct': return Struct({key: self[key] for key, val in self.items() if kv_func(key, val)})
    def inverse(self) -> 'Struct': return Struct({v: k for k, v in self.__dict__.items()})
    def update(self, *args: Any, **kwargs: Any) -> 'Struct': self.__dict__.update(Struct(*args, **kwargs).__dict__); return self
    def delete(self, key: Optional[str] = None, keys: Optional[list[str]] = None, kv_func: Optional[Callable[[Any, Any], Any]] = None) -> 'Struct':
        for key in ([key] if key else [] + (keys if keys is not None else [])): self.__dict__.__delitem__(key)
        if kv_func is not None:
            for k, v in self.items():
                if kv_func(k, v): self.__dict__.__delitem__(k)
        return self
    def _pandas_repr(self, justify: int, return_str: bool = False, limit: int = 30):
        import pandas as pd
        import numpy as np
        col2: List[Any] = self.values().apply(lambda x: str(type(x)).split("'")[1])
        col3: List[Any] = self.values().apply(lambda x: Display.get_repr(x, justify=justify, limit=limit).replace("\n", " "))
        array = np.array([self.keys(), col2, col3]).T
        res: pd.DataFrame = pd.DataFrame(array, columns=["key", "dtype", "details"])
        return res if not return_str else str(res)
    def print(self, dtype: bool = True, return_str: bool = False, justify: int = 30, as_config: bool = False, as_yaml: bool = False,  # type: ignore # pylint: disable=W0237
              limit: int = 50, title: str = "", attrs: bool = False, **kwargs: Any) -> Union[str, None]:  # type: ignore
        _ = attrs
        import pandas as pd
        if as_config and not return_str:
            from rich import inspect
            inspect(self, value=False, title=title, docs=False, dunder=False, sort=False)
            return None
        if not bool(self):
            if return_str: return "Empty Struct."
            else: print("Empty Struct."); return None
        else:
            if as_yaml or as_config:
                import yaml
                tmp: str = yaml.dump(self.__dict__) if as_yaml else Display.config(self.__dict__, justify=justify, **kwargs)
                if return_str: return tmp
                else:
                    from rich.syntax import Syntax
                    from rich.console import Console
                    console = Console()
                    console.print(Syntax(tmp, "yaml"))
                    return None
            else:
                tmp2 = self._pandas_repr(justify=justify, return_str=False, limit=limit)
                if isinstance(tmp2, pd.DataFrame):
                    res = tmp2.drop(columns=[] if dtype else ["dtype"])
                else: raise TypeError(f"Unexpected type {type(tmp2)}")
                if not return_str:
                    # import tabulate
                    import rich
                    rich.print(res.to_markdown())
                    # else: print(res)
                    return None
                return str(res)
    @staticmethod
    def concat_values(*dicts: dict[Any, Any], orient: Literal["dict", "list", "series", "split", "tight", "index"] = 'list') -> 'Struct':
        import pandas as pd
        tmp = [Struct(x).to_dataframe() for x in dicts]
        res = pd.concat(tmp).to_dict(orient=orient)
        return Struct(res)  # type: ignore
    def plot_plt(self, title: str = '', xlabel: str = '', ylabel: str = '', **kwargs: Any):
        from crocodile.matplotlib_management import LineArtist
        artist = LineArtist(figname='Structure Plot', **kwargs)
        artist.plot_dict(self.__dict__, title=title, xlabel=xlabel, ylabel=ylabel)
        return artist
    def plot_plotly(self):
        import plotly.express as px
        fig = px.line(self.__dict__)
        fig.show()
        return fig


class Display:
    @staticmethod
    def set_pandas_display(rows: int = 1000, columns: int = 1000, width: int = 5000, colwidth: int = 40) -> None:
        import pandas as pd
        pd.set_option('display.max_colwidth', colwidth)
        pd.set_option('display.max_columns', columns)
        pd.set_option('display.width', width)
        pd.set_option('display.max_rows', rows)
    @staticmethod
    def set_pandas_auto_width():
        import pandas as pd
        pd.set_option('width', 0)  # this way, pandas is told to detect window length and act appropriately.  For fixed width host windows, this is recommended to avoid chaos due to line-wrapping.
    @staticmethod
    def set_numpy_display(precision: int = 3, linewidth: int = 250, suppress: bool = True, floatmode: Literal['fixed', 'unique', 'maxprec', 'maxprec_equal'] = 'fixed', **kwargs: Any) -> None:
        import numpy as np
        np.set_printoptions(precision=precision, suppress=suppress, linewidth=linewidth, floatmode=floatmode, formatter={'float_kind':'{:0.2f}'.format}, **kwargs)
    @staticmethod
    def config(mydict: dict[Any, Any], sep: str = "\n", justify: int = 15, quotes: bool = False):
        return sep.join([f"{key:>{justify}} = {repr(val) if quotes else val}" for key, val in mydict.items()])
    @staticmethod
    def f(str_: str, limit: int = 10000000000, justify: int = 50, direc: str = "<") -> str:
        return f"{(str_[:limit - 4] + '... ' if len(str_) > limit else str_):{direc}{justify}}"
    @staticmethod
    def eng():
        import pandas as pd
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        # pd.options.float_format = '{:, .5f}'.format
        pd.set_option('precision', 7)  # pd.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    @staticmethod
    def outline(array: 'Any', name: str = "Array", printit: bool = True):
        str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"
        if printit: print(str_)
        return str_
    @staticmethod
    def get_repr(data: Any, justify: int = 15, limit: int = 10000, direc: str = "<") -> str:
        if (dtype := data.__class__.__name__) in {'list', 'str'}: str_ = data if dtype == 'str' else f"list. length = {len(data)}. " + ("1st item type: " + str(type(data[0])).split("'")[1]) if len(data) > 0 else " "
        elif dtype in {"DataFrame", "Series"}: str_ = f"Pandas DF: shape = {data.shape}, dtype = {data.dtypes}." if dtype == 'DataFrame' else f"Pandas Series: Length = {len(data)}, Keys = {Display.get_repr(data.keys().to_list())}."
        else: str_ = f"shape = {data.shape}, dtype = {data.dtype}." if dtype == 'ndarray' else repr(data)
        return Display.f(str_.replace("\n", ", "), justify=justify, limit=limit, direc=direc)
    @staticmethod
    def print_string_list(mylist: list[Any], char_per_row: int = 125, sep: str = " ", style: Callable[[Any], str] = str, _counter: int = 0):
        for item in mylist:
            _ = print("") if (_counter + len(style(item))) // char_per_row > 0 else print(style(item), end=sep)
            _counter = len(style(item)) if (_counter + len(style(item))) // char_per_row > 0 else _counter + len(style(item))


if __name__ == '__main__':
    pass
