
"""
Core
"""

from pathlib import Path
from typing import Optional, Union, Generic, TypeVar, Type, List as ListType, Any, Iterator, Callable, Iterable, Hashable, Protocol
import datetime

_Slice = TypeVar('_Slice', bound='Slicable')
class Slicable(Protocol):
    def __getitem__(self: _Slice, i: slice) -> _Slice: ...


T = TypeVar('T')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
PLike = Union[str, Path]


# ============================== Accessories ============================================
def validate_name(astring: str, replace: str = '_') -> str: return __import__("re").sub(r'[^-a-zA-Z0-9_.()]+', replace, str(astring))
def timestamp(fmt: Optional[str] = None, name: Optional[str] = None) -> str: return ((name + '_') if name is not None else '') + __import__("datetime").datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')  # isoformat is not compatible with file naming convention, fmt here is.
def str2timedelta(shift: str) -> datetime.timedelta:  # Converts a human readable string like '1m' or '1d' to a timedate object. In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    key, val = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}[shift[-1]], float(shift[:-1])
    key, val = ("days", val * 30) if key == "months" else (("weeks", val * 52) if key == "years" else (key, val)); return __import__("datetime").timedelta(**{key: val})
def install_n_import(library: str, package: Optional[str] = None, fromlist: Optional[list[str]] = None):  # sometimes package name is different from import, e.g. skimage.
    try: return __import__(library, fromlist=fromlist if fromlist is not None else ())
    except (ImportError, ModuleNotFoundError): __import__("subprocess").check_call([__import__("sys").executable, "-m", "pip", "install", package or library]); return __import__(library, fromlist=fromlist if fromlist is not None else ())
def randstr(length: int = 10, lower: bool = True, upper: bool = True, digits: bool = True, punctuation: bool = False, safe: bool = False, noun: bool = False) -> str:
    if safe: return __import__("secrets").token_urlsafe(length)  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
    if noun: return install_n_import("randomname").get_name()
    string = __import__("string"); return ''.join(__import__("random").choices((string.ascii_lowercase if lower else "") + (string.ascii_uppercase if upper else "") + (string.digits if digits else "") + (string.punctuation if punctuation else ""), k=length))


def save_decorator(ext: str = ""):  # apply default paths, add extension to path, print the saved file path
    def decorator(func: Callable[..., Any]):
        def wrapper(obj: Any, path: Union[str, Path, None] = None, verbose: bool = True, add_suffix: bool = False, desc: str = "", class_name: str = "", **kwargs: Any):
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
            if verbose: print(f"SAVED {desc or path.name} {obj.__class__.__name__}: {f(repr(obj), justify=0, limit=50)}  @ `{path.absolute().as_uri()}`. Size = {path.stat().st_size / 1024**2:0.2f} MB")  # |  Directory: `{path.parent.absolute().as_uri()}`
            return path
        return wrapper
    return decorator


@save_decorator(".json")
def json(obj: Any, path: PLike, indent: Optional[str] = None, encoding: str = 'utf-8', **kwargs: Any):
    _ = encoding
    return Path(path).write_text(__import__("json").dumps(obj, indent=indent, default=lambda x: x.__dict__, **kwargs), encoding="utf-8")
@save_decorator(".yml")
def yaml(obj: dict[Any, Any], path: PLike, **kwargs: Any):
    with open(Path(path), 'w', encoding="utf-8") as file: __import__("yaml").dump(obj, file, **kwargs)
@save_decorator(".toml")
def toml(obj: dict[Any, Any], path: PLike, encoding: str = 'utf-8'): return Path(path).write_text(install_n_import("toml").dumps(obj), encoding=encoding)
@save_decorator(".ini")
def ini(obj: dict[Any, Any], path: PLike, **kwargs: Any):
    conf = install_n_import("configparser").ConfigParser(); conf.read_dict(obj)
    with open(path, 'w', encoding="utf-8") as configfile: conf.write(configfile, **kwargs)
@save_decorator(".csv")
def csv(obj: Any, path: PLike): return obj.to_frame('dtypes').reset_index().to_csv(str(path) + ".dtypes")
@save_decorator(".npy")
def npy(obj: Any, path: PLike, **kwargs: Any): return __import__('numpy').save(path, obj, **kwargs)
# @save_decorator(".mat")
# def mat(mdict, path=None, **kwargs): _ = [mdict.__setitem(key, []) for key, value in mdict.items() if value is None]; from scipy.io import savemat; savemat(str(path), mdict, **kwargs)  # Avoid using mat as it lacks perfect restoration: * `None` type is not accepted. Scalars are conveteed to [1 x 1] arrays.
@save_decorator(".pkl")
def vanilla_pickle(obj: Any, path: PLike, **kwargs: Any): return Path(path).write_bytes(__import__("pickle").dumps(obj, **kwargs))
@save_decorator(".pkl")
def pickle(obj: Any, path: PLike, r: bool = False, **kwargs: Any): return Path(path).write_bytes(__import__("dill").dumps(obj, recurse=r, **kwargs))  # In IPyconsole of Pycharm, this works only if object is of a an imported class. Don't use with objects defined at main.
def pickles(obj: Any, r: bool = False, **kwargs: Any): return __import__("dill").dumps(obj, r=r, **kwargs)
class Save:
    json = json
    yaml = yaml
    toml = toml
    ini = ini
    csv = csv
    npy = npy
    # mat = mat
    vanilla_pickle = vanilla_pickle
    pickle = pickle
    pickles = pickles


# ====================================== Object Management ====================================
class Base(object):
    def __init__(self, *args: Any, **kwargs: Any): _ = args, kwargs
    def __getstate__(self) -> dict[str, Any]: return self.__dict__.copy()
    def __setstate__(self, state: dict[str, Any]): self.__dict__.update(state)
    def __deepcopy__(self, *args: Any, **kwargs: Any):
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(__import__("copy").deepcopy(self.__dict__))
        return obj
    def __copy__(self, *args: Any, **kwargs: Any): obj = self.__class__(*args, **kwargs); obj.__dict__.update(self.__dict__.copy()); return obj
    # def eval(self, string_, func=False, other=False): return string_ if type(string_) is not str else eval((("lambda x, y: " if other else "lambda x:") if not str(string_).startswith("lambda") and func else "") + string_ + (self if False else ''))
    # def exec(self, expr: str) -> 'Base': exec(expr); return self  # exec returns None.
    def save(self, path: Union[str, Path, None] = None, add_suffix: bool = True, save_code: bool = False, verbose: bool = True, data_only: bool = True, desc: str = ""):  # + (".dat" if data_only else "")
        saved_file = Save.pickle(obj=self.__getstate__() if data_only else self, path=path, verbose=verbose, add_suffix=add_suffix, class_name="." + self.__class__.__name__, desc=desc or (f"Data of {self.__class__}" if data_only else desc))
        _ = self.save_code(path=saved_file.parent.joinpath(saved_file.name + "_saved_code.py")) if save_code else None; return self
    @classmethod
    def from_saved_data(cls, path: PLike, *args: Any, **kwargs: Any): obj = cls(*args, **kwargs); obj.__setstate__(dict(__import__("dill").loads(Path(path).read_bytes()))); return obj
    def save_code(self, path: Union[str, Path]):
        if hasattr(module := __import__("inspect").getmodule(self), "__file__"): file = Path(module.__file__)
        else: raise FileNotFoundError(f"Attempted to save code from a script running in interactive session! module should be imported instead.")
        _ = Path(path).expanduser().write_text(encoding='utf-8', data=file.read_text(encoding='utf-8')); return Path(path) if type(path) is str else path  # path could be tb.P, better than Path
    def get_attributes(self, remove_base_attrs: bool = True, return_objects: bool = False, fields: bool = True, methods: bool = True):
        import inspect
        remove_vals = Base().get_attributes(remove_base_attrs=False) if remove_base_attrs else []
        attrs: List[Any] = List(dir(self)).filter(lambda x: '__' not in x and not x.startswith('_')).remove(values=remove_vals)
        attrs: List[Any] = attrs.filter(lambda x: (inspect.ismethod(getattr(self, x)) if not fields else True) and ((not inspect.ismethod(getattr(self, x))) if not methods else True))  # logic (questionable): anything that is not a method is a field
        return List([getattr(self, x) for x in attrs]) if return_objects else List(attrs)
    def print(self, dtype: bool = False, attrs: bool = False, **kwargs: Any): return Struct(self.__dict__).update(attrs=self.get_attributes() if attrs else None).print(dtype=dtype, **kwargs)
    @staticmethod
    def get_state(obj: Any, repr_func: Callable[[Any], dict[str, Any]] = lambda x: x, exclude: Optional[list[str]] = None) -> dict[str, Any]:
        if not any([hasattr(obj, "__getstate__"), hasattr(obj, "__dict__")]): return repr_func(obj)
        return (tmp if type(tmp := obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__) is not dict else Struct(tmp).filter(lambda k, v: k not in (exclude or [])).apply2values(lambda k, v: Base.get_state(v, exclude=exclude, repr_func=repr_func)).__dict__)
    def viz_composition_heirarchy(self, depth: int = 3, obj: Any = None, filt: Optional[Callable[[Any], None]] = None):
        install_n_import("objgraph").show_refs([self] if obj is None else [obj], max_depth=depth, filename=str(filename := Path(__import__("tempfile").gettempdir()).joinpath("graph_viz_" + randstr(noun=True) + ".png")), filter=filt)
        _ = __import__("os").startfile(str(filename.absolute())) if __import__("sys").platform == "win32" else None; return filename


class List(Generic[T]):  # Inheriting from Base gives save method.  # Use this class to keep items of the same type."""
    def __init__(self, obj_list: Union[ListType[T], None, Iterator[T], Iterable[T]] = None) -> None: super().__init__(); self.list = list(obj_list) if obj_list is not None else []
    def __repr__(self): return f"List [{len(self.list)} elements]. First Item: " + f"{get_repr(self.list[0], justify=0, limit=100)}" if len(self.list) > 0 else f"An Empty List []"
    def print(self, sep: str = '\n', styler: Callable[[Any], str] = repr, return_str: bool = False, **kwargs: dict[str, Any]):
        res = sep.join([f"{idx:2}- {styler(item)}" for idx, item in enumerate(self.list)])
        _ = print(res) if not return_str else None; _ = kwargs
        return res if return_str else None
    def __deepcopy__(self, arg: Any) -> "List[T]": _ = arg; return List([__import__("copy").deepcopy(i) for i in self.list])
    def __bool__(self) -> bool: return bool(self.list)
    def __contains__(self, key: str) -> bool: return key in self.list
    def __copy__(self) -> 'List[T]': return List(self.list.copy())
    def __getstate__(self) -> list[T]: return self.list
    def __setstate__(self, state: list[T]): self.list = state
    def __len__(self) -> int: return len(self.list)
    def __iter__(self) -> Iterator[T]: return iter(self.list)
    def __array__(self): import numpy as np; return np.array(self.list)  # compatibility with numpy
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
        every = every if to is None else __import__("math").ceil(len(self) / to)
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
    def reduce(self, func: Callable[[T, T], T] = lambda x, y: x + y, default: Optional[T] = None) -> 'List[T]':  # type: ignore
        from functools import reduce
        if default is None: return List(reduce(func, self.list))  # type: ignore
        return List(reduce(func, self.list, default))  # type: ignore
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
    def to_series(self): return __import__("pandas").Series(self.list)
    def to_list(self) -> list[T]: return self.list
    def to_numpy(self, **kwargs: Any) -> 'Any': import numpy as np; return np.array(self.list, **kwargs)
    def to_struct(self, key_val: Optional[Callable[[T], Any]] = None) -> 'Struct': return Struct.from_keys_values_pairs(self.apply(func=key_val if key_val else lambda x: (str(x), x)))
    # def index(self, val: int) -> int: return self.list.index(val)
    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> 'List[T]': return List(self.list[start:stop:step])
    def __getitem__(self, key: Union[int, list[int], '_Slice']) -> Union[T, 'List[T]']:
        if isinstance(key, (list, Iterable, Iterator)): return List(self.list[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
        # elif isinstance(key, str): return List(item[key] for item in self.list)  # access keys like dictionaries.
        elif isinstance(key, int): return self.list[key]
        # assert isinstance(key, slice)
        return List(self.list[key])  # type: ignore # noqa: call-overload  # slices
    def apply(self, func: Union[Callable[[T], T2], Callable[[T, T2], T3]], *args: Any, other: Optional['List[T]'] = None, filt: Callable[[T], bool] = lambda _x: True,
              jobs: Optional[int] = None, prefer: Optional[str] = [None, 'processes', 'threads'][0], verbose: bool = False, desc: Optional[str] = None, **kwargs: Any,
              ) -> 'Union[List[Union[T2, T3]], List[T3]]':
        # if depth > 1: self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth - 1, **kwargs))
        from tqdm import tqdm
        iterator: Iterable[Any]
        if other is None:
            iterator = (self.list if not verbose else tqdm(self.list, desc=desc))
        else: iterator = (zip(self.list, other) if not verbose else tqdm(zip(self.list, other), desc=desc))
        if jobs:
            from joblib import Parallel, delayed
            if other is None: return List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, *args, **kwargs) for x in iterator))  # type: ignore
            return List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, y) for x, y in iterator))  # type: ignore
        if other is None:
            return List([func(x, *args, **kwargs) for x in iterator if filt(x)])
        # func_: Callable[[T, T2], T3]
        return List([func(x, y) for x, y in iterator])  # type: ignore
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
        super(Struct, self).__init__(); self.__dict__ = final_dict  # type: ignore
    # @staticmethod
    # def recursive_struct(mydict: dict[Any, Any]) -> 'Struct': struct = Struct(mydict); [struct.__setitem__(key, Struct.recursive_struct(val) if type(val) is dict else val) for key, val in struct.items()]; return struct
    # @staticmethod
    # def recursive_dict(struct) -> 'Struct': _ = [struct.__dict__.__setitem__(key, Struct.recursive_dict(val) if type(val) is Struct else val) for key, val in struct.__dict__.items()]; return struct.__dict__
    def save_json(self, path: Optional[PLike] = None, indent: Optional[str] = None): return Save.json(obj=self.__dict__, path=path, indent=indent)
    @classmethod
    def from_keys_values(cls, k: Iterable[str], v: Iterable[Any]) -> 'Struct': return Struct(dict(zip(k, v)))
    from_keys_values_pairs = classmethod(lambda cls, my_list: cls({k: v for k, v in my_list}))
    @classmethod
    def from_names(cls, names: list[str], default_: Optional[Any] = None) -> 'Struct': return cls.from_keys_values(k=names, v=default_ or [None] * len(names))  # Mimick NamedTuple and defaultdict
    def spawn_from_values(self, values: Union[list[Any], List[Any]]) -> 'Struct': return self.from_keys_values(list(self.keys()), values)
    def spawn_from_keys(self, keys: Union[list[str], List[str]]) -> 'Struct': return self.from_keys_values(keys, list(self.values()))
    def to_default(self, default: Optional[Callable[[], Any]] = lambda: None): tmp2 = __import__("collections").defaultdict(default); tmp2.update(self.__dict__); self.__dict__ = tmp2; return self
    def __str__(self, sep: str = "\n"): return config(self.__dict__, sep=sep)
    def __getattr__(self, item: str) -> 'Struct':
        try: return self.__dict__[item]
        except KeyError as ke: raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}') from ke  # this works better with the linter. replacing Key error with Attribute error makes class work nicely with hasattr() by returning False.
    clean_view = property(lambda self: type("TempClass", (object,), self.__dict__))
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
    def to_dataframe(self, *args: Any, **kwargs: Any): return __import__("pandas").DataFrame(self.__dict__, *args, **kwargs)
    def keys(self, verbose: bool = False) -> 'List[Any]': return List(list(self.__dict__.keys())) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.keys())
    def values(self, verbose: bool = False) -> 'List[Any]': return List(list(self.__dict__.values())) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.values())
    def items(self, verbose: bool = False, desc: str = "") -> 'List[Any]': return List(self.__dict__.items()) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.items(), desc=desc)
    def get(self, key: Optional[str] = None, default: Optional[Any] = None, strict: bool = False, keys: Union[None, list[str]] = None) -> 'Union[Any, List[Any]]':
        if keys is not None: return List([self.__dict__.get(key, default) if not strict else self[key] for key in keys])
        if key is not None: return (self.__dict__.get(key, default) if not strict else self[key])
        else: raise ValueError("Either key or keys should be passed.")
    def apply2keys(self, kv_func: Callable[[Any, Any], Any], verbose: bool = False, desc: str = "") -> 'Struct': return Struct({kv_func(key, val): val for key, val in self.items(verbose=verbose, desc=desc)})
    def apply2values(self, kv_func: Callable[[Any, Any], Any], verbose: bool = False, desc: str = "") -> 'Struct': _ = [self.__setitem__(key, kv_func(key, val)) for key, val in self.items(verbose=verbose, desc=desc)]; return self
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
        col3: List[Any] = self.values().apply(lambda x: get_repr(x, justify=justify, limit=limit).replace("\n", " "))
        array = np.array([self.keys(), col2, col3]).T
        res: pd.DataFrame = pd.DataFrame(array, columns=["key", "dtype", "details"])
        return res if not return_str else str(res)
    def print(self, dtype: bool = True, return_str: bool = False, justify: int = 30, as_config: bool = False, as_yaml: bool = False,  # type: ignore # pylint: disable=W0237
              limit: int = 50, title: str = "", attrs: bool = False, **kwargs: Any) -> Union[str, None]:  # type: ignore
        _ = attrs
        import pandas as pd
        if as_config and not return_str:
            install_n_import("rich").inspect(self, value=False, title=title, docs=False, sort=False)
            return None
        if not bool(self):
            if return_str: return f"Empty Struct."
            else: print(f"Empty Struct."); return None
        else:
            if as_yaml or as_config:
                tmp: str = __import__("yaml").dump(self.__dict__) if as_yaml else config(self.__dict__, justify=justify, **kwargs)
                if return_str: return tmp
                else: print(tmp); return None
            else:
                tmp2 = self._pandas_repr(justify=justify, return_str=False, limit=limit)
                if isinstance(tmp2, pd.DataFrame):
                    res = tmp2.drop(columns=[] if dtype else ["dtype"])
                else: raise TypeError(f"Unexpected type {type(tmp2)}")
                if not return_str:
                    if install_n_import("tabulate"):
                        install_n_import("rich").print(res.to_markdown())
                    else: print(res)
                    return None
                return str(res)

    @staticmethod
    def concat_values(*dicts: dict[Any, Any], orient: str = 'list') -> 'Struct':
        import pandas as pd
        assert orient in ["dict", "list", "series", "split", "tight", "index"]
        return Struct(pd.concat(List(dicts).apply(lambda x: Struct(x).to_dataframe()).list).to_dict(orient=orient))  # type: ignore
    def plot_plt(self, title: str = '', xlabel: str = '', ylabel: str = '', **kwargs: Any):
        from crocodile.matplotlib_management import LineArtist
        artist = LineArtist(figname='Structure Plot', **kwargs)
        artist.plot_dict(self.__dict__, title=title, xlabel=xlabel, ylabel=ylabel)
        return artist
    def plot_plotly(self):
        from crocodile.plotly_management import px
        fig = px.line(self.__dict__)
        fig.show()
        return fig


def set_pandas_display(rows: int = 1000, columns: int = 1000, width: int = 5000, colwidth: int = 40) -> None:
    import pandas as pd; pd.set_option('display.max_colwidth', colwidth); pd.set_option('display.max_columns', columns); pd.set_option('display.width', width); pd.set_option('display.max_rows', rows)
def set_pandas_auto_width(): __import__("pandas").set_option('width', 0)  # this way, pandas is told to detect window length and act appropriately.  For fixed width host windows, this is recommended to avoid chaos due to line-wrapping.
def set_numpy_display(precision: int = 3, linewidth: int = 250, suppress: bool = True, floatmode: str = 'fixed', **kwargs: Any) -> None: __import__("numpy").set_printoptions(precision=precision, suppress=suppress, linewidth=linewidth, floatmode=floatmode, **kwargs)
def config(mydict: dict[Any, Any], sep: str = "\n", justify: int = 15, quotes: bool = False): return sep.join([f"{key:>{justify}} = {repr(val) if quotes else val}" for key, val in mydict.items()])
def f(str_: str, limit: int = 10000000000, justify: int = 50, direc: str = "<") -> str: return f"{(str_[:limit - 4] + '... ' if len(str_) > limit else str_):{direc}{justify}}"
def eng(): __import__("pandas").set_eng_float_format(accuracy=3, use_eng_prefix=True); __import__("pandas").options.float_format = '{:, .5f}'.format; __import__("pandas").set_option('precision', 7)  # __import__("pandas").set_printoptions(formatter={'float': '{: 0.3f}'.format})
def outline(array: 'Any', name: str = "Array", printit: bool = True): str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"; _ = print(str_) if printit else None; return str_
def get_repr(data: Any, justify: int = 15, limit: int = 10000, direc: str = "<") -> str:
    if (dtype := data.__class__.__name__) in {'list', 'str'}: str_ = data if dtype == 'str' else f"list. length = {len(data)}. " + ("1st item type: " + str(type(data[0])).split("'")[1]) if len(data) > 0 else " "
    elif dtype in {"DataFrame", "Series"}: str_ = f"Pandas DF: shape = {data.shape}, dtype = {data.dtypes}." if dtype == 'DataFrame' else f"Pandas Series: Length = {len(data)}, Keys = {get_repr(data.keys().to_list())}."
    else: str_ = f"shape = {data.shape}, dtype = {data.dtype}." if dtype == 'ndarray' else repr(data)
    return f(str_.replace("\n", ", "), justify=justify, limit=limit, direc=direc)
def print_string_list(mylist: list[Any], char_per_row: int = 125, sep: str = " ", style: Callable[[Any], str] = str, _counter: int = 0):
    for item in mylist: _ = print("") if (_counter + len(style(item))) // char_per_row > 0 else print(style(item), end=sep); _counter = len(style(item)) if (_counter + len(style(item))) // char_per_row > 0 else _counter + len(style(item))
class Display:
    set_pandas_display = staticmethod(set_pandas_display)
    set_pandas_auto_width = staticmethod(set_pandas_auto_width)
    set_numpy_display = staticmethod(set_numpy_display)
    config = staticmethod(config)
    f = staticmethod(f)
    eng = staticmethod(eng)
    outline = staticmethod(outline)
    get_repr = staticmethod(get_repr)
    print_string_list = staticmethod(print_string_list)  # or D = type('D', (object, ), dict(set_pandas_display


if __name__ == '__main__':
    pass
