from pathlib import Path
from typing import Optional, Union, Generic, TypeVar, Literal, List as ListType, Any, Iterator, Callable, Iterable, Protocol, ParamSpec

from crocodile.core_modules.core_1 import Save, randstr, install_n_import
from crocodile.core_modules.core_4 import Display



_Slice = TypeVar('_Slice', bound='Slicable')
class Slicable(Protocol):
    def __getitem__(self: _Slice, i: slice) -> _Slice: ...


T = TypeVar('T')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
PLike = Union[str, Path]
PS = ParamSpec('PS')

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
    def print(self, dtype: bool = False, attrs: bool = False, **kwargs: Any):
        from crocodile.core_modules.core_3 import Struct
        return Struct(self.__dict__).update(attrs=self.get_attributes() if attrs else None).print(dtype=dtype, **kwargs)
    @staticmethod
    def get_state(obj: Any, repr_func: Callable[[Any], dict[str, Any]] = lambda x: x, exclude: Optional[list[str]] = None) -> dict[str, Any]:
        if not any([hasattr(obj, "__getstate__"), hasattr(obj, "__dict__")]): return repr_func(obj)
        from crocodile.core_modules.core_3 import Struct
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
    # def to_struct(self, key_val: Optional[Callable[[T], tuple[Any, Any]]] = None) -> 'Struct':
    #     return Struct.from_keys_values_pairs(self.apply(func=key_val if key_val else lambda x: (str(x), x)).list)
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
