
from pathlib import Path
from typing import Optional, Union, TypeVar, Type, Literal, Any, Callable, Iterable, Hashable, ParamSpec

from crocodile.core_modules.core_2 import Base, List
from crocodile.core_modules.core_4 import Display


T = TypeVar('T')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
PLike = Union[str, Path]
PS = ParamSpec('PS')


class Struct(Base):  # inheriting from dict gives `get` method, should give `__contains__` but not working. # Inheriting from Base gives `save` method.
    """Use this class to keep bits and sundry items. Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience"""
    def __init__(self, dictionary: Union[dict[Any, Any], Type[object], None] = None, **kwargs: Any):
        if dictionary is None or isinstance(dictionary, dict): final_dict: dict[str, Any] = {} if dictionary is None else dictionary
        else:
            final_dict = (dict(dictionary) if dictionary.__class__.__name__ == "mappingproxy" else dictionary.__dict__)  # type: ignore
        final_dict.update(kwargs)  # type ignore
        super(Struct, self).__init__()
        self.__dict__ = final_dict  # type: ignore
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
