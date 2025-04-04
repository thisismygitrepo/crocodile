
from pathlib import Path
from typing import Union, TypeVar, Literal, Any, Callable, Protocol, ParamSpec


_Slice = TypeVar('_Slice', bound='Slicable')
class Slicable(Protocol):
    def __getitem__(self: _Slice, i: slice) -> _Slice: ...


T = TypeVar('T')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
PLike = Union[str, Path]
PS = ParamSpec('PS')


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
