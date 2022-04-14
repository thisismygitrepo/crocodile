
"""
"""
# from __future__ import annotations
from pathlib import Path

# ============================== Accessories ============================================


def validate_name(astring: str, replace='_') -> str: return __import__("re").sub(r'^(?=\d)|\W', replace, str(astring))
def timestamp(fmt=None, name=None): return (name + '_' + __import__("datetime").datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')) if name is not None else __import__("datetime").datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')  # isoformat is not compatible with file naming convention, fmt here is.


def str2timedelta(shift):
    """Converts a human readable string like '1m' or '1d' to a timedate object. In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    key, val = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}[shift[-1]], eval(shift[:-1])
    key, val = ("days", val * 30) if key == "months" else (("weeks", val * 52) if key == "years" else (key, val)); return __import__("datetime").timedelta(**{key: val})


def randstr(length=15, lower=True, upper=True, digits=True, punctuation=False, safe=False) -> str:
    if safe: return __import__("secrets").token_urlsafe(length)  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
    string = __import__("string"); return ''.join(__import__("random").choices((string.ascii_lowercase if lower else "") + (string.ascii_uppercase if upper else "") + (string.digits if digits else "") + (string.punctuation if punctuation else ""), k=length))


def install_n_import(package, name=None):
    try: return __import__(package)
    except ImportError: __import__("subprocess").check_call([__import__("sys").executable, "-m", "pip", "install", name or package])
    return __import__(package)


# ====================================== Classes ====================================


def save_decorator(ext=""):  # apply default paths, add extension to path, print the saved file path
    def decorator(func):
        def wrapper(obj, path: str = None, verbose=True, add_suffix=True, desc="", class_name="", **kwargs):
            if path is None: path = Path.home().joinpath("tmp_results/tmp_files").joinpath(randstr()); print(f"tb.core: Warning: Path not passed to {func}. A default path has been chosen: {path.absolute().as_uri()}") if verbose else None
            if add_suffix:
                [(print(f"tb.core: Warning: suffix `{a_suffix}` is added to path passed {path}") if verbose else None) for a_suffix in [ext, class_name] if a_suffix not in str(path)]
                path = str(path).replace(ext, "").replace(class_name, "") + class_name + ext; path = Path(path).expanduser().resolve(); path.parent.mkdir(parents=True, exist_ok=True)
            func(path=path, obj=obj, **kwargs); print(f"SAVED {desc} {obj.__class__.__name__}: {Display.f(repr(obj), 50)}  @ `{path.absolute().as_uri()}` |  Directory: `{path.parent.absolute().as_uri()}`") if verbose else None
            return path
        return wrapper
    return decorator


class Save:
    @staticmethod
    @save_decorator(".csv")
    def csv(obj, path=None): obj.to_frame('dtypes').reset_index().to_csv(path + ".dtypes")
    @staticmethod
    @save_decorator(".npy")
    def npy(obj, path, **kwargs): import numpy as np; np.save(path, obj, **kwargs)
    @staticmethod
    @save_decorator(".mat")
    def mat(mdict, path=None, **kwargs): [mdict.__setitem(key, []) for key, value in mdict.items() if value is None]; from scipy.io import savemat; savemat(str(path), mdict, **kwargs)  # Avoid using mat as it lacks perfect restoration: * `None` type is not accepted. Scalars are conveteed to [1 x 1] arrays.
    @staticmethod
    @save_decorator(".json")
    def json(obj, path=None, **kwargs): Path(path).write_text(__import__("json").dumps(obj, default=lambda x: x.__dict__, **kwargs))
    @staticmethod
    @save_decorator(".yml")
    def yaml(obj, path, **kwargs): Path(path).write_bytes(__import__("yaml").dumps(obj, **kwargs))
    @staticmethod
    @save_decorator(".pkl")
    def vanilla_pickle(obj, path, **kwargs): Path(path).write_bytes(__import__("pickle").dumps(obj, **kwargs))
    @staticmethod
    @save_decorator(".pkl")
    def pickle(obj=None, path=None, r=False, **kwargs): Path(path).write_bytes(__import__("dill").dumps(obj, recurse=r, **kwargs))
    pickles = staticmethod(lambda obj: __import__("dill").dumps(obj))


class Base(object):
    def __init__(self, *args, **kwargs): pass
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)
    def print(self, typeinfo=False): Struct(self.__dict__).print(dtype=typeinfo)
    def __deepcopy__(self, *args, **kwargs): obj = self.__class__(*args, **kwargs); obj.__dict__.update(__import__("copy").deepcopy(self.__dict__)); return obj
    def __copy__(self, *args, **kwargs): obj = self.__class__(*args, **kwargs); obj.__dict__.update(self.__dict__.copy()); return obj
    def evalstr(self, string_, func=True, other=False): return string_ if type(string_) is not str else eval((("lambda x, y: " if other else "lambda x:") if not string_.startswith("lambda") and func else "") + string_ + (self if False else ''))
    def save(self, path=None, add_suffix=True, save_code=False, verbose=True, data_only=False, desc=""):
        saved_file = Save.pickle(obj=self.__getstate__() if data_only else self, path=path, verbose=verbose, add_suffix=add_suffix, class_name="." + self.__class__.__name__ + (".dat" if data_only else ""), desc=desc or (f"Data of {self.__class__}" if data_only else desc))
        if save_code: self.save_code(path=saved_file.parent.joinpath(saved_file.name + "_saved_code.py")); return self
    @classmethod
    def from_saved_data(cls, path, *args, **kwargs): obj = cls(*args, **kwargs); obj.__setstate__(dict(__import__("dill").loads(Path(path).read_bytes()))); return obj
    def save_code(self, path):
        if hasattr(module := __import__("inspect").getmodule(self), "__file__"): file = Path(module.__file__)
        else: raise FileNotFoundError(f"Attempted to save code from a script running in interactive session! module should be imported instead.")
        Path(path).expanduser().write_text(file.read_text()); return Path(path) if type(path) is str else path  # path could be tb.P, better than Path
    def get_attributes(self, remove_base_attrs=True, return_objects=False, fields=True, methods=True):
        attrs = list(filter(lambda x: ('__' not in x) and not x.startswith("_"), dir(self))); [attrs.remove(x) for x in Base().get_attributes(remove_base_attrs=False)] if remove_base_attrs else None
        if not fields: attrs = list(filter(lambda x: __import__("inspect").ismethod(getattr(self, x)), attrs))  # logic (questionable): anything that is not a method is a field
        if not methods: attrs = list(filter(lambda x: not __import__("inspect").ismethod(getattr(self, x)), attrs))
        if return_objects: attrs = [getattr(self, x) for x in attrs]; return List(attrs)
    def viz_composition_heirarchy(self, depth=3, obj=None, filt=None):
        filename = Path(__import__("tempfile").gettempdir()).joinpath("graph_viz_" + randstr() + ".png")
        install_n_import("objgraph").show_refs([self] if obj is None else [obj], max_depth=depth, filename=str(filename), filter=filt)
        __import__("os").startfile(str(filename.absolute())) if __import__("sys").platform == "win32" else None; return filename


class List(Base):  # Inheriting from Base gives save method.  # Use this class to keep items of the same type."""
    def __init__(self, obj_list=None): super().__init__(); self.list = list(obj_list) if obj_list is not None else []
    def save_items(self, directory, names=None, saver=None): [(saver or Save.pickle)(path=directory / name, obj=item) for name, item in zip(names or range(len(self)), self.list)]
    def __repr__(self): return f"List [{len(self.list)} elements]. First Item: " + f"{Display.get_repr(self.list[0])}" if len(self.list) > 0 else f"An Empty List []"
    def __deepcopy__(self): return List([__import__("copy").deepcopy(i) for i in self.list])
    def __bool__(self): return bool(self.list)
    def __contains__(self, key): return key in self.list
    def __copy__(self) -> 'List': return List(self.list.copy())
    def __getstate__(self): return self.list
    def __setstate__(self, state): self.list = state
    def __len__(self): return len(self.list)
    def __iter__(self): return iter(self.list)
    len = property(lambda self: self.list.__len__())
    # ================= call methods =====================================
    def __getattr__(self, name) -> 'List': return List(getattr(i, name) for i in self.list)  # fallback position when __getattribute__ mechanism fails.
    def __call__(self, *args, **kwargs) -> 'List': return List(i(*args, **kwargs) for i in self.list)
    # ======================== Access Methods ==========================================
    def __setitem__(self, key, value): self.list[key] = value
    def sample(self, size=1, replace=False, p=None) -> 'List': return self[list(__import__("numpy").random.choice(len(self), size, replace=replace, p=p))]
    def index_items(self, idx) -> 'List': return List([item[idx] for item in self.list])
    def find_index(self, func) -> 'List': return List([idx for idx, x in enumerate(self.list) if self.evalstr(func)(x)])
    def filter(self, func): return List([item for item in self.list if self.evalstr(func, func=True)(item)])
    # ======================= Modify Methods ===============================
    def reduce(self, func) -> 'List': return __import__("functools").reduce(self.evalstr(func, func=True, other=True), self.list)
    def append(self, item) -> 'List': self.list.append(item); return self
    def __add__(self, other) -> 'List': return List(self.list + list(other))  # implement coersion
    def __radd__(self, other) -> 'List': return List(self.list + list(other))
    def __iadd__(self, other) -> 'List': self.list = self.list + list(other); return self  # inplace add.
    def sort(self, key=None, reverse=False) -> 'List': self.list.sort(key=key, reverse=reverse); return self
    def sorted(self, *args, **kwargs) -> 'List': return List(sorted(self.list, *args, **kwargs))
    def insert(self, __index: int, __object): self.list.insert(__index, __object); return self
    def exec(self, expr: str) -> 'List': _ = self; return exec(expr)
    def modify(self, expr: str, other=None) -> 'List': [exec(expr) for idx, x in enumerate(self.list)] if other is None else [exec(expr) for idx, (x, y) in enumerate(zip(self.list, other))]; return self
    def remove(self, value=None, values=None) -> 'List': [self.list.remove(a_val) for a_val in ((values or []) + ([value] if value else []))]; return self
    def print(self, nl=1, sep=False, style=repr): [print(f"{idx:2}- {style(item)}", '\n' * (nl-1), sep * 100 if sep else ' ') for idx, item in enumerate(self.list)]
    def to_series(self): return __import__("pandas").Series(self.list)
    def to_list(self) -> list: return self.list
    def to_numpy(self): import numpy as np; return np.array(self.list)
    np = property(lambda self: self.to_numpy())
    def to_struct(self, key_val=None) -> 'Struct': return Struct.from_keys_values_pairs(self.apply(self.evalstr(key_val) if key_val else lambda x: (str(x), x)))
    def __getitem__(self, key: str or list or slice) -> 'List':
        if type(key) is list: return List(self[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
        elif type(key) is str: return List(item[key] for item in self.list)  # access keys like dictionaries.
        return self.list[key] if type(key) is not slice else List(self.list[key])  # must be an integer or slice: behaves similarly to Numpy A[1] vs A[1:2]
    def apply(self, func, *args, other=None, filt=lambda x: True, jobs=None, depth=1, verbose=False, desc=None, **kwargs) -> 'List':
        if depth > 1: self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth-1, **kwargs)); func = self.evalstr(func, other=bool(other))
        iterator = (self.list if not verbose else install_n_import("tqdm").tqdm(self.list, desc=desc)) if other is None else (zip(self.list, other) if not verbose else install_n_import("tqdm").tqdm(zip(self.list, other), desc=desc))
        if jobs: from joblib import Parallel, delayed; return List(Parallel(n_jobs=jobs)(delayed(func)(x, *args, **kwargs) for x in iterator)) if other is None else List(Parallel(n_jobs=jobs)(delayed(func)(x, y) for x, y in iterator))
        return List([func(x, *args, **kwargs) for x in iterator if filt(x)]) if other is None else List([func(x, y) for x, y in iterator])
    def to_dataframe(self, names=None, minimal=False, obj_included=True):
        df = __import__("pandas").DataFrame(columns=(['object'] if obj_included or names else []) + list(self.list[0].__dict__.keys()))
        if minimal: return df
        for i, obj in enumerate(self.list):  # Populate the dataframe:
            if obj_included or names: df.loc[i] = ([obj] if names is None else [names[i]]) + list(self.list[i].__dict__.values())
            else: df.loc[i] = list(self.list[i].__dict__.values())
        return df


class Struct(Base):  # inheriting from dict gives `get` method, should give `__contains__` but not working. # Inheriting from Base gives `save` method.
    """Use this class to keep bits and sundry items. Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience"""
    def __init__(self, dictionary=None, **kwargs):
        if dictionary is None or type(dictionary) is dict: final_dict = dict() if dictionary is None else dictionary
        else: final_dict = (dict(dictionary) if dictionary.__class__.__name__ == "mappingproxy" else dictionary.__dict__)
        final_dict.update(kwargs); super(Struct, self).__init__(); self.__dict__ = final_dict
    @staticmethod
    def recursive_struct(mydict) -> 'Struct': struct = Struct(mydict); [struct.__setitem__(key, Struct.recursive_struct(val) if type(val) is dict else val) for key, val in struct.items()]; return struct
    @staticmethod
    def recursive_dict(struct) -> 'Struct': [struct.__dict__.__setitem__(key, Struct.recursive_dict(val) if type(val) is Struct else val) for key, val in struct.__dict__.items()]; return struct.__dict__
    def save_json(self, path=None): return Save.json(obj=self.__dict__, path=path)
    from_keys_values = classmethod(lambda cls, k, v: cls(dict(zip(k, v))))
    from_keys_values_pairs = classmethod(lambda cls, my_list: cls({k: v for k, v in my_list}))
    @classmethod
    def from_names(cls, names, default_=None) -> 'Struct': return cls.from_keys_values(k=names, v=default_ or [None] * len(names))  # Mimick NamedTuple and defaultdict
    def spawn_from_values(self, values) -> 'Struct': return self.from_keys_values(self.keys(), self.evalstr(values, func=False))
    def spawn_from_keys(self, keys) -> 'Struct': return self.from_keys_values(self.evalstr(keys, func=False), self.values())
    def to_default(self, default=lambda: None): tmp2 = __import__("collections").defaultdict(default); tmp2.update(self.__dict__); self.__dict__ = tmp2; return self
    def __str__(self, newline=True): return Display.config(self.__dict__, newline=newline)  # == self.print(config=True)
    def __getattr__(self, item) -> 'Struct':
        try: return self.__dict__[item]
        except KeyError: raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')  # this works better with the linter. replacing Key error with Attribute error makes class work nicely with hasattr() by returning False.
    clean_view = property(lambda self: type("TempClass", (object,), self.__dict__))
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
    def copy(self) -> 'Struct': return Struct(self.__dict__.copy())
    dict = property(lambda self: self.__dict__)   # allows getting dictionary version without accessing private memebers explicitly.
    @dict.setter
    def dict(self, adict): self.__dict__ = adict
    def to_dataframe(self, *args, **kwargs): return __import__("pandas").DataFrame(self.__dict__, *args, **kwargs)
    def keys(self, verbose=False) -> 'List': return List(list(self.dict.keys())) if not verbose else install_n_import("tqdm").tqdm(self.dict.keys())
    def values(self, verbose=False) -> 'List': return List(list(self.dict.values())) if not verbose else install_n_import("tqdm").tqdm(self.dict.values())
    def items(self, verbose=False, desc="") -> 'List': return List(self.dict.items()) if not verbose else install_n_import("tqdm").tqdm(self.dict.items(), desc=desc)
    def get_values(self, keys) -> 'List': return List([self[key] for key in keys])
    def apply_to_keys(self, kv_func, verbose=False, desc="") -> 'Struct': return Struct({kv_func(key, val): val for key, val in self.items(verbose=verbose, desc=desc)})
    def apply_to_values(self, kv_func, verbose=False, desc="") -> 'Struct': [self.__setitem__(key, kv_func(key, val)) for key, val in self.items(verbose=verbose, desc=desc)]; return self
    def filter(self, kv_func=None) -> 'Struct': return Struct({key: self[key] for key, val in self.items() if kv_func(key, val)})
    def inverse(self) -> 'Struct': return Struct({v: k for k, v in self.dict.items()})
    def update(self, *args, **kwargs) -> 'Struct': self.__dict__.update(Struct(*args, **kwargs).__dict__); return self
    def delete(self, key=None, keys=None, kv_func=None) -> 'Struct': [self.__dict__.__delitem__(key) for key in ([key] if key else [] + keys or [])]; [self.__dict__.__delitem__(k) for k, v in self.items() if kv_func(k, v)] if kv_func is not None else None; return self
    def _pandas_repr(self, limit): return __import__("pandas").DataFrame(__import__("numpy").array([self.keys(), self.values().apply(lambda x: str(type(x)).split("'")[1]), self.values().apply(lambda x: Display.get_repr(x, limit=limit).replace("\n", " "))]).T, columns=["key", "dtype", "details"])
    def print(self, dtype=True, return_str=False, limit=50, config=False, yaml=False, newline=True): res = f"Empty Struct." if not bool(self) else ((__import__("yaml").dump(self.__dict__) if yaml else Display.config(self.__dict__, newline=newline, limit=limit)) if yaml or config else self._pandas_repr(limit).drop(columns=[] if dtype else ["dtype"])); print(res) if not return_str else None; return res if return_str else self
    @staticmethod
    def concat_values(*dicts, orient='list') -> 'Struct': return Struct(__import__("pandas").concat(List(dicts).apply(lambda x: Struct(x).to_dataframe())).to_dict(orient=orient))
    def plot(self, artist=None, use_plt=True):
        if not use_plt: fig = __import__("crocodile.plotly_management").px.line(self.__dict__); fig.show(); return fig
        plt = __import__("matplotlib").pyplot
        if artist is None: fig, artist = plt.subplots()  # artist = Artist(figname='Structure Plot')  # removed for disentanglement
        for key, val in self.items(): artist.plot(val, label=key)
        try: artist.legend()
        except AttributeError: pass
        return artist


class Display:
    @staticmethod
    def set_pandas_display(rows=1000, columns=1000, width=5000, colwidth=40): import pandas as pd; pd.set_option('display.max_colwidth', colwidth); pd.set_option('display.max_columns', columns); pd.set_option('display.width', width); pd.set_option('display.max_rows', rows)
    set_pandas_auto_width = staticmethod(lambda: __import__("pandas").set_option('display.width', 0))  # this way, pandas is told to detect window length and act appropriately.  For fixed width host windows, this is recommended to avoid chaos due to line-wrapping.
    config = staticmethod(lambda mydict, newline=True, limit=15, justify=True: "".join([f"{key:>{limit if justify else 0}} = {val}" + ("\n" if newline else ", ") for key, val in mydict.items()]))
    f = staticmethod(lambda str_, limit=50, direc="<": f'{(str_[:limit - 4] + " ..." if len(str_) > limit else str_):{direc}{limit}}')
    @staticmethod
    def eng(): __import__("pandas").set_eng_float_format(accuracy=3, use_eng_prefix=True); __import__("pandas").options.display.float_format = '{:, .5f}'.format; __import__("pandas").set_option('precision', 7)  # __import__("pandas").set_printoptions(formatter={'float': '{: 0.3f}'.format})
    @staticmethod
    def outline(array, name="Array", printit=True): str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"; print(str_) if printit else None; return str_
    @staticmethod
    def get_repr(data, limit=50, justify=False):
        if type(data) in {list, str}: string_ = data if type(data) is str else f"list. length = {len(data)}. " + ("1st item type: " + str(type(data[0])).split("'")[1]) if len(data) > 0 else " "
        elif type(data) is __import__("numpy").ndarray: string_ = f"shape = {data.shape}, dtype = {data.dtype}."
        elif type(data) is __import__("pandas").DataFrame: string_ = f"Pandas DF: shape = {data.shape}, dtype = {data.dtypes}."
        elif type(data) is __import__("pandas").Series: string_ = f"Pandas Series: Length = {len(data)}, Keys = {Display.get_repr(data.keys().to_list())}."
        else: string_ = repr(data)
        return f'{(string_[:limit - 4] + "... " if len(string_) > limit else string_):>{limit if justify else 0}}'
    @staticmethod
    def print_string_list(mylist, char_per_row=125, sep=" "):
        counter = 0
        for item in mylist:
            print(item, end=sep); counter += len(item)
            if not counter <= char_per_row: counter = 0; print("\n")


if __name__ == '__main__':
    pass
