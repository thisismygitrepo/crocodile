
"""
"""

from pathlib import Path


# ============================== Accessories ============================================
def validate_name(astring: str, replace='_') -> str: return __import__("re").sub(r'[^-a-zA-Z0-9_.()]+', replace, str(astring))
def timestamp(fmt=None, name=None): return ((name + '_') if name is not None else '') + __import__("datetime").datetime.now().strftime(fmt or '%Y-%m-%d-%I-%M-%S-%p-%f')  # isoformat is not compatible with file naming convention, fmt here is.
def str2timedelta(shift):  # Converts a human readable string like '1m' or '1d' to a timedate object. In essence, its gives a `2m` short for `pd.timedelta(minutes=2)`"""
    key, val = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}[shift[-1]], eval(shift[:-1])
    key, val = ("days", val * 30) if key == "months" else (("weeks", val * 52) if key == "years" else (key, val)); return __import__("datetime").timedelta(**{key: val})
def randstr(length=10, lower=True, upper=True, digits=True, punctuation=False, safe=False) -> str:
    if safe: return __import__("secrets").token_urlsafe(length)  # interannly, it uses: random.SystemRandom or os.urandom which is hardware-based, not pseudo
    string = __import__("string"); return ''.join(__import__("random").choices((string.ascii_lowercase if lower else "") + (string.ascii_uppercase if upper else "") + (string.digits if digits else "") + (string.punctuation if punctuation else ""), k=length))
def install_n_import(package, name=None, **kwargs):  # sometimes package name is different from import, e.g. skimage.
    try: return __import__(package, **kwargs)
    except ImportError: __import__("subprocess").check_call([__import__("sys").executable, "-m", "pip", "install", name or package]); return __import__(package, **kwargs)
def get_env(): import crocodile.environment as env; return env


def save_decorator(ext=""):  # apply default paths, add extension to path, print the saved file path
    def decorator(func):
        def wrapper(obj, path: str = None, verbose=True, add_suffix=True, desc="", class_name="", **kwargs):
            if path is None: path = Path.home().joinpath("tmp_results/tmp_files").joinpath(randstr()); print(f"tb.core: Warning: Path not passed to {func}. A default path has been chosen: {path.absolute().as_uri()}") if verbose else None
            if add_suffix:
                [(print(f"tb.core: Warning: suffix `{a_suffix}` is added to path passed {path}") if verbose else None) for a_suffix in [ext, class_name] if a_suffix not in str(path)]
                path = str(path).replace(ext, "").replace(class_name, "") + class_name + ext; path = Path(path).expanduser().resolve(); path.parent.mkdir(parents=True, exist_ok=True)
            func(path=path, obj=obj, **kwargs); print(f"SAVED {desc or path.name} {obj.__class__.__name__}: {f(repr(obj), justify=0, limit=50)}  @ `{path.absolute().as_uri()}`. Size = {path.stat().st_size / 1024**2:0.2f} MB") if verbose else None  # |  Directory: `{path.parent.absolute().as_uri()}`
            return path
        return wrapper
    return decorator


@save_decorator(".json")
def json(obj, path=None, indent=None, **kwargs): return Path(path).write_text(__import__("json").dumps(obj, indent=indent, default=lambda x: x.__dict__, **kwargs))
@save_decorator(".yml")
def yaml(obj, path, **kwargs):
    with open(Path(path), 'w') as file: __import__("yaml").dump(obj, file, **kwargs)
@save_decorator(".toml")
def toml(obj: dict, path): return Path(path).write_text(install_n_import("toml").dumps(obj))
@save_decorator(".ini")
def ini(obj: dict, path, **kwargs):
    conf = install_n_import("configparser").ConfigParser(); conf.read_dict(obj)
    with open(path, 'w') as configfile: conf.write(configfile, **kwargs)
@save_decorator(".csv")
def csv(obj, path=None): return obj.to_frame('dtypes').reset_index().to_csv(path + ".dtypes")
@save_decorator(".npy")
def npy(obj, path, **kwargs): return __import__('numpy').save(path, obj, **kwargs)
@save_decorator(".mat")
def mat(mdict, path=None, **kwargs): [mdict.__setitem(key, []) for key, value in mdict.items() if value is None]; from scipy.io import savemat; savemat(str(path), mdict, **kwargs)  # Avoid using mat as it lacks perfect restoration: * `None` type is not accepted. Scalars are conveteed to [1 x 1] arrays.
@save_decorator(".pkl")
def vanilla_pickle(obj, path, **kwargs): return Path(path).write_bytes(__import__("pickle").dumps(obj, **kwargs))
@save_decorator(".pkl")
def pickle(obj=None, path=None, r=False, **kwargs): return Path(path).write_bytes(__import__("dill").dumps(obj, recurse=r, **kwargs))  # In IPyconsole of Pycharm, this works only if object is of a an imported class. Don't use with objects defined at main.
def pickles(obj, r=False, **kwargs): return __import__("dill").dumps(obj, r=r, **kwargs)
class Save: json = json; yaml = yaml; toml = toml; ini = ini; csv = csv; npy = npy; mat = mat; vanilla_pickle = vanilla_pickle; pickle = pickle; pickles = pickles


# ====================================== Object Management ====================================
class Base(object):
    def __init__(self, *args, **kwargs): _ = args, kwargs
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, state): self.__dict__.update(state)
    def __deepcopy__(self, *args, **kwargs): obj = self.__class__(*args, **kwargs); obj.__dict__.update(__import__("copy").deepcopy(self.__dict__)); return obj
    def __copy__(self, *args, **kwargs): obj = self.__class__(*args, **kwargs); obj.__dict__.update(self.__dict__.copy()); return obj
    def eval(self, string_, func=False, other=False): return string_ if type(string_) is not str else eval((("lambda x, y: " if other else "lambda x:") if not str(string_).startswith("lambda") and func else "") + string_ + (self if False else ''))
    def exec(self, expr: str) -> 'Base': exec(expr); return self  # exec returns None.
    def save(self, path=None, add_suffix=True, save_code=False, verbose=True, data_only=True, desc=""):  # + (".dat" if data_only else "")
        saved_file = Save.pickle(obj=self.__getstate__() if data_only else self, path=path, verbose=verbose, add_suffix=add_suffix, class_name="." + self.__class__.__name__, desc=desc or (f"Data of {self.__class__}" if data_only else desc))
        self.save_code(path=saved_file.parent.joinpath(saved_file.name + "_saved_code.py")) if save_code else None; return self
    @classmethod
    def from_saved_data(cls, path, *args, **kwargs): obj = cls(*args, **kwargs); obj.__setstate__(dict(__import__("dill").loads(Path(path).read_bytes()))); return obj
    def save_code(self, path):
        if hasattr(module := __import__("inspect").getmodule(self), "__file__"): file = Path(module.__file__)
        else: raise FileNotFoundError(f"Attempted to save code from a script running in interactive session! module should be imported instead.")
        Path(path).expanduser().write_text(file.read_text()); return Path(path) if type(path) is str else path  # path could be tb.P, better than Path
    def get_attributes(self, remove_base_attrs=True, return_objects=False, fields=True, methods=True):
        attrs = List(dir(self)).filter(lambda x: '__' not in x and not x.startswith('_')).remove(values=Base().get_attributes(remove_base_attrs=False)if remove_base_attrs else []); import inspect
        attrs = attrs.filter(lambda x: (inspect.ismethod(getattr(self, x)) if not fields else True) and ((not inspect.ismethod(getattr(self, x))) if not methods else True))  # logic (questionable): anything that is not a method is a field
        return List([getattr(self, x) for x in attrs]) if return_objects else List(attrs)
    @staticmethod
    def print(self, dtype=False, attrs=False, **kwargs): return Struct(self.__dict__).update(attrs=self.get_attributes() if attrs else None).print(dtype=dtype, **kwargs)
    @staticmethod
    def get_state(obj, repr_func=lambda x: x, exclude=None) -> dict: return repr_func(obj) if not any([hasattr(obj, "__getstate__"), hasattr(obj, "__dict__")]) else (tmp if type(tmp := obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__) is not dict else Struct(tmp).filter(lambda k, v: k not in (exclude or [])).apply2values(lambda k, v: Base.get_state(v, exclude=exclude, repr_func=repr_func)).__dict__)
    def viz_composition_heirarchy(self, depth=3, obj=None, filt=None):
        install_n_import("objgraph").show_refs([self] if obj is None else [obj], max_depth=depth, filename=str(filename := Path(__import__("tempfile").gettempdir()).joinpath("graph_viz_" + randstr() + ".png")), filter=filt)
        __import__("os").startfile(str(filename.absolute())) if __import__("sys").platform == "win32" else None; return filename


class List(Base):  # Inheriting from Base gives save method.  # Use this class to keep items of the same type."""
    def __init__(self, obj_list=None): super().__init__(); self.list = list(obj_list) if obj_list is not None else []
    def save_items(self, directory, names=None, saver=None): [(saver or Save.pickle)(path=directory / name, obj=item) for name, item in zip(names or range(len(self)), self.list)]
    def __repr__(self): return f"List [{len(self.list)} elements]. First Item: " + f"{get_repr(self.list[0], justify=0, limit=100)}" if len(self.list) > 0 else f"An Empty List []"
    def print(self, sep='\n', styler=repr, return_str=False, **kwargs): res = sep.join([f"{idx:2}- {styler(item)}" for idx, item in enumerate(self.list)]); print(res) if not return_str else None; return res if return_str else None
    def __deepcopy__(self): return List([__import__("copy").deepcopy(i) for i in self.list])
    def __bool__(self): return bool(self.list)
    def __contains__(self, key): return key in self.list
    def __copy__(self) -> 'List': return List(self.list.copy())
    def __getstate__(self): return self.list
    def __setstate__(self, state): self.list = state
    def __len__(self): return len(self.list)
    def __iter__(self): return iter(self.list)
    def __array__(self): import numpy as np; return np.array(self.list)  # compatibility with numpy
    len = property(lambda self: self.list.__len__())
    # ================= call methods =====================================
    def __getattr__(self, name) -> 'List': return List(getattr(i, name) for i in self.list)  # fallback position when __getattribute__ mechanism fails.
    def __call__(self, *args, **kwargs) -> 'List': return List(i(*args, **kwargs) for i in self.list)
    # ======================== Access Methods ==========================================
    def __setitem__(self, key, value): self.list[key] = value
    def sample(self, size=1, replace=False, p=None) -> 'List': return self[list(__import__("numpy").random.choice(len(self), size, replace=replace, p=p))]
    def split(self, every=1, to=None) -> 'List': every = every if to is None else __import__("math").ceil(len(self) / to); return List([(self[ix:ix+every] if ix+every < len(self) else self[ix:len(self)]) for ix in range(0, len(self), every)])
    def filter(self, func, which=lambda idx, x: x) -> 'List': self.eval(func, func=True); return List([which(idx, x) for idx, x in enumerate(self.list) if func(x)])
    # ======================= Modify Methods ===============================
    def reduce(self, func=lambda x, y: x+y, default=None) -> 'List': args = (self.eval(func, func=True, other=True), self.list) + ((default,) if default is not None else ()); return __import__("functools").reduce(*args)
    def append(self, item) -> 'List': self.list.append(item); return self
    def __add__(self, other) -> 'List': return List(self.list + list(other))  # implement coersion
    def __radd__(self, other) -> 'List': return List(list(other) + self.list)
    def __iadd__(self, other) -> 'List': self.list = self.list + list(other); return self  # inplace add.
    def sort(self, key=None, reverse=False) -> 'List': self.list.sort(key=key, reverse=reverse); return self
    def sorted(self, *args, **kwargs) -> 'List': return List(sorted(self.list, *args, **kwargs))
    def insert(self, __index: int, __object): self.list.insert(__index, __object); return self
    def modify(self, expr: str, other=None) -> 'List': [exec(expr) for idx, x in enumerate(self.list)] if other is None else [exec(expr) for idx, (x, y) in enumerate(zip(self.list, other))]; return self
    def remove(self, value=None, values=None, strict=True) -> 'List': [self.list.remove(a_val) for a_val in ((values or []) + ([value] if value else [])) if strict or value in self.list]; return self
    def to_series(self): return __import__("pandas").Series(self.list)
    def to_list(self) -> list: return self.list
    def to_numpy(self, **kwargs): import numpy as np; return np.array(self.list, **kwargs)
    np = property(lambda self: self.to_numpy())
    def to_struct(self, key_val=None) -> 'Struct': return Struct.from_keys_values_pairs(self.apply(self.eval(key_val, func=True) if key_val else lambda x: (str(x), x)))
    def __getitem__(self, key: str or list or slice):
        if type(key) is list: return List(self[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
        elif type(key) is str: return List(item[key] for item in self.list)  # access keys like dictionaries.
        return self.list[key] if type(key) is not slice else List(self.list[key])  # must be an integer or slice: behaves similarly to Numpy A[1] vs A[1:2]
    def apply(self, func, *args, other=None, filt=lambda x: True, jobs=None, prefer=[None, 'processes', 'threads'][0], depth=1, verbose=False, desc=None, **kwargs) -> 'List':
        if depth > 1: self.apply(lambda x: x.apply(func, *args, other=other, jobs=jobs, depth=depth-1, **kwargs)); func = self.eval(func, func=True, other=bool(other))
        iterator = (self.list if not verbose else install_n_import("tqdm").tqdm(self.list, desc=desc)) if other is None else (zip(self.list, other) if not verbose else install_n_import("tqdm").tqdm(zip(self.list, other), desc=desc))
        if jobs: from joblib import Parallel, delayed; return List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, *args, **kwargs) for x in iterator)) if other is None else List(Parallel(n_jobs=jobs, prefer=prefer)(delayed(func)(x, y) for x, y in iterator))
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
    def save_json(self, path=None, indent=None): return Save.json(obj=self.__dict__, path=path, indent=indent)
    from_keys_values = classmethod(lambda cls, k, v: cls(dict(zip(k, v))))
    from_keys_values_pairs = classmethod(lambda cls, my_list: cls({k: v for k, v in my_list}))
    @classmethod
    def from_names(cls, names, default_=None) -> 'Struct': return cls.from_keys_values(k=names, v=default_ or [None] * len(names))  # Mimick NamedTuple and defaultdict
    def spawn_from_values(self, values) -> 'Struct': return self.from_keys_values(self.keys(), self.eval(values, func=False))
    def spawn_from_keys(self, keys) -> 'Struct': return self.from_keys_values(self.eval(keys, func=False), self.values())
    def to_default(self, default=lambda: None): tmp2 = __import__("collections").defaultdict(default); tmp2.update(self.__dict__); self.__dict__ = tmp2; return self
    def __str__(self, sep="\n"): return config(self.__dict__, sep=sep)
    def __getattr__(self, item) -> 'Struct':
        try: return self.__dict__[item]
        except KeyError: raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')  # this works better with the linter. replacing Key error with Attribute error makes class work nicely with hasattr() by returning False.
    clean_view = property(lambda self: type("TempClass", (object,), self.__dict__))
    def __repr__(self, limit=150): return "Struct: " + Display.get_repr(self.keys().list.__repr__(), limit=limit, justify=0)
    def __getitem__(self, item): return self.__dict__[item]  # thus, gives both dot notation and string access to elements.
    def __setitem__(self, key, value): self.__dict__[key] = value
    def __bool__(self): return bool(self.__dict__)
    def __contains__(self, key): return key in self.__dict__
    def __len__(self): return len(self.keys())
    def __getstate__(self): return self.__dict__  # serialization
    def __setstate__(self, state): self.__dict__ = state
    def __iter__(self): return iter(self.__dict__.items())
    def __delitem__(self, key): del self.__dict__[key]
    def copy(self) -> 'Struct': return Struct(self.__dict__.copy())
    def to_dataframe(self, *args, **kwargs): return __import__("pandas").DataFrame(self.__dict__, *args, **kwargs)
    def keys(self, verbose=False) -> 'List': return List(list(self.__dict__.keys())) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.keys())
    def values(self, verbose=False) -> 'List': return List(list(self.__dict__.values())) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.values())
    def items(self, verbose=False, desc="") -> 'List': return List(self.__dict__.items()) if not verbose else install_n_import("tqdm").tqdm(self.__dict__.items(), desc=desc)
    def get(self, key=None, default=None, strict=False, keys=None) -> 'List': return List([self.__dict__.get(key, default) if not strict else self[key] for key in (keys if keys is not None else [])]) if keys is not None else (self.__dict__.get(key, default) if not strict else self[key])
    def apply2keys(self, kv_func, verbose=False, desc="") -> 'Struct': return Struct({kv_func(key, val): val for key, val in self.items(verbose=verbose, desc=desc)})
    def apply2values(self, kv_func, verbose=False, desc="") -> 'Struct': [self.__setitem__(key, kv_func(key, val)) for key, val in self.items(verbose=verbose, desc=desc)]; return self
    def apply(self, kv_func) -> 'List': return self.items().apply(lambda item: kv_func(item[0], item[1]))
    def filter(self, kv_func=None) -> 'Struct': return Struct({key: self[key] for key, val in self.items() if kv_func(key, val)})
    def inverse(self) -> 'Struct': return Struct({v: k for k, v in self.__dict__.items()})
    def update(self, *args, **kwargs) -> 'Struct': self.__dict__.update(Struct(*args, **kwargs).__dict__); return self
    def delete(self, key=None, keys=None, kv_func=None) -> 'Struct': [self.__dict__.__delitem__(key) for key in ([key] if key else [] + (keys if keys is not None else []))]; [self.__dict__.__delitem__(k) for k, v in self.items() if kv_func(k, v)] if kv_func is not None else None; return self
    def _pandas_repr(self, justify, return_str=False, limit=30): res = __import__("pandas").DataFrame(__import__("numpy").array([self.keys(), self.values().apply(lambda x: str(type(x)).split("'")[1]), self.values().apply(lambda x: get_repr(x, justify=justify, limit=limit).replace("\n", " "))]).T, columns=["key", "dtype", "details"]); return res if not return_str else str(res)
    def print(self, dtype=True, return_str=False, justify=30, as_config=False, as_yaml=False, limit=50, title="", **kwargs):
        if as_config and not return_str: install_n_import("rich").inspect(self, value=False, title=title, docs=False, sort=False); return self
        res = f"Empty Struct." if not bool(self) else ((__import__("yaml").dump(self.__dict__) if as_yaml else config(self.__dict__, justify=justify, **kwargs)) if as_yaml or as_config else self._pandas_repr(justify=justify, return_str=False, limit=limit).drop(columns=[] if dtype else ["dtype"]))
        (install_n_import("rich").print(res.to_markdown()) if "DataFrame" in res.__class__.__name__ else print(res)) if not return_str else None; return str(res) if return_str else self
    @staticmethod
    def concat_values(*dicts, orient='list') -> 'Struct': return Struct(__import__("pandas").concat(List(dicts).apply(lambda x: Struct(x).to_dataframe())).to_dict(orient=orient))
    def plot(self, use_plt=True, title='', xlabel='', ylabel='', **kwargs):
        if not use_plt: fig = __import__("crocodile.plotly_management").px.line(self.__dict__); fig.show(); return fig
        else: artist = __import__("crocodile").matplotlib_management.Artist(figname='Structure Plot', **kwargs); artist.plot_dict(self.__dict__, title=title, xlabel=xlabel, ylabel=ylabel); return artist

def set_pandas_display(rows=1000, columns=1000, width=5000, colwidth=40) -> None: import pandas as pd; pd.set_option('display.max_colwidth', colwidth); pd.set_option('display.max_columns', columns); pd.set_option('display.width', width); pd.set_option('display.max_rows', rows)
def set_pandas_auto_width(): __import__("pandas").set_option('width', 0)  # this way, pandas is told to detect window length and act appropriately.  For fixed width host windows, this is recommended to avoid chaos due to line-wrapping.
def set_numpy_display(precision=3, linewidth=250, suppress=True, floatmode='fixed', **kwargs) -> None: __import__("numpy").set_printoptions(precision=precision, suppress=suppress, linewidth=linewidth, floatmode=floatmode, **kwargs)
def config(mydict, sep="\n", justify=15, quotes=False): return sep.join([f"{key:>{justify}} = {repr(val) if quotes else val}" for key, val in mydict.items()])
def f(str_, limit=float('inf'), justify=50, direc="<") -> str: return f"{(str_[:limit - 4] + '... ' if len(str_) > limit else str_):{direc}{justify}}"
def eng(): __import__("pandas").set_eng_float_format(accuracy=3, use_eng_prefix=True); __import__("pandas").options.float_format = '{:, .5f}'.format; __import__("pandas").set_option('precision', 7)  # __import__("pandas").set_printoptions(formatter={'float': '{: 0.3f}'.format})
def outline(array, name="Array", printit=True): str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"; print(str_) if printit else None; return str_
def get_repr(data, justify=15, limit=float('inf'), direc="<") -> str:
    if (dtype := data.__class__.__name__) in {'list', 'str'}: str_ = data if dtype == 'str' else f"list. length = {len(data)}. " + ("1st item type: " + str(type(data[0])).split("'")[1]) if len(data) > 0 else " "
    elif dtype in {"DataFrame", "Series"}: str_ = f"Pandas DF: shape = {data.shape}, dtype = {data.dtypes}." if dtype == 'DataFrame' else f"Pandas Series: Length = {len(data)}, Keys = {get_repr(data.keys().to_list())}."
    else: str_ = f"shape = {data.shape}, dtype = {data.dtype}." if dtype == 'ndarray' else repr(data)
    return f(str_.replace("\n", ", "), justify=justify, limit=limit, direc=direc)
def print_string_list(mylist, char_per_row=125, sep=" ", style=str, _counter=0):
    for item in mylist: print("") if (_counter + len(style(item))) // char_per_row > 0 else print(style(item), end=sep); _counter = len(style(item)) if (_counter + len(style(item))) // char_per_row > 0 else _counter + len(style(item))
class Display: set_pandas_display = set_pandas_display; set_pandas_auto_width = set_pandas_auto_width; set_numpy_display = set_numpy_display; config = config; f = f; eng = eng; outline = outline; get_repr = get_repr; print_string_list = print_string_list  # or D = type('D', (object, ), dict(set_pandas_display


if __name__ == '__main__':
    pass
