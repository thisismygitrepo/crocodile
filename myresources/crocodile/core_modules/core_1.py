

from pathlib import Path
from typing import Optional, Union, TypeVar, Any, Callable, Protocol, ParamSpec

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
    print(f"🔥 Running the script in the ve `{ve_name}`".center(75, "="))
    subprocess.run(fire_script, shell=True, check=True, executable='/bin/bash')
    return fire_script


def save_decorator(ext: str):  # apply default paths, add extension to path, print the saved file path
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
                def f(str_: str, limit: int = 10000000000, justify: int = 50, direc: str = "<") -> str:
                    return f"{(str_[:limit - 4] + '... ' if len(str_) > limit else str_):{direc}{justify}}"
                try: print(f"💽 SAVED {desc or path.name} {obj.__class__.__name__}: {f(repr(obj), justify=0, limit=75)}\n💽 SAVED {path.stat().st_size / 1024**2:0.2f} MB @ `{path.absolute().as_uri()}`.")  # |  Directory: `{path.parent.absolute().as_uri()}`
                except UnicodeEncodeError as err: print(f"crocodile.core: Warning: UnicodeEncodeError: {err}")
            return path
        return wrapper
    return decorator


class Save:
    @staticmethod
    @save_decorator(".parquet")
    def parquet(obj: Any, path: PLike):
        obj.to_parquet(path, index=False)
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
    def toml(obj: dict[Any, Any], path: PLike, encoding: str = 'utf-8'):
        return Path(path).write_text(install_n_import("toml").dumps(obj), encoding=encoding)
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
