

from crocodile.core import install_n_import
from pathlib import Path
from typing import Any, Optional


class Read:
    @staticmethod
    def read(path: 'PLike', **kwargs: Any) -> Any:
        if Path(path).is_dir(): raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        suffix = Path(path).suffix[1:]
        if suffix == "": raise ValueError(f"File type could not be inferred from suffix. Suffix is empty. Path: {path}")
        if suffix == "sqlite":
            from crocodile.database import DBMS
            res = DBMS.from_local_db(path=path)
            print(res.describe_db())
            return res
        try: return getattr(Read, suffix)(str(path), **kwargs)
        except AttributeError as err:
            if "type object 'Read' has no attribute" not in str(err): raise AttributeError(err) from err
            if suffix in ('eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'):
                import matplotlib.pyplot as pyplot
                return pyplot.imread(path, **kwargs)  # from: plt.gcf().canvas.get_supported_filetypes().keys():
            if suffix == "parquet":
                import pandas as pd
                return pd.read_parquet(path, **kwargs)
            elif suffix == "csv":
                import pandas as pd
                return pd.read_csv(path, **kwargs)
            try:
                guess = install_n_import('magic', 'python-magic').from_file(path)
                raise AttributeError(f"Unknown file type. failed to recognize the suffix `{suffix}`. According to libmagic1, the file seems to be: {guess}") from err
            except ImportError as err2:
                print(f"💥 Unknown file type. failed to recognize the suffix `{suffix}` of file {path} ")
                raise ImportError(err) from err2
    @staticmethod
    def json(path: 'PLike', r: bool = False, **kwargs: Any) -> Any:  # return could be list or dict etc
        import json
        try:
            mydict = json.loads(Path(path).read_text(encoding='utf-8'), **kwargs)
        except Exception:
            import pyjson5
            mydict = pyjson5.loads(Path(path).read_text(encoding='utf-8'), **kwargs)  # file has C-style comments.
        _ = r
        return mydict
    @staticmethod
    def yaml(path: 'PLike', r: bool = False) -> Any:  # return could be list or dict etc
        import yaml  # type: ignore
        with open(str(path), "r", encoding="utf-8") as file:
            mydict = yaml.load(file, Loader=yaml.FullLoader)
        _ = r
        return mydict
    @staticmethod
    def ini(path: 'PLike', encoding: Optional[str] = None):
        if not Path(path).exists() or Path(path).is_dir(): raise FileNotFoundError(f"File not found or is a directory: {path}")
        import configparser
        res = configparser.ConfigParser()
        res.read(filenames=[str(path)], encoding=encoding)
        return res
    @staticmethod
    def toml(path: 'PLike'):
        import tomli
        return tomli.loads(Path(path).read_text(encoding='utf-8'))
    @staticmethod
    def npy(path: 'PLike', **kwargs: Any):
        import numpy as np
        data = np.load(str(path), allow_pickle=True, **kwargs)
        # data = data.item() if data.dtype == np.object else data
        return data
    @staticmethod
    def pickle(path: 'PLike', **kwargs: Any):
        import pickle
        try: return pickle.loads(Path(path).read_bytes(), **kwargs)
        except BaseException as ex:
            print(f"💥 Failed to load pickle file `{path}` with error:\n{ex}")
            raise ex
    @staticmethod
    def pkl(path: 'PLike', **kwargs: Any): return Read.pickle(path, **kwargs)
    @staticmethod
    def dill(path: 'PLike', **kwargs: Any) -> Any:
        """handles imports automatically provided that saved object was from an imported class (not in defined in __main__)"""
        import dill
        obj = dill.loads(str=Path(path).read_bytes(), **kwargs)
        return obj
    @staticmethod
    def py(path: 'PLike', init_globals: Optional[dict[str, Any]] = None, run_name: Optional[str] = None):
        import runpy
        return runpy.run_path(str(path), init_globals=init_globals, run_name=run_name)
    @staticmethod
    def txt(path: 'PLike', encoding: str = 'utf-8') -> str: return Path(path).read_text(encoding=encoding)
    @staticmethod
    def parquet(path: 'PLike', **kwargs: Any):
        import pandas as pd
        return pd.read_parquet(path, **kwargs)


if __name__ == '__main__':
    from crocodile.file_management_helpers.file4 import PLike
