
"""
File
"""

# %%

from crocodile.core import List, timestamp, randstr, validate_name, str2timedelta, Save, Path, install_n_import
from datetime import datetime, timedelta
from typing import Any, Optional, Union, Callable, TypeVar, TypeAlias, Literal, NoReturn, Protocol


OPLike: TypeAlias = Union[str, 'P', Path, None]
PLike: TypeAlias = Union[str, 'P', Path]
FILE_MODE: TypeAlias = Literal['r', 'w', 'x', 'a']
SHUTIL_FORMATS: TypeAlias = Literal["zip", "tar", "gztar", "bztar", "xztar"]


# %% =============================== Security ================================================
def obscure(msg: bytes) -> bytes: import base64; import zlib; return base64.urlsafe_b64encode(zlib.compress(msg, 9))
def unobscure(obscured: bytes) -> bytes: return __import__("zlib").decompress(__import__("base64").urlsafe_b64decode(obscured))
def hashpwd(password: str): import bcrypt; return bcrypt.hashpw(password=password.encode(), salt=bcrypt.gensalt()).decode()
def pwd2key(password: str, salt: Optional[bytes] = None, iterations: int = 10) -> bytes:  # Derive a secret key from a given password and salt"""
    import base64
    if salt is None:
        import hashlib
        m = hashlib.sha256()
        m.update(password.encode("utf-8"))
        return base64.urlsafe_b64encode(m.digest())  # make url-safe bytes required by Ferent.
    from cryptography.hazmat.primitives import hashes; from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return base64.urlsafe_b64encode(PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=None).derive(password.encode()))
def encrypt(msg: bytes, key: Optional[bytes] = None, pwd: Optional[str] = None, salted: bool = True, iteration: Optional[int] = None, gen_key: bool = False) -> bytes:
    import base64; from cryptography.fernet import Fernet
    salt, iteration = None, None
    if pwd is not None:  # generate it from password
        assert (key is None) and (type(pwd) is str), f"❌ You can either pass key or pwd, or none of them, but not both."
        import secrets
        iteration = iteration or secrets.randbelow(1_000_000)
        salt = secrets.token_bytes(16) if salted else None
        key_resolved = pwd2key(pwd, salt, iteration)
    elif key is None:
        if gen_key:
            key_resolved = Fernet.generate_key()
            P.home().joinpath('dotfiles/creds/data/encrypted_files_key.bytes').write_bytes(key_resolved, overwrite=False)
        else:
            try: key_resolved = P.home().joinpath("dotfiles/creds/data/encrypted_files_key.bytes").read_bytes(); print(f"⚠️ Using key from: {P.home().joinpath('dotfiles/creds/data/encrypted_files_key.bytes')}")
            except FileNotFoundError as err:
                print("\n" * 3, "~" * 50, f"""Consider Loading up your dotfiles or pass `gen_key=True` to make and save one.""", "~" * 50, "\n" * 3)
                raise FileNotFoundError(err) from err
    elif isinstance(key, (str, P, Path)): key_resolved = P(key).read_bytes()  # a path to a key file was passed, read it:
    elif type(key) is bytes: key_resolved = key  # key passed explicitly
    else: raise TypeError(f"❌ Key must be either a path, bytes object or None.")
    code = Fernet(key=key_resolved).encrypt(msg)
    if pwd is not None and salt is not None and iteration is not None: return base64.urlsafe_b64encode(b'%b%b%b' % (salt, iteration.to_bytes(4, 'big'), base64.urlsafe_b64decode(code)))
    return code
def decrypt(token: bytes, key: Optional[bytes] = None, pwd: Optional[str] = None, salted: bool = True) -> bytes:
    if pwd is not None:
        assert key is None, f"❌ You can either pass key or pwd, or none of them, but not both."
        if salted:
            decoded = __import__("base64").urlsafe_b64decode(token); salt, iterations, token = decoded[:16], decoded[16:20], __import__("base64").urlsafe_b64encode(decoded[20:])
            key_resolved = pwd2key(pwd, salt, int.from_bytes(iterations, 'big'))
        else: key_resolved = pwd2key(pwd)  # trailing `;` prevents IPython from caching the result.
    if type(key) is bytes: key_resolved = key  # passsed explicitly
    elif key is None: key_resolved = P.home().joinpath("dotfiles/creds/data/encrypted_files_key.bytes").read_bytes()  # read from file
    elif isinstance(key, (str, P, Path)): key_resolved = P(key).read_bytes()  # passed a path to a file containing kwy
    else: raise TypeError(f"❌ Key must be either str, P, Path, bytes or None. Recieved: {type(key)}")
    from cryptography.fernet import Fernet
    return Fernet(key=key_resolved).decrypt(token)
def unlock(drive: str = "D:", pwd: Optional[str] = None, auto_unlock: bool = False):
    from crocodile.meta import Terminal
    return Terminal().run(f"""$SecureString = ConvertTo-SecureString "{pwd or P.home().joinpath("dotfiles/creds/data/bitlocker_pwd").read_text()}" -AsPlainText -Force; Unlock-BitLocker -MountPoint "{drive}" -Password $SecureString; """ + (f'Enable-BitLockerAutoUnlock -MountPoint "{drive}"' if auto_unlock else ''), shell="powershell")


# %% =================================== File ============================================
# def mat(path, remove_meta=False, **kwargs): res = Struct(__import__("scipy.io").__dict__["io"].loadmat(path, **kwargs)); List(res.keys()).filter("x.startswith('__')").apply(lambda x: res.__delattr__(x)) if remove_meta else None; return res

class Read:
    @staticmethod
    def read(path: PLike, **kwargs: Any):
        suffix = Path(path).suffix[1:]
        if suffix == "": raise ValueError(f"File type could not be inferred from suffix. Suffix is empty. Path: {path}")
        if suffix == "sqlite":
            from crocodile.database import DBMS
            return DBMS.from_local_db(path=path)
        try: return getattr(Read, suffix)(str(path), **kwargs)
        except AttributeError as err:
            if "type object 'Read' has no attribute" not in str(err): raise AttributeError(err) from err
            if suffix in ('eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'): return __import__("matplotlib").pyplot.imread(path, **kwargs)  # from: plt.gcf().canvas.get_supported_filetypes().keys():
            if suffix == ".parquet": return __import__("pandas").read_parquet(path, **kwargs)
            try: raise AttributeError(f"Unknown file type. failed to recognize the suffix `{suffix}`. According to libmagic1, the file seems to be: {install_n_import('magic', 'python-magic').from_file(path)}") from err
            except ImportError as err2: print(f"💥 Unknown file type. failed to recognize the suffix `{suffix}` of file {path} "); raise ImportError(err) from err2
    @staticmethod
    def json(path: PLike, r: bool = False, **kwargs: Any) -> Any:  # return could be list or dict etc
        try: mydict = __import__("json").loads(P(path).read_text(), **kwargs)
        except Exception: mydict = install_n_import("pyjson5").loads(P(path).read_text(), **kwargs)  # file has C-style comments.
        _ = r
        return mydict
    @staticmethod
    def yaml(path: PLike, r: bool = False) -> Any:  # return could be list or dict etc
        with open(str(path), "r", encoding="utf-8") as file: mydict = __import__("yaml").load(file, Loader=__import__("yaml").FullLoader)
        _ = r
        return mydict
    @staticmethod
    def ini(path: PLike):
        if not Path(path).exists() or Path(path).is_dir(): raise FileNotFoundError(f"File not found: {path}")
        import configparser; res = configparser.ConfigParser(); res.read(filenames=[str(path)]); return res
    @staticmethod
    def toml(path: PLike): return install_n_import("tomli").loads(P(path).read_text())

    @staticmethod
    def npy(path: PLike, **kwargs: Any):
        import numpy as np
        data = np.load(str(path), allow_pickle=True, **kwargs)
        # data = data.item() if data.dtype == np.object else data
        return data
    @staticmethod
    def csv(path: PLike, **kwargs: Any): return __import__("pandas").read_csv(path, **kwargs)

    @staticmethod
    def vanilla_pickle(path: PLike, **kwargs: Any): return __import__("pickle").loads(P(path).read_bytes(), **kwargs)
    pkl = staticmethod(vanilla_pickle)
    pickle = staticmethod(vanilla_pickle)
    @staticmethod
    def pickles(bytes_obj: bytes): return __import__("dill").loads(bytes_obj)  # handles imports automatically provided that saved object was from an imported class (not in defined in __main__)
    @staticmethod
    def dill(path: PLike, **kwargs: Any) -> Any:
        import dill
        obj = dill.loads(str=P(path).read_bytes(), **kwargs)
        return obj

    @staticmethod
    def py(path: PLike, init_globals: Optional[dict[str, Any]] = None, run_name: Optional[str] = None): return __import__("runpy").run_path(path, init_globals=init_globals, run_name=run_name)
    @staticmethod
    def txt(path: PLike, encoding: str = 'utf-8') -> str: return P(path).read_text(encoding=encoding)


def modify_text(txt_raw: str, txt_search: str, txt_alt: Union[str, Callable[[str], str]], replace_line: bool = True, notfound_append: bool = False, prepend: bool = False, strict: bool = False):
    lines, bingo = txt_raw.split("\n"), False
    if not replace_line:  # no need for line splitting
        assert isinstance(txt_alt, str), f"txt_alt must be a string if notfound_append is True. It is not: {txt_alt}"
        if txt_search in txt_raw: return txt_raw.replace(txt_search, txt_alt)
        return txt_raw + "\n" + txt_alt if notfound_append else txt_raw
    for idx, line in enumerate(lines):
        if txt_search in line:
            if isinstance(txt_alt, str): lines[idx] = txt_alt
            elif callable(txt_alt): lines[idx] = txt_alt(line)
            bingo = True
    if strict and not bingo: raise ValueError(f"txt_search `{txt_search}` not found in txt_raw `{txt_raw}`")
    if bingo is False and notfound_append is True:
        assert isinstance(txt_alt, str), f"txt_alt must be a string if notfound_append is True. It is not: {txt_alt}"
        if prepend: lines.insert(0, txt_alt)
        else: lines.append(txt_alt)  # txt not found, add it anyway.
    return "\n".join(lines)


class P(type(Path()), Path):  # type: ignore # pylint: disable=E0241
    # ============= Path management ==================
    """ The default behaviour of methods acting on underlying disk object is to perform the action and return a new path referring to the mutated object in disk drive.
    However, there is a flag `orig` that makes the function return orignal path object `self` as opposed to the new one pointing to new object.
    Additionally, the fate of the original object can be decided by a flag `inplace` which means `replace` it defaults to False and in essence, it deletes the original underlying object.
    This can be seen in `zip` and `encrypt` but not in `copy`, `move`, `retitle` because the fate of original file is dictated already.
    Furthermore, those methods are accompanied with print statement explaining what happened to the object."""
    def delete(self, sure: bool = False, verbose: bool = True) -> 'P':  # slf = self.expanduser().resolve() don't resolve symlinks.
        if not sure: _ = print(f"❌ Did NOT DELETE because user is not sure. file: {repr(self)}.") if verbose else None; return self
        if not self.exists(): self.unlink(missing_ok=True); _ = print(f"❌ Could NOT DELETE nonexisting file {repr(self)}. ") if verbose else None; return self  # broken symlinks exhibit funny existence behaviour, catch them here.
        _ = self.unlink(missing_ok=True) if self.is_file() or self.is_symlink() else __import__("shutil").rmtree(self, ignore_errors=False); _ = print(f"🗑️ ❌ DELETED {repr(self)}.") if verbose else None; return self
    def send2trash(self, verbose: bool = True) -> 'P':
        if self.exists():
            install_n_import(library="send2trash").send2trash(self.resolve().str)
            _ = print(f"🗑️ TRASHED {repr(self)}") if verbose else None; return self  # do not expand user symlinks.
        elif verbose: print(f"💥 Could NOT trash {self}"); return self
        return self
    def move(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, rel2it: bool = False, overwrite: bool = False, verbose: bool = True, parents: bool = True, content: bool = False) -> 'P':
        path = self._resolve_path(folder=folder, name=name, path=path, default_name=self.absolute().name, rel2it=rel2it)
        _ = path.parent.create(parents=True, exist_ok=True) if parents else None; slf = self.expanduser().resolve()
        if content:
            assert self.is_dir(), NotADirectoryError(f"💥 When `content` flag is set to True, path must be a directory. It is not: `{repr(self)}`")
            self.search("*").apply(lambda x: x.move(folder=path.parent, content=False, overwrite=overwrite)); return path  # contents live within this directory.
        if overwrite: tmp_path = slf.rename(path.parent.absolute() / randstr()); path.delete(sure=True, verbose=verbose); tmp_path.rename(path)  # works if moving a path up and parent has same name
        else: slf.rename(path)  # self._return(res=path, inplace=True, operation='rename', orig=False, verbose=verbose, strict=True, msg='')
        _ = print(f"🚚 MOVED {repr(self)} ==> {repr(path)}`") if verbose else None; return path
    def copy(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, content: bool = False, verbose: bool = True, append: Optional[str] = None, overwrite: bool = False, orig: bool = False) -> 'P':  # tested %100  # TODO: replace `content` flag with ability to interpret "*" in resolve method.
        dest = self._resolve_path(folder=folder, name=name, path=path, default_name=self.name, rel2it=False)
        dest, slf = dest.expanduser().resolve().create(parents_only=True), self.expanduser().resolve()
        dest = self.append(append if append is not None else f"_copy_{randstr()}") if dest == slf else dest
        _ = dest.delete(sure=True) if not content and overwrite and dest.exists() else None
        if not content and not overwrite and dest.exists(): raise FileExistsError(f"Destination already exists: {repr(dest)}")
        if slf.is_file(): __import__("shutil").copy(str(slf), str(dest)); _ = print(f"🖨️ COPIED {repr(slf)} ==> {repr(dest)}") if verbose else None
        elif slf.is_dir(): dest = dest.parent if content else dest; __import__("distutils.dir_util").__dict__["dir_util"].copy_tree(str(slf), str(dest)); _ = print(f"COPIED {'Content of ' if content else ''} {repr(slf)} ==> {repr(dest)}") if verbose else None
        else: print(f"💥 Could NOT COPY. Not a file nor a path: {repr(slf)}.")
        return dest if not orig else self
    # ======================================= File Editing / Reading ===================================
    def readit(self, reader: Optional[Callable[[PLike], Any]] = None, strict: bool = True, default: Optional[Any] = None, verbose: bool = False, **kwargs: Any) -> 'Any':
        if not (slf := self.expanduser().resolve()).exists():
            if strict: raise FileNotFoundError(f"`{slf}` is no where to be found!")
            else: _ = (print(f"💥 tb.P.readit warning: FileNotFoundError, skipping reading of file `{self}") if verbose else None); return default
        if verbose: print(f"Reading {slf} ({slf.size()} MB) ...")
        filename = slf.unzip(folder=slf.tmp(folder="tmp_unzipped"), verbose=verbose) if '.zip' in str(slf) else slf
        try: return Read.read(filename, **kwargs) if reader is None else reader(str(filename), **kwargs)
        except IOError as ioe: raise IOError from ioe
    def start(self, opener: Optional[str] = None):
        if str(self).startswith("http") or str(self).startswith("www"): __import__("webbrowser").open(str(self)); return self
        if __import__("sys").platform == "win32":  # double quotes fail with cmd. # __import__("os").startfile(filename)  # works for files and folders alike, but if opener is given, e.g. opener="start"
            __import__("subprocess").Popen(f"powershell start '{self.expanduser().resolve().str}'" if opener is None else rf'powershell {opener} \'{self}\''); return self  # fails for folders. Start must be passed, but is not defined.
        elif __import__("sys").platform == 'linux': __import__("subprocess").call(["xdg-open", self.expanduser().resolve().str]); return self  # works for files and folders alike
        else: __import__("subprocess").call(["open", self.expanduser().resolve().str]); return self  # works for files and folders alike  # mac
    def __call__(self, *args: Any, **kwargs: Any) -> 'P': self.start(*args, **kwargs); return self
    # def append_text(self, appendix: str) -> 'P': self.write_text(self.read_text() + appendix); return self
    def modify_text(self, txt_search: str, txt_alt: str, replace_line: bool = False, notfound_append: bool = False, prepend: bool = False, encoding: str = 'utf-8'):
        if not self.exists(): self.create(parents_only=True).write_text(txt_search)
        return self.write_text(modify_text(txt_raw=self.read_text(encoding=encoding), txt_search=txt_search, txt_alt=txt_alt, replace_line=replace_line, notfound_append=notfound_append, prepend=prepend), encoding=encoding)
    def download_to_memory(self, allow_redirects: bool = True, timeout: Optional[int] = None, params: Any = None) -> 'Any':
        import requests
        return requests.get(self.as_url_str(), allow_redirects=allow_redirects, timeout=timeout, params=params)  # Alternative: from urllib import request; request.urlopen(url).read().decode('utf-8').
    def download(self, folder: OPLike = None, name: OPLike = None, allow_redirects: bool = True, timeout: Optional[int] = None, params: Any = None) -> 'P':
        import requests
        response = requests.get(self.as_url_str(), allow_redirects=allow_redirects, timeout=timeout, params=params)  # Alternative: from urllib import request; request.urlopen(url).read().decode('utf-8').
        # if memory: return response  # r.contents is bytes encoded as per docs of requests.
        if name is not None: f_name = name
        else: f_name = validate_name(str(P(response.history[-1].url).name if len(response.history) > 0 else P(response.url).name))
        return (P.home().joinpath("Downloads") if folder is None else P(folder)).joinpath(f_name).create(parents_only=True).write_bytes(response.content)
    def _return(self, res: 'P', inlieu: bool = False, inplace: bool = False, operation: Optional[str] = None, overwrite: bool = False, orig: bool = False, verbose: bool = False, strict: bool = True, msg: str = "", __delayed_msg__: str = "") -> 'P':
        if inlieu:
            self._str = str(res)  # type: ignore # pylint: disable=W0201
        if inplace:
            assert self.exists(), f"`inplace` flag is only relevant if the path exists. It doesn't {self}"
            if operation == "rename":
                if overwrite and res.exists(): res.delete(sure=True, verbose=verbose)
                if not overwrite and res.exists():
                    if strict: raise FileExistsError(f"File {res} already exists.")
                    else: _ = print(f"⚠️ SKIPPED RENAMING {repr(self)} ➡️ {repr(res)} because FileExistsError and scrict=False policy.") if verbose else None; return self if orig else res
                self.rename(res)
                msg = msg or f"RENAMED {repr(self)} ➡️ {repr(res)}"
            elif operation == "delete":
                self.delete(sure=True, verbose=False)
                __delayed_msg__ = f"DELETED 🗑️❌ {repr(self)}."
        if verbose and msg != "":
            try: print(msg)  # emojie print error.
            except UnicodeEncodeError: print(f"tb.P._return warning: UnicodeEncodeError, could not print message.")
        if verbose and __delayed_msg__ != "":
            try: print(__delayed_msg__)
            except UnicodeEncodeError: print(f"tb.P._return warning: UnicodeEncodeError, could not print message.")
        return self if orig else res
    # ================================ Path Object management ===========================================
    """ Distinction between Path object and the underlying file on disk that the path may refer to. Two distinct flags are used:
        `inplace`: the operation on the path object will affect the underlying file on disk if this flag is raised, otherwise the method will only alter the string.
        `inliue`: the method acts on the path object itself instead of creating a new one if this flag is raised.
        `orig`: whether the method returns the original path object or a new one."""
    def prepend(self, prefix: str, suffix: Optional[str] = None, verbose: bool = True, **kwargs: Any):
        """Returns a new path object with the name prepended to the stem of the path."""
        return self._return(self.parent.joinpath(prefix + self.trunk + (suffix or ''.join(('bruh' + self).suffixes))), operation="rename", verbose=verbose, **kwargs)  # Path('.ssh').suffix fails, 'bruh' fixes it.
    def append(self, name: str = '', index: bool = False, suffix: Optional[str] = None, verbose: bool = True, **kwargs: Any) -> 'P':
        """Returns a new path object with the name appended to the stem of the path. If `index` is True, the name will be the index of the path in the parent directory."""
        if index: return self.append(name=f'_{len(self.parent.search(f"*{self.trunk}*"))}', index=False, verbose=verbose, suffix=suffix, **kwargs)
        return self._return(self.parent.joinpath(self.trunk + (name or ("_" + str(timestamp()))) + (suffix or ''.join(('bruh' + self).suffixes))), operation="rename", verbose=verbose, **kwargs)
    def with_trunk(self, name: str, verbose: bool = True, **kwargs: Any): return self._return(self.parent.joinpath(name + "".join(self.suffixes)), operation="rename", verbose=verbose, **kwargs)  # Complementary to `with_stem` and `with_suffix`
    def with_name(self, name: str, verbose: bool = True, inplace: bool = False, **kwargs: Any): assert type(name) is str, "name must be a string."; return self._return(self.parent / name, verbose=verbose, operation="rename", inplace=inplace, **kwargs)
    def switch(self, key: str, val: str, verbose: bool = True, **kwargs: Any): return self._return(P(str(self).replace(key, val)), operation="rename", verbose=verbose, **kwargs)  # Like string replace method, but `replace` is an already defined method."""
    def switch_by_index(self, idx: int, val: str, verbose: bool = True, **kwargs: Any): return self._return(P(*[val if index == idx else value for index, value in enumerate(self.parts)]), operation="rename", verbose=verbose, **kwargs)
    # ============================= attributes of object ======================================
    @property
    def trunk(self) -> str: return self.name.split('.')[0]  # """ useful if you have multiple dots in file path where `.stem` fails."""
    @property
    def len(self) -> int: return self.__len__()
    @property
    def items(self) -> List[str]: return List(self.parts)
    def __len__(self) -> int: return len(self.parts)
    def __contains__(self, item: PLike): return P(item).as_posix() in self.as_posix()
    def __iter__(self): return self.parts.__iter__()
    def __deepcopy__(self, *args: Any, **kwargs: Any) -> 'P': _ = args, kwargs; return P(str(self))
    def __getstate__(self) -> str: return str(self)
    def __setstate__(self, state: str): self._str = str(state)  # pylint: disable=W0201
    def __add__(self, other: PLike) -> 'P':
        return self.parent.joinpath(self.name + str(other))  # used append and prepend if the addition wanted to be before suffix.
    def __radd__(self, other: PLike) -> 'P':
        return self.parent.joinpath(str(other) + self.name)  # other + P and `other` doesn't know how to make this addition.
    def __sub__(self, other: PLike) -> 'P': res = P(str(self).replace(str(other), "")); return (res[1:] if str(res[0]) in {"\\", "/"} else res) if len(res) else res  # paths starting with "/" are problematic. e.g ~ / "/path" doesn't work.
    def rel2cwd(self, inlieu: bool = False) -> 'P': return self._return(P(self.expanduser().absolute().relative_to(Path.cwd())), inlieu)
    def rel2home(self, inlieu: bool = False) -> 'P': return self._return(P(self.expanduser().absolute().relative_to(Path.home())), inlieu)  # very similat to collapseuser but without "~" being added so its consistent with rel2cwd.
    def collapseuser(self, strict: bool = True, placeholder: str = "~") -> 'P':  # opposite of `expanduser` resolve is crucial to fix Windows cases insensitivty problem.
        if strict: assert P.home() in self.expanduser().absolute().resolve(), ValueError(f"`{P.home()}` is not in the subpath of `{self}`")
        if (str(self).startswith(placeholder) or P.home().as_posix() not in self.resolve().as_posix()): return self
        return self._return(P(placeholder) / (self.expanduser().absolute().resolve(strict=strict) - P.home()))  # resolve also solves the problem of Windows case insensitivty.
    def __getitem__(self, slici: Union[int, list[int], slice]):
        if isinstance(slici, list): return P(*[self[item] for item in slici])
        elif isinstance(slici, int): return P(self.parts[slici])
        return P(*self.parts[slici])  # must be a slice
    def __setitem__(self, key: Union['str', int, slice], value: PLike):
        fullparts, new = list(self.parts), list(P(value).parts)
        if type(key) is str: idx = fullparts.index(key); fullparts.remove(key); fullparts = fullparts[:idx] + new + fullparts[idx + 1:]
        elif type(key) is int: fullparts = fullparts[:key] + new + fullparts[key + 1:]
        elif type(key) is slice: fullparts = fullparts[:(0 if key.start is None else key.start)] + new + fullparts[(len(fullparts) if key.stop is None else key.stop):]
        self._str = str(P(*fullparts))  # pylint: disable=W0201  # similar attributes: # self._parts # self._pparts # self._cparts # self._cached_cparts
    def split(self, at: Optional[str] = None, index: Optional[int] = None, sep: Literal[-1, 0, 1] = 1, strict: bool = True):
        if index is None and at is not None:  # at is provided  # ====================================   Splitting
            if not strict:  # behaves like split method of string
                one, two = (items := str(self).split(sep=str(at)))[0], items[1]; one, two = P(one[:-1]) if one.endswith("/") else P(one), P(two[1:]) if two.startswith("/") else P(two)
            else:  # "strict": # raises an error if exact match is not found.
                index = self.parts.index(str(at)); one, two = self[0:index], self[index + 1:]  # both one and two do not include the split item.
        elif index is not None and at is None:  # index is provided
            one, two = self[:index], P(*self.parts[index + 1:])
            at = self.parts[index]  # this is needed below.
        else: raise ValueError("Either `index` or `at` can be provided. Both are not allowed simulatanesouly.")
        if sep == 0: return one, two  # neither of the portions get the sperator appended to it. # ================================  appending `at` to one of the portions
        elif sep == 1: return one, at / two   # append it to right portion
        elif sep == -1: return one / at, two  # append it to left portion.
        else: raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")
    def __repr__(self):  # this is useful only for the console
        if self.is_symlink():
            try: target = self.resolve()  # broken symolinks are funny, and almost always fail `resolve` method.
            except Exception: target = "BROKEN LINK " + str(self)  # avoid infinite recursions for broken links.
            return "🔗 Symlink '" + str(self) + "' ==> " + (str(target) if target == self else str(target))
        elif self.is_absolute(): return self._type() + " '" + str(self.clickable()) + "'" + (" | " + self.time(which="c").isoformat()[:-7].replace("T", "  ") if self.exists() else "") + (f" | {self.size()} Mb" if self.is_file() else "")
        elif "http" in str(self): return "🕸️ URL " + str(self.as_url_str())
        else: return "📍 Relative " + "'" + str(self) + "'"  # not much can be said about a relative path.
    # def __str__(self): return self.as_url_str() if "http" in self else self._str
    def pistol(self):
        import os
        os.system(command=f"pistol {self}")
    def size(self, units: Literal['b', 'kb', 'mb', 'gb'] = 'mb') -> float:  # ===================================== File Specs ==========================================================================================
        total_size = self.stat().st_size if self.is_file() else sum([item.stat().st_size for item in self.rglob("*") if item.is_file()])
        tmp: int
        match units:
            case "b": tmp = 1024 ** 0
            case "kb": tmp = 1024 ** 1
            case "mb": tmp = 1024 ** 2
            case "gb": tmp = 1024 ** 3
        return round(number=total_size / tmp, ndigits=1)
    def time(self, which: Literal["m", "c", "a"] = "m", **kwargs: Any):
        """* `m`: last mofidication of content, i.e. the time it was created.
        * `c`: last status change (its inode is changed, permissions, path, but not content)
        * `a`: last access (read)
        """
        match which:
            case "m": tmp = self.stat().st_mtime
            case "a": tmp = self.stat().st_atime
            case "c": tmp = self.stat().st_ctime
        return datetime.fromtimestamp(tmp, **kwargs)
    def stats(self) -> dict[str, Any]: return dict(size=self.size(), content_mod_time=self.time(which="m"), attr_mod_time=self.time(which="c"), last_access_time=self.time(which="a"), group_id_owner=self.stat().st_gid, user_id_owner=self.stat().st_uid)
    # ================================ String Nature management ====================================
    def _type(self):
        if self.absolute():
            if self.is_file(): return "📄"
            elif self.is_dir(): return "📁"
            return "👻NotExist"
        return "📍Relative"
    def clickable(self, inlieu: bool = False) -> 'P': return self._return(P(self.expanduser().resolve().as_uri()), inlieu)
    def as_url_str(self) -> 'str': return self.as_posix().replace("https:/", "https://").replace("http:/", "http://")
    def as_url_obj(self):
        import urllib3
        tmp = urllib3.connection_from_url(str(self))
        return tmp
    def as_unix(self, inlieu: bool = False) -> 'P': return self._return(P(str(self).replace('\\', '/').replace('//', '/')), inlieu)
    def as_zip_path(self): res = self.expanduser().resolve(); return __import__("zipfile").Path(res)  # .str.split(".zip") tmp=res[1]+(".zip" if len(res) > 2 else ""); root=res[0]+".zip", at=P(tmp).as_posix())  # TODO
    def as_str(self) -> str: return str(self)
    def get_num(self, astring: Optional['str'] = None): int("".join(filter(str.isdigit, str(astring or self.stem))))
    def validate_name(self, replace: str = '_'): return validate_name(self.trunk, replace=replace)
    # ========================== override =======================================
    def write_text(self, data: str, encoding: str = 'utf-8', newline: Optional[str] = None) -> 'P':
        self.parent.mkdir(parents=True, exist_ok=True)
        super(P, self).write_text(data, encoding=encoding, newline=newline); return self
    def read_text(self, encoding: Optional[str] = 'utf-8') -> str: return super(P, self).read_text(encoding=encoding)
    def write_bytes(self, data: bytes, overwrite: bool = False) -> 'P':
        slf = self.expanduser().absolute(); _ = slf.delete(sure=True) if overwrite and slf.exists() else None; res = super(P, slf).write_bytes(data)
        if res == 0: raise RuntimeError(f"Could not save file on disk.")
        return self
    def touch(self, mode: int = 0o666, parents: bool = True, exist_ok: bool = True) -> 'P':  # pylint: disable=W0237
        _ = self.parent.create(parents=parents) if parents else None; super(P, self).touch(mode=mode, exist_ok=exist_ok); return self
    def symlink_from(self, src_folder: OPLike = None, src_file: OPLike = None, verbose: bool = False, overwrite: bool = False):
        assert self.expanduser().exists(), "self must exist if this method is used."
        if src_file is not None:
            assert src_folder is None, "You can only pass source or source_dir, not both."
        result = P(src_folder or P.cwd()).expanduser().absolute() / self.name
        return result.symlink_to(self, verbose=verbose, overwrite=overwrite)
    def symlink_to(self, target: PLike, verbose: bool = True, overwrite: bool = False, orig: bool = False):  # pylint: disable=W0237
        self.parent.create(); assert (target := P(target).expanduser().resolve()).exists(), f"Target path `{target}` doesn't exist. This will create a broken link."
        if overwrite and (self.is_symlink() or self.exists()): self.delete(sure=True, verbose=verbose)
        from platform import system
        from crocodile.meta import Terminal
        if system() == "Windows" and not Terminal.is_user_admin():  # you cannot create symlink without priviliages.
            Terminal.run_as_admin(file=__import__("sys").executable, params=f" -c \"from pathlib import Path; Path(r'{self.expanduser()}').symlink_to(r'{str(target)}')\"", wait=True)
        else: super(P, self.expanduser()).symlink_to(str(target))
        return self._return(P(target), inlieu=False, inplace=False, orig=orig, verbose=verbose, msg=f"LINKED {repr(self)} ➡️ {repr(target)}")
    def resolve(self, strict: bool = False):
        try: return super(P, self).resolve(strict=strict)
        except OSError: return self
    # ======================================== Folder management =======================================
    def search(self, pattern: str = '*', r: bool = False, files: bool = True, folders: bool = True, compressed: bool = False, dotfiles: bool = False, filters: Optional[list[Callable[[Any], bool]]] = None, not_in: Optional[list[str]] = None,
               exts: Optional[list[str]] = None, win_order: bool = False) -> List['P']:
        if isinstance(not_in, list):
            tmp = [lambda x: all([str(notin) not in str(x) for notin in not_in])]  # type: ignore
        else: tmp = []
        if isinstance(exts, list):
            tmp2 = [lambda x: any([ext in x.name for ext in exts])]  # type: ignore
        else: tmp2 = []
        filters = (filters or []) + tmp + tmp2
        if ".zip" in (slf := self.expanduser().resolve()) and compressed:  # the root (self) is itself a zip archive (as opposed to some search results are zip archives)
            root = slf.as_zip_path(); raw = List(root.iterdir()) if not r else List(__import__("zipfile").ZipFile(str(slf)).namelist()).apply(root.joinpath)
            return raw.filter(lambda zip_path: __import__("fnmatch").fnmatch(zip_path.at, pattern)).filter(lambda x: (folders or x.is_file()) and (files or x.is_dir()))  # .apply(lambda x: P(str(x)))
        elif dotfiles: raw = slf.glob(pattern) if not r else self.rglob(pattern)
        else:
            from glob import glob
            raw = glob(str(slf / "**" / pattern), recursive=r) if r else __import__("glob").glob(str(slf.joinpath(pattern)))  # glob ignroes dot and hidden files
        if ".zip" not in slf and compressed:
            tmp = [P(comp_file).search(pattern=pattern, r=r, files=files, folders=folders, compressed=True, dotfiles=dotfiles, filters=filters, not_in=not_in, win_order=win_order) for comp_file in self.search("*.zip", r=r)]
            raw += List(tmp).reduce()  # type: ignore
        processed = []
        for item in raw:
            item_ = P(item)
            if all([item_.is_dir() if not files else True, item_.is_file() if not folders else True] + [afilter(item_) for afilter in filters]):
                processed.append(item_)
        # processed = List([P(item) for item in raw if (lambda item_: all([item_.is_dir() if not files else True, item_.is_file() if not folders else True] + [afilter(item_) for afilter in filters]))(P(item))])
        if not win_order: return List(processed)
        import re
        # return processed if not win_order else processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in __import__("re").split('([0-9]+)', x.stem)])
        processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', string=x.stem)])
        return List(processed)

    def tree(self, *args: Any, **kwargs: Any): return __import__("crocodile.msc.odds").msc.odds.__dict__['tree'](self, *args, **kwargs)
    @property
    def browse(self): return self.search("*").to_struct(key_val=lambda x: ("qq_" + validate_name(str(x)), x)).clean_view
    def create(self, parents: bool = True, exist_ok: bool = True, parents_only: bool = False) -> 'P':
        target_path = self.parent if parents_only else self
        target_path.mkdir(parents=parents, exist_ok=exist_ok)
        return self
    def chdir(self) -> 'P': __import__("os").chdir(str(self.expanduser())); return self
    def listdir(self) -> List['P']: return List(__import__("os").listdir(self.expanduser().resolve())).apply(lambda x: P(x))  # pylint: disable=W0108
    @staticmethod
    def tempdir() -> 'P': return P(__import__("tempfile").mktemp())
    @staticmethod
    def temp() -> 'P': return P(__import__("tempfile").gettempdir())
    @staticmethod
    def tmpdir(prefix: str = "") -> 'P': return P.tmp(folder=rf"tmp_dirs/{prefix + ('_' if prefix != '' else '') + randstr()}")
    @staticmethod
    def tmpfile(name: OPLike = None, suffix: str = "", folder: OPLike = None, tstamp: bool = False, noun: bool = False) -> 'P':
        tmp = randstr(noun=noun) if name is not None else str(name)
        return P.tmp(file=tmp + "_" + randstr() + (("_" + str(timestamp())) if tstamp else "") + suffix, folder=folder or "tmp_files")
    @staticmethod
    def tmp(folder: OPLike = None, file: Optional[str] = None, root: str = "~/tmp_results") -> 'P':
        return P(root).expanduser().joinpath(folder or "").joinpath(file or "").create(parents_only=True if file else False)
    # ====================================== Compression & Encryption ===========================================
    def zip(self, path: OPLike = None, folder: OPLike = None, name: OPLike = None, arcname: Optional[str] = None, inplace: bool = False, verbose: bool = True,
            content: bool = False, orig: bool = False, use_7z: bool = False, pwd: Optional[str] = None, mode: FILE_MODE = 'w', **kwargs: Any) -> 'P':
        path, slf = self._resolve_path(folder, name, path, self.name).expanduser().resolve(), self.expanduser().resolve()
        if use_7z:  # benefits over regular zip and encrypt: can handle very large files with low memory footprint
            path = path + '.7z' if not path.suffix == '.7z' else path
            with install_n_import("py7zr").SevenZipFile(file=path, mode=mode, password=pwd) as archive: archive.writeall(path=str(slf), arcname=None)
        else:
            arcname_obj = P(arcname or slf.name)
            if arcname_obj.name != slf.name: arcname_obj /= slf.name  # arcname has to start from somewhere and end with filename
            if slf.is_file(): path = Compression.zip_file(ip_path=str(slf), op_path=str(path + f".zip" if path.suffix != ".zip" else path), arcname=arcname_obj, mode=mode, **kwargs)
            else:
                if content: root_dir, base_dir = slf, "."
                else: root_dir, base_dir = slf.split(at=str(arcname_obj[0]), sep=1)[0], str(arcname_obj)
                path = P(Compression.compress_folder(root_dir=str(root_dir), op_path=str(path), base_dir=base_dir, fmt='zip', **kwargs))  # TODO: see if this supports mode
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"ZIPPED {repr(slf)} ==>  {repr(path)}")
    def unzip(self, folder: OPLike = None, fname: OPLike = None, verbose: bool = True, content: bool = False, inplace: bool = False, overwrite: bool = False, orig: bool = False,
              pwd: Optional[str] = None, tmp: bool = False, pattern: Optional[str] = None, merge: bool = False, **kwargs: Any) -> 'P':
        _ = merge
        if tmp: return self.unzip(folder=P.tmp().joinpath("tmp_unzips").joinpath(randstr()), content=True).joinpath(self.stem)
        slf = zipfile = self.expanduser().resolve()
        if any(ztype in slf.parent for ztype in (".zip", ".7z")):  # path include a zip archive in the middle.
            if (ztype := [item for item in (".zip", ".7z", "") if item in str(slf)][0]) == "": return slf
            zipfile, fname = slf.split(at=str(List(slf.parts).filter(lambda x: ztype in x)[0]), sep=-1)
        folder = (zipfile.parent / zipfile.stem) if folder is None else P(folder).expanduser().absolute().resolve().joinpath(zipfile.stem)
        folder = folder if not content else folder.parent
        if slf.suffix == ".7z":
            if overwrite: P(folder).delete(sure=True)
            result = folder
            with install_n_import("py7zr").SevenZipFile(file=slf, mode='r', password=pwd) as archive:
                if pattern is not None:
                    import re
                    pat = re.compile(pattern)
                    archive.extract(path=folder, targets=[f for f in archive.getnames() if pat.match(f)])
                else: archive.extractall(path=folder)
        else:
            if overwrite:
                if not content: P(folder).joinpath(fname or "").delete(sure=True, verbose=True)  # deletes a specific file / folder that has the same name as the zip file without extension.
                else: List([x for x in __import__("zipfile").ZipFile(self.str).namelist() if "/" not in x or (len(x.split('/')) == 2 and x.endswith("/"))]).apply(lambda item: P(folder).joinpath(fname or "", item.replace("/", "")).delete(sure=True, verbose=True))
            result = Compression.unzip(zipfile.str, str(folder), None if fname is None else P(fname).as_posix(), **kwargs)
            assert isinstance(result, P)
        return self._return(P(result), inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNZIPPED {repr(zipfile)} ==> {repr(result)}")
    def tar(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name + ".tar").expanduser().resolve()
        Compression.tar(self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"TARRED {repr(self)} ==>  {repr(op_path)}")
    def untar(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name.replace(".tar", "")).expanduser().resolve()
        Compression.untar(self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNTARRED {repr(self)} ==>  {repr(op_path)}")
    def gz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name + ".gz").expanduser().resolve()
        Compression.gz(file=self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"GZED {repr(self)} ==>  {repr(op_path)}")
    def ungz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name.replace(".gz", "")).expanduser().resolve()
        Compression.ungz(self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNGZED {repr(self)} ==>  {repr(op_path)}")
    def xz(self, name: OPLike = None, folder: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name + ".xz").expanduser().resolve()
        Compression.xz(self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"XZED {repr(self)} ==>  {repr(op_path)}")
    def unxz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name.replace(".xz", "")).expanduser().resolve()
        Compression.unxz(self.expanduser().resolve().str, op_path=op_path.str); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNXZED {repr(self)} ==>  {repr(op_path)}")
    def tar_gz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P': return self.tar(inplace=inplace).gz(folder=folder, name=name, path=path, inplace=True, orig=orig, verbose=verbose)
    def ungz_untar(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        return self.ungz(name=f"tmp_{randstr()}.tar", inplace=inplace).untar(folder=folder, name=name, path=path, inplace=True, orig=orig, verbose=verbose)  # this works for .tgz suffix as well as .tar.gz
    def tar_xz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P': return self.tar(inplace=inplace).xz(folder=folder, name=name, path=path, inplace=True, orig=orig, verbose=verbose)
    def unxz_untar(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P': return self.unxz(inplace=inplace).untar(folder=folder, name=name, path=path, inplace=True, orig=orig, verbose=verbose)
    def unbz(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        op_path = self._resolve_path(folder, name, path, self.name.replace(".bz", "").replace(".tbz", ".tar")).expanduser().resolve()
        Compression.unbz(self.expanduser().resolve().str, op_path=str(op_path)); return self._return(op_path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNBZED {repr(self)} ==>  {repr(op_path)}")
    def decompress(self, folder: OPLike = None, name: OPLike = None, path: OPLike = None, inplace: bool = False, orig: bool = False, verbose: bool = True) -> 'P':
        if "tar.gz" in self or "tgz" in self: res = self.ungz_untar(folder=folder, path=path, name=name, inplace=inplace, verbose=verbose, orig=orig)
        elif "zip" in self: res = self.unzip(folder=folder, path=path, name=name, inplace=inplace, verbose=verbose, orig=orig)
        elif "tar.xz" in self: res = self.unxz_untar(folder=folder, path=path, name=name, inplace=inplace, verbose=verbose, orig=orig)
        else: res = self
        return res
    def encrypt(self, key: Optional[bytes] = None, pwd: Optional[str] = None, folder: OPLike = None, name: OPLike = None, path: OPLike = None, verbose: bool = True, suffix: str = ".enc", inplace: bool = False, orig: bool = False) -> 'P':  # see: https://stackoverflow.com/questions/42568262/how-to-encrypt-text-with-a-password-in-python & https://stackoverflow.com/questions/2490334/simple-way-to-encode-a-string-according-to-a-password"""
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.name + suffix)
        assert slf.is_file(), f"Cannot encrypt a directory. You might want to try `zip_n_encrypt`. {self}"; path.write_bytes(encrypt(msg=slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"🔒🔑 ENCRYPTED: {repr(slf)} ==> {repr(path)}.")
    def decrypt(self, key: Optional[bytes] = None, pwd: Optional[str] = None, path: OPLike = None, folder: OPLike = None, name: OPLike = None, verbose: bool = True, suffix: str = ".enc", **kwargs: Any) -> 'P':
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.name.replace(suffix, "") if suffix in slf.name else "decrypted_" + slf.name).write_bytes(decrypt(slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, operation="delete", verbose=verbose, msg=f"🔓🔑 DECRYPTED: {repr(slf)} ==> {repr(path)}.", **kwargs)
    def zip_n_encrypt(self, key: Optional[bytes] = None, pwd: Optional[str] = None, inplace: bool = False, verbose: bool = True, orig: bool = False, content: bool = False) -> 'P': return self.zip(inplace=inplace, verbose=verbose, content=content).encrypt(key=key, pwd=pwd, verbose=verbose, inplace=True) if not orig else self
    def decrypt_n_unzip(self, key: Optional[bytes] = None, pwd: Optional[str] = None, inplace: bool = False, verbose: bool = True, orig: bool = False) -> 'P': return self.decrypt(key=key, pwd=pwd, verbose=verbose, inplace=inplace).unzip(folder=None, inplace=True, content=False) if not orig else self
    def _resolve_path(self, folder: OPLike, name: OPLike, path: OPLike, default_name: str, rel2it: bool = False) -> 'P':  # From all arguments, figure out what is the final path.
        """:param rel2it: `folder` or `path` are relative to `self` as opposed to cwd. This is used when resolving '../dir'"""
        if path is not None:
            path = P(self.joinpath(path).resolve() if rel2it else path).expanduser().resolve()
            assert folder is None and name is None, f"If `path` is passed, `folder` and `name` cannot be passed."; assert not path.is_dir(), f"`path` passed is a directory! it must not be that. If this is meant, pass it with `folder` kwarg. `{path}`"
            return path
        name, folder = (default_name if name is None else str(name)), (self.parent if folder is None else folder)  # good for edge cases of path with single part.  # means same directory, just different name
        return P(self.joinpath(folder).resolve() if rel2it else folder).expanduser().resolve() / name
    def checksum(self, kind: str = ["md5", "sha256"][1]): import hashlib; myhash = {"md5": hashlib.md5, "sha256": hashlib.sha256}[kind](); myhash.update(self.read_bytes()); return myhash.hexdigest()
    @staticmethod
    def get_env():
        import crocodile.environment as env
        return env
    def share_on_cloud(self, timeout: int = 60_000) -> 'P':
        import requests
        return P(requests.put(url=f"https://transfer.sh/{self.expanduser().name}", data=self.expanduser().absolute().read_bytes(), timeout=timeout).text)
    def share_on_network(self, username: OPLike = None, password: Optional[str] = None): from crocodile.meta import Terminal; Terminal(stdout=None).run(f"sharing {self} {('--username ' + str(username)) if username else ''} {('--password ' + password) if password else ''}", shell="powershell")
    def to_qr(self, text: bool = True, path: OPLike = None) -> None:
        qrcode = install_n_import("qrcode"); qr = qrcode.QRCode()
        qr.add_data(str(self) if "http" in str(self) else (self.read_text() if text else self.read_bytes()))
        import io; f = io.StringIO(); qr.print_ascii(out=f); f.seek(0)
        print(f.read()); _ = qr.make_image().save(path) if path is not None else None
    def get_remote_path(self, root: Optional[str], os_specific: bool = False, rel2home: bool = True, strict: bool = True, obfuscate: bool = False) -> 'P':
        tmp1: str = (__import__('platform').system().lower() if os_specific else 'generic_os')
        if not rel2home: path = self
        else:
            try: path = self.rel2home()
            except ValueError as ve:
                if strict: raise ve
                path = self
        if obfuscate:
            from crocodile.msc.obfuscater import obfuscate as obfuscate_func
            path = path.with_name(name=obfuscate_func(seed=P.home().joinpath('dotfiles/creds/data/obfuscation_seed').read_text().rstrip(), data=path.name))
        if isinstance(root, str):  # the following is to avoid the confusing behaviour of A.joinpath(B) if B is absolute.
            part1 = path.parts[0]
            if part1 == "/": sanitized_path = path[1:].as_posix()
            else: sanitized_path = path.as_posix()
            return P(root + "/" + tmp1 + "/" + sanitized_path)
        return tmp1 / path
    def to_cloud(self, cloud: str, remotepath: OPLike = None, zip: bool = False,encrypt: bool = False,  # pylint: disable=W0621, W0622
                 key: Optional[bytes] = None, pwd: Optional[str] = None, rel2home: bool = False, strict: bool = True,
                 obfuscate: bool = False,
                 share: bool = False, verbose: bool = True, os_specific: bool = False, transfers: int = 10, root: Optional[str] = "myhome") -> 'P':
        localpath, to_del = self.expanduser().absolute(), []
        if zip: localpath = localpath.zip(inplace=False); to_del.append(localpath)
        if encrypt: localpath = localpath.encrypt(key=key, pwd=pwd, inplace=False); to_del.append(localpath)
        if remotepath is None:
            rp = localpath.get_remote_path(root=root, os_specific=os_specific, rel2home=rel2home, strict=strict, obfuscate=obfuscate)  # if rel2home else (P(root) / localpath if root is not None else localpath)
        else: rp = P(remotepath)
        rclone_cmd = f"""rclone copyto '{localpath.as_posix()}' '{cloud}:{rp.as_posix()}' {'--progress' if verbose else ''} --transfers={transfers}"""
        from crocodile.meta import Terminal, subprocess; _ = print(f"{'⬆️'*5} UPLOADING with `{rclone_cmd}`") if verbose else None
        res = Terminal(stdout=None if verbose else subprocess.PIPE).run(rclone_cmd, shell="powershell").capture()
        _ = [item.delete(sure=True) for item in to_del]; _ = print(f"{'⬆️'*5} UPLOAD COMPLETED.") if verbose else None
        assert res.is_successful(strict_err=False, strict_returcode=True), res.print(capture=False)
        if share:
            if verbose: print("🔗 SHARING FILE")
            res = Terminal().run(f"""rclone link '{cloud}:{rp.as_posix()}'""", shell="powershell").capture()
            tmp = res.op2path(strict_err=True, strict_returncode=True)
            if tmp is None:
                res.print()
                raise RuntimeError(f"💥 Could not get link for {self}.")
            return tmp
        return self
    def from_cloud(self, cloud: str, remotepath: OPLike = None, decrypt: bool = False, unzip: bool = False,  # type: ignore  # pylint: disable=W0621
                   key: Optional[bytes] = None, pwd: Optional[str] = None, rel2home: bool = False, os_specific: bool = False, strict: bool = True, obfuscate: bool = False,
                   transfers: int = 10, root: Optional[str] = "myhome", verbose: bool = True, overwrite: bool = True, merge: bool = False,):
        if remotepath is None:
            remotepath = self.get_remote_path(root=root, os_specific=os_specific, rel2home=rel2home, strict=strict)
            remotepath += ".zip" if unzip else ""; remotepath += ".enc" if decrypt else ""
        else: remotepath = P(remotepath)
        localpath = self.expanduser().absolute()
        localpath += ".zip" if unzip else ""
        localpath += ".enc" if decrypt else ""
        rclone_cmd = f"""rclone copyto '{cloud}:{remotepath.as_posix()}' '{localpath.as_posix()}' {'--progress' if verbose else ''} --transfers={transfers}"""
        from crocodile.meta import Terminal, subprocess
        _ = print(f"{'⬇️' * 5} DOWNLOADING with `{rclone_cmd}`") if verbose else None
        res = Terminal(stdout=None if verbose else subprocess.PIPE).run(rclone_cmd, shell="powershell")
        assert res.is_successful(strict_err=False, strict_returcode=True), res.print(capture=False)
        if decrypt: localpath = localpath.decrypt(key=key, pwd=pwd, inplace=True)
        if unzip: localpath = localpath.unzip(inplace=True, verbose=True, overwrite=overwrite, content=True, merge=merge)
        return localpath
    def sync_to_cloud(self, cloud: str, sync_up: bool = False, sync_down: bool = False, os_specific: bool = False, rel2home: bool = True, transfers: int = 10, delete: bool = False, root: Optional[str] = "myhome", verbose: bool = True):
        tmp1, tmp2 = self.expanduser().absolute().create(parents_only=True).as_posix(), self.get_remote_path(root=root, os_specific=os_specific).as_posix()
        source, target = (tmp1, f"{cloud}:{tmp2 if rel2home else tmp1}") if sync_up else (f"{cloud}:{tmp2 if rel2home else tmp1}", tmp1)  # in bisync direction is irrelavent.
        if not sync_down and not sync_up: _ = print(f"SYNCING 🔄️ {source} {'<>' * 7} {target}`") if verbose else None; rclone_cmd = f"""rclone bisync '{source}' '{target}' --resync --remove-empty-dirs """
        else: print(f"SYNCING 🔄️ {source} {'>' * 15} {target}`"); rclone_cmd = f"""rclone sync '{source}' '{target}' """
        rclone_cmd += f" --progress --transfers={transfers} --verbose"; rclone_cmd += (" --delete-during" if delete else ""); from crocodile.meta import Terminal, subprocess; _ = print(rclone_cmd) if verbose else None
        res = Terminal(stdout=None if verbose else subprocess.PIPE).run(rclone_cmd, shell="powershell")
        assert res.is_successful(strict_err=False, strict_returcode=True), res.print(capture=False)
        return self
    @property  # kept at the bottom because it confuses the linters
    def str(self) -> str: return str(self)  # or self._str


class Compression:
    @staticmethod
    def compress_folder(root_dir: str, op_path: str, base_dir: str, fmt: SHUTIL_FORMATS = 'zip', verbose: bool = False, **kwargs: Any) -> str:  # shutil works with folders nicely (recursion is done interally) # directory to be archived: root_dir\base_dir, unless base_dir is passed as absolute path. # when archive opened; base_dir will be found."""
        base_name = op_path[:-4] if op_path.endswith(".zip") else op_path  # .zip is added automatically by library, hence we'd like to avoid repeating it if user sent it.
        import shutil; return shutil.make_archive(base_name=base_name, format=fmt, root_dir=root_dir, base_dir=base_dir, verbose=verbose, **kwargs)  # returned path possible have added extension.
    @staticmethod
    def zip_file(ip_path: str, op_path: str, arcname: OPLike = None, password: Optional[bytes] = None, mode: FILE_MODE = "w", **kwargs: Any):
        """arcname determines the directory of the file being archived inside the archive. Defaults to same as original directory except for drive.
        When changed, it should still include the file path in its end. If arcname = filename without any path, then, it will be in the root of the archive."""
        import zipfile
        with zipfile.ZipFile(op_path, mode=mode) as jungle_zip:
            if password is not None: jungle_zip.setpassword(pwd=password)
            jungle_zip.write(filename=str(ip_path), arcname=str(arcname) if arcname is not None else None, compress_type=zipfile.ZIP_DEFLATED, **kwargs)
        return P(op_path)
    @staticmethod
    def unzip(ip_path: str, op_path: str, fname: OPLike = None, password: Optional[str] = None, memory: bool = False, **kwargs: Any):
        with __import__("zipfile").ZipFile(str(ip_path), 'r') as zipObj:
            if memory: return {name: zipObj.read(name) for name in zipObj.namelist()} if fname is None else zipObj.read(fname)
            if fname is None: zipObj.extractall(op_path, pwd=password, **kwargs); return P(op_path)
            else: zipObj.extract(member=str(fname), path=str(op_path), pwd=password); return P(op_path) / fname
    @staticmethod
    def gz(file: str, op_path: str):  # see this on what to use: https://stackoverflow.com/questions/10540935/what-is-the-difference-between-tar-and-zip
        with open(file, 'rb') as f_in:
            with __import__("gzip").open(op_path, 'wb') as f_out: __import__("shutil").copyfileobj(f_in, f_out)
        return P(op_path)
    @staticmethod
    def ungz(path: str, op_path: str):
        with __import__("gzip").open(path, 'r') as f_in, open(op_path, 'wb') as f_out: __import__("shutil").copyfileobj(f_in, f_out)
        return P(op_path)
    @staticmethod
    def unbz(path: str, op_path: str):
        with __import__("bz2").BZ2File(path, 'r') as fr, open(str(op_path), 'wb') as fw: __import__("shutil").copyfileobj(fr, fw)
        return P(op_path)
    @staticmethod
    def xz(path: str, op_path: str):
        with __import__("lzma").open(op_path, "w") as f: f.write(path)
    @staticmethod
    def unxz(ip_path: str, op_path: str):
        with __import__("lzma").open(ip_path) as file: P(op_path).write_bytes(file.read())
    @staticmethod
    def tar(path: str, op_path: str):
        with __import__("tarfile").open(op_path, "w:gz") as tar_: tar_.add(str(path), arcname=__import__("os").path.basename(path))
        return P(op_path)
    @staticmethod
    def untar(path: str, op_path: str, fname: OPLike = None, mode: str = 'r', **kwargs: Any):
        with __import__("tarfile").open(str(path), mode) as file:
            if fname is None: file.extractall(path=op_path, **kwargs)  # extract all files in the archive
            else: file.extract(fname, **kwargs)
        return P(op_path)


T = TypeVar('T')


class PrintFunc(Protocol):
    def __call__(self, *args: str) -> Union[NoReturn, None]: ...


class Cache:  # This class helps to accelrate access to latest data coming from expensive function. The class has two flavours, memory-based and disk-based variants."""
    def __init__(self, source_func: Callable[[], 'T'], expire: Union[str, timedelta] = "1m", logger: Optional[PrintFunc] = None, path: OPLike = None, saver: Callable[[T, PLike], Any] = Save.vanilla_pickle, reader: Callable[[PLike], T] = Read.read, name: Optional[str] = None) -> None:
        self.cache: Optional[T] = None  # fridge content
        self.source_func = source_func  # function which when called returns a fresh object to be frozen.
        self.path: Optional[P] = P(path) if path else None  # if path is passed, it will function as disk-based flavour.
        self.time_produced = datetime.now()  # if path is None else
        self.save = saver
        self.reader = reader
        self.logger = logger
        self.expire = str2timedelta(expire) if isinstance(expire, str) else expire
        self.name = name if isinstance(name, str) else str(self.source_func)
    @property
    def age(self):
        """Throws AttributeError if called before cache is populated and path doesn't exists"""
        if self.path is None:  # memory-based cache.
            return datetime.now() - self.time_produced
        return datetime.now() - datetime.fromtimestamp(self.path.stat().st_mtime)
    def __setstate__(self, state: dict[str, Any]) -> None: self.__dict__.update(state); self.path = P.home() / self.path if self.path is not None else self.path
    def __getstate__(self) -> dict[str, Any]: state = self.__dict__.copy(); state["path"] = self.path.rel2home() if self.path is not None else state["path"]; return state  # With this implementation, instances can be pickled and loaded up in different machine and still works.
    def __call__(self, fresh: bool = False) -> 'T':  # type: ignore
        if fresh or self.cache is None:  # populate cache for the first time
            if not fresh and self.path is not None and self.path.exists():
                age = datetime.now() - datetime.fromtimestamp(self.path.stat().st_mtime)
                if self.logger: self.logger(f"⚠️ {self.name} cache: Reading cached values from `{self.path}`. Lag = {age} ...")
                self.cache = self.reader(self.path)
                return self(fresh=False)  # may be the cache is old ==> check that by passing it through the logic again.
            else:
                if self.logger: self.logger(f"⚠️ {self.name} cache: Populating fresh cache from {self.source_func}. Previous cache never existed or a there was an explicit fresh order.")
                self.cache = self.source_func()  # fresh data.
                if self.path is None: self.time_produced = datetime.now()
                else: self.save(self.cache, self.path)
        else:  # cache exists
            age = self.age
            if self.age > self.expire:
                if self.logger: self.logger(f"⚠️ {self.name} cache: Updating cache from {self.source_func}. Age = {age} ...")
                self.cache = self.source_func()
                if self.path is None: self.time_produced = datetime.now()
                else: self.save(self.cache, self.path)
            else:
                if self.logger: self.logger(f"⚠️ {self.name} cache: Using cached values. Lag = {age}.")
        return self.cache  # type: ignore


if __name__ == '__main__':
    # print('hi from file_managements')
    pass

# %%
