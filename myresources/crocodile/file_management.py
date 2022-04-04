
from crocodile.core import Struct, List, timestamp, randstr, validate_name, str2timedelta, Save, Path, install_n_import
from datetime import datetime


# =============================== Security ================================================
def obscure(msg: bytes) -> bytes: return __import__("base64").urlsafe_b64encode(__import__("zlib").compress(msg, 9))
def unobscure(obscured: bytes) -> bytes: return __import__("zlib").decompress(__import__("base64").urlsafe_b64decode(obscured))


def pwd2key(password: str, salt=None, iterations=None) -> bytes:
    """Derive a secret key from a given password and salt"""
    if salt is None:
        m = __import__("hashlib").sha256(); m.update(password.encode("utf-8"))
        return __import__("base64").urlsafe_b64encode(m.digest())  # make url-safe bytes required by Ferent.
    from cryptography.hazmat.primitives import hashes; from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return __import__("base64").urlsafe_b64encode(PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=None).derive(password.encode()))


def encrypt(msg: bytes, key=None, pwd: str = None, salted=True, iteration: int = None) -> bytes:
    salt = None  # silence the linter.
    if pwd is not None:  # generate it from password
        assert (key is None) and (type(pwd) is str), f"You can either pass key or pwd, or none of them, but not both."
        if salted: import secrets; salt, iteration = secrets.token_bytes(16), iteration or secrets.randbelow(1_000_000)
        else: salt, iteration = None, None
        key = pwd2key(pwd, salt, iteration)
    elif key is None:  # generate a new key: discouraged, always make your keys/pwd before invoking the func.
        key = __import__("cryptography.fernet").__dict__["fernet"].Fernet.generate_key()  # uses random bytes, more secure but no string representation
        print(f"KEY SAVED @ {repr(P.tmpdir().joinpath('key.bytes').write_bytes(key))}")  # without verbosity check:
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # a path to a key file was passed, read it:
    elif type(key) is bytes: pass  # key passed explicitly
    else: raise TypeError(f"Key must be either a path, bytes object or None.")
    code = __import__("cryptography.fernet").__dict__["fernet"].Fernet(key).encrypt(msg)
    return __import__("base64").urlsafe_b64encode(b'%b%b%b' % (salt, iteration.to_bytes(4, 'big'), __import__("base64").urlsafe_b64decode(code))) if pwd is not None and salted is True else code


def decrypt(token: bytes, key=None, pwd: str = None, salted=True) -> bytes:
    if pwd is not None:
        assert key is None, f"You can either pass key or pwd, or none of them, but not both."
        if salted:
            decoded = __import__("base64").urlsafe_b64decode(token)
            salt, iterations, token = decoded[:16], decoded[16:20], __import__("base64").urlsafe_b64encode(decoded[20:])
            key = pwd2key(pwd, salt, int.from_bytes(iterations, 'big'))
        else: key = pwd2key(pwd)
    if type(key) is bytes: pass   # passsed explicitly
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # passed a path to a file containing kwy
    else: raise TypeError(f"Key must be either str, P, Path or bytes.")
    return __import__("cryptography.fernet").__dict__["fernet"].Fernet(key).decrypt(token)


# =================================== File ============================================

class Read(object):
    @staticmethod
    def read(path, **kwargs):
        suffix = Path(path).suffix[1:]
        try: return getattr(Read, suffix)(str(path), **kwargs)
        except AttributeError:
            if suffix in ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']: return __import__("matplotlib.pyplot").imread(path, **kwargs)  # from: plt.gcf().canvas.get_supported_filetypes().keys():
            raise AttributeError(f"Unknown file type. failed to recognize the suffix {suffix}")

    @staticmethod
    def npy(path, **kwargs):
        import numpy as np; data = np.load(str(path), allow_pickle=True, **kwargs)
        if data.dtype == np.object: data = data.item(); return Struct(data) if type(data) is dict else data

    @staticmethod
    def mat(path, remove_meta=False, **kwargs):
        res = Struct(__import__("scipy.io").__dict__["io"].loadmat(path, **kwargs))
        if remove_meta: List(res.keys()).filter("x.startswith('__')").apply(lambda x: res.__delattr__(x))
        return res

    @staticmethod
    def json(path, r=False, **kwargs):
        """Returns a Structure"""
        try: mydict = __import__("json").loads(P(path).read_text(), **kwargs)
        except Exception: mydict = install_n_import("pyjson5").loads(P(path).read_text(), **kwargs)  # file has C-style comments.
        return Struct.recursive_struct(mydict) if r else Struct(mydict)

    @staticmethod
    def yaml(path, r=False):
        import yaml
        with open(str(path), "r") as file: mydict = yaml.load(file, Loader=yaml.FullLoader)
        return Struct(mydict) if not r else Struct.recursive_struct(mydict)

    @staticmethod
    def csv(path, **kwargs): return __import__("pandas").read_csv(path, **kwargs)
    @staticmethod
    def pkl(*args, **kwargs): return Read.pickle(*args, **kwargs)
    py = staticmethod(lambda path: Struct(__import__("runpy").run_path(path)))
    pickles = staticmethod(lambda bytes_obj: __import__("dill").loads(bytes_obj))
    @staticmethod
    def pickle(path, **kwargs): obj = __import__("dill").loads(P(path).read_bytes(), **kwargs); return Struct(obj) if type(obj) is dict else obj


class P(type(Path()), Path):
    # ==================================== File management =========================================
    """ The default behaviour of methods acting on underlying disk object is to perform the action
        and return a new path referring to the mutated object in disk drive. However, there is a flag `orig` that makes
        the function return orignal path object `self` as opposed to the new one pointing to new object.
        Additionally, the fate of the original object can be decided by a flag `inplace` which means `replace`
        it defaults to False and in essence, it deletes the original underlying object. This can be seen in `zip` and `encrypt`
        but not in `copy`, `move`, `retitle` because the fate of original file is dictated already.
        Furthermore, those methods are accompanied with print statement explaining what happened to the object.
    """
    def delete(self, sure=False, verbose=True):
        slf = self  # slf = self.expanduser().resolve() don't resolve symlinks.
        if sure:
            if not slf.exists():
                slf.unlink(missing_ok=True)  # broken symlinks exhibit funny existence behaviour, catch them here.
                if verbose: print(f"Could NOT DELETE nonexisting file {repr(slf)}. ")
                return slf  # terminate the function.
            if slf.is_file() or slf.is_symlink(): slf.unlink(missing_ok=True)
            else: __import__("shutil").rmtree(slf, ignore_errors=True)
            if verbose: print(f"DELETED {repr(slf)}.")
        elif verbose: print(f"Did NOT DELETE because user is not sure. file: {repr(slf)}.")
        return self

    def send2trash(self, verbose=True):
        if self.exists(): install_n_import("send2trash").send2trash(self.resolve().str); print(f"TRASHED {repr(self)}") if verbose else None  # do not expand user symlinks.
        elif verbose: print(f"Could NOT trash {self}"); return self

    def move(self, folder=None, name=None, path=None, rel2it=False, overwrite=False, verbose=True, parents=True, content=False):
        path = self._resolve_path(folder=folder, name=name, path=path, default_name=self.absolute().name, rel2it=rel2it)
        name, folder = path.name, path.parent
        if parents: folder.create(parents=True, exist_ok=True)
        slf = self.expanduser().resolve()
        if content:
            assert self.is_dir(), NotADirectoryError(f"When `content` flag is set to True, path must be a directory. It is not: `{repr(self)}`")
            self.search("*").apply(lambda x: x.move(path=path, content=False))
            return path  # contents live within this directory.
        if overwrite:  # the following works safely even if you are moving a path up and parent has same path.
            path_ = P(folder).absolute() / randstr()  # no conflict with existing files/dirs of same `self.path`
            slf.rename(path_)  # no error are likely to occur as the random path won't cause conflict.
            # now we can delete any potential conflict before eventually taking its path
            path.delete(sure=True, verbose=False)  # It is important to delete after moving
            # because `self` could be within the file you want to delete.
            path_.rename(path)
        else: slf.rename(path)
        if verbose: print(f"MOVED {repr(self)} ==> {repr(path)}`")
        return path

    def retitle(self, name, overwrite=False, verbose=True, orig=False):
        """Unlike the builtin `rename`, this doesn't require or change full path, only file name."""
        assert type(name) is str, "New new should be a string representing file name alone."
        new_path = self.parent / name
        if overwrite and new_path.exists(): new_path.delete(sure=True)
        self.rename(new_path)
        return self._return(new_path, inlieu=False, inplace=False, orig=orig, verbose=verbose, msg=f"RENAMED {repr(self)} ==> {repr(new_path)}")

    def copy(self, folder=None, name=None, path=None, content=False, verbose=True, append=f"_copy_{randstr()}", overwrite=False, orig=False):  # tested %100
        if folder is not None and path is None:
            if name is None: dest = P(folder).expanduser().resolve().create()
            else: dest, content = P(folder).expanduser().resolve() / name, True
        elif path is not None and folder is None: dest, content = P(path), True  # this way, the destination will be filled with contents of `self`
        elif path is None and folder is None: dest = self.with_name(str(name)) if name is not None else self.append(append)
        else: raise NotImplementedError
        dest, slf = dest.expanduser().resolve().create(parent_only=True), self.expanduser().resolve()
        if overwrite and dest.exists(): dest.delete(sure=True)
        if slf.is_file():
            __import__("shutil").copy(str(slf), str(dest))
            if verbose: print(f"COPIED {repr(slf)} ==> {repr(dest)}")
        elif slf.is_dir():
            __import__("distutils.dir_util").__dict__["dir_util"].copy_tree(str(slf), str(dest) if content else str(P(dest).joinpath(slf.name).create()))
            if verbose: print(f"COPIED {'Content of ' if content else ''} {repr(slf)} ==> {repr(dest)}")
        else: print(f"Could NOT COPY. Not a file nor a path: {repr(slf)}.")
        return dest / slf.name if not orig else self

    # ======================================= File Editing / Reading ===================================
    def readit(self, reader=None, notfound=FileNotFoundError, verbose=False, **kwargs):
        if not self.exists():
            if notfound is FileNotFoundError: raise FileNotFoundError(f"`{self}` is no where to be found!")
            else: return notfound
        filename = self
        if '.zip' in str(self): filename = self.unzip(folder=self.tmp(folder="unzipped"), verbose=verbose)
        try: return Read.read(filename, **kwargs) if reader is None else reader(str(filename), **kwargs)
        except IOError: raise IOError

    def start(self, opener=None):
        import subprocess
        if str(self).startswith("http") or str(self).startswith("www"): __import__("webbrowser").open(str(self)); return self
        filename = self.expanduser().resolve().str
        if __import__("sys").platform == "win32":
            if opener is None: tmp = f"powershell start '{filename}'"  # double quotes fail with cmd.
            else: tmp = rf'powershell {opener} \'{self}\'' # __import__("os)s.tartfile(filename)  # works for files and folders alike, but if opener is given, e.g. opener="start"
            subprocess.Popen(tmp)  # fails for folders. Start must be passed, but is not defined.
        elif __import__("sys").platform == 'linux': subprocess.call(["xdg-open", filename])  # works for files and folders alike
        else:  subprocess.call(["open", filename])  # works for files and folders alike  # mac
        return self

    def __call__(self, *args, **kwargs): self.start(*args, **kwargs); return self
    def append_text(self, appendix): self.write_text(self.read_text() + appendix); return self
    def read_fresh_from(self, source_func, expire="1w", save=Save.pickle, read=Read.read): return Fridge(source_func=source_func, path=self, expire=expire, save=save, read=read)

    def modify_text(self, txt, alt, newline=False, notfound_append=False, encoding=None):
        """
        :param notfound_append: if file not found, append the text to it.
        :param encoding:
        :param txt: text to be searched for in the file. The line in which it is found will be up for change.
        :param alt: alternative text that will replace `txt`. Either a string or a function returning a string
        :param newline: completely remove the line in which `txt` was found and replace it with `alt`.
        :return:
        * This method is suitable for config files and simple scripts that has one-liners in it,
        * File is created if it doesn't exist.
        * Text is simply appended if not found in the text file.
        """
        if not self.exists(): self.create(parent_only=True).write_text(txt)
        lines = self.read_text(encoding=encoding).split("\n")
        bingo = False
        for idx, line in enumerate(lines):
            if txt in line:
                bingo = True
                if newline is True: lines[idx] = alt if type(alt) is str else alt(line)
                else: lines[idx] = line.replace(txt, alt if type(alt) is str else alt(line))
        if bingo is False and notfound_append is True: lines.append(alt)  # txt not found, add it anyway.
        return self.write_text("\n".join(lines), encoding=encoding)

    def download(self, directory=None, name=None, memory=False, allow_redirects=True, params=None):  # fails at html.
        response = __import__("requests").get(self.as_url_str(), allow_redirects=allow_redirects, params=params)  # Alternative: from urllib import request; request.urlopen(url).read().decode('utf-8').
        return response if memory else (P.home().joinpath("Downloads") if directory is None else P(directory)).joinpath(name or self.name).write_bytes(response.content)  # r.contents is bytes encoded as per docs of requests.

    def _return(self, res, inlieu: bool, inplace=False, operation=None, orig=False, verbose=False, msg=""):
        if inlieu: self._str = str(res)
        if inplace:
            assert res.exists(), f"`inplace` flag is only relevant if the path exists. It doesn't {self}"
            if operation == "rename": self.rename(res)
            if operation == "delete": self.delete(sure=True, verbose=verbose)
        if verbose: print(msg)
        return self if orig else res

    # ================================ Path Object management ===========================================
    """ Distinction between Path object and the underlying file on disk that the path may refer to. Two distinct flags are used:
        `inplace`: the operation on the path object will affect the underlying file on disk if this flag is raised, otherwise the method will only alter the string.
        `inliue`: the method acts on the path object itself instead of creating a new one if this flag is raised.
        `orig`: whether the method returns the original path object or a new one.
    """
    def prepend(self, prefix, suffix=None, inlieu=False, inplace=False): return self._return(self.parent.joinpath(prefix + self.trunk + (suffix or ''.join(self.suffixes))), inlieu=inlieu, inplace=inplace, operation="rename")
    def append(self, name='', suffix=None, inplace=False, inlieu=False): return self._return(self.parent.joinpath(self.trunk + name + (suffix or ''.join(self.suffixes))), inlieu=inlieu, inplace=inplace, operation="rename")
    def append_time_stamp(self, fmt=None, inlieu=False, inplace=False): return self._return(self.append(name="_" + timestamp(fmt=fmt)), inlieu=inlieu, inplace=inplace, operation="rename")
    def with_trunk(self, name, inlieu=False, inplace=False): return self._return(self.parent.joinpath(name + "".join(self.suffixes)), inlieu=inlieu, inplace=inplace, operation="rename")  # Complementary to `with_stem` and `with_suffix`
    def switch(self, key: str, val: str, inlieu=False, inplace=False): return self._return(P(str(self).replace(key, val)), inlieu=inlieu, inplace=inplace, operation="rename")  # Like string replce method, but `replace` is an already defined method."""
    def switch_by_index(self, idx: int, val: str, inplace=False, inlieu=False): return self._return(P(*[val if index == idx else value for index, value in enumerate(self.parts)]), inlieu=inlieu, inplace=inplace, operation="rename")
    # ============================= attributes of object ======================================
    trunk = property(lambda self: self.name.split('.')[0])  # """ useful if you have multiple dots in file path where `.stem` fails."""
    len = property(lambda self: self.__len__())
    str = property(lambda self: str(self))  # or self._str
    items = property(lambda self: List(self.parts))
    def __len__(self): return len(self.parts)
    def __contains__(self, item): return item in self.parts
    def __iter__(self): return self.parts.__iter__()
    def __deepcopy__(self): return P(str(self))
    def __getstate__(self): return str(self)
    def __setstate__(self, state): self._str = str(state)
    def __add__(self, other): return self.parent.joinpath(self.stem + str(other))
    def __radd__(self, other): return self.parent.joinpath(str(other) + self.stem)  # other + P and `other` doesn't know how to make this addition.
    def __rtruediv__(self, other): super(P, self).__rtruediv__(other)
    def __sub__(self, other): res = P(str(self).replace(str(other), "")); return res[1:] if str(res[0]) in {"\\", "/"} else res  # paths starting with "/" are problematic. e.g ~ / "/path" doesn't work.
    def rel2cwd(self, inlieu=False): return self._return(P(self.relative_to(Path.cwd())), inlieu)
    def rel2home(self, inlieu=False): return self._return(P(self.relative_to(Path.home())), inlieu)  # opposite of `expanduser`

    def collapseuser(self, strict=True, inlieu=False):
        if strict: assert str(P.home()) in str(self), ValueError(f"{str(P.home())} is not in the subpath of {str(self)} OR one path is relative and the other is absolute.")
        return self if "~" in self else self._return("~" / (self - P.home()), inlieu)

    def split(self, at: str = None, index: int = None, sep: int = 1, mode=["strict", "lenient"][0]):
        """Splits a path at a given string or index
        :param at: string telling where to split.
        :param index: integer telling at which index to split.
        :param sep: can be either [-1, 0, 1]. Determines where the separator is going to live with: left portion, none or right portion.
        :param mode: "lenient" mode makes `split` method behaves like split method of string. This can produce unwanted behaviour due to e.g. patial matches. 'strict' mode is the default which only splits at exact match.
        :return: two paths
        """
        # ====================================   Splitting
        if index is None and (at is not None):  # at is provided
            if mode == "lenient":
                items = str(self).split(sep=str(at))
                one, two = items[0], items[1]
                one, two = one[:-1] if one.endswith("/") else one, two[1:] if two.startswith("/") else two
            else:  # "strict"
                index = self.parts.index(str(at))  # raises an error if exact match is not found.
                one, two = self[0:index], self[index + 1:]  # both one and two do not include the split item.
            one, two = P(one), P(two)
        elif index is not None and (at is None):  # index is provided
            one, two = self[:index], P(*self.parts[index + 1:])
            at = self[index]  # this is needed below.
        else: raise ValueError("Either `index` or `at` can be provided. Both are not allowed simulatanesouly.")
        # ================================  appending `at` to one of the portions
        if sep == 0: pass  # neither of the portions get the sperator appended to it.
        elif sep == 1: two = at / two   # append it to right portion
        elif sep == -1: one = one / at  # append it to left portion.
        else: raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")
        return one, two

    def __getitem__(self, slici):  # tested.
        if type(slici) is list: return P(*[self[item] for item in slici])
        else: return P(*self.parts[slici]) if type(slici) is slice else P(self.parts[slici])  # it is an integer

    def __setitem__(self, key: str or int or slice, value: str or Path):
        fullparts, new = list(self.parts), list(P(value).parts)
        if type(key) is str:
            idx = fullparts.index(key)
            fullparts.remove(key)
            fullparts = fullparts[:idx] + new + fullparts[idx + 1:]
        elif type(key) is int or type(key) is slice:
            if type(key) is int: fullparts = fullparts[:key] + new + fullparts[key + 1:]  # replace this entry
            elif type(key) is slice:
                if key.stop is None: key = slice(key.start, len(fullparts), key.step)
                if key.start is None: key = slice(0, key.stop, key.step)
                fullparts = fullparts[:key.start] + new + fullparts[key.stop:]
        self._str = str(P(*fullparts))  # similar attributes: # self._parts # self._pparts # self._cparts # self._cached_cparts

    def __repr__(self):  # this is useful only for the console
        rep = "P:"
        if self.is_symlink():
            try: target = self.resolve()  # broken symolinks are funny, and almost always fail `resolve` method.
            except Exception: target = "BROKEN LINK " + str(self)
            if target == self: target = str(target)  # avoid infinite recursions for broken links.
            rep += " Symlink '" + str(self) + "' ==> " + repr(target)
        elif self.is_absolute():
            rep += " " + self._type() + " '" + self.clickable() + "'"
            if self.exists():
                rep += " | " + self.time(which="c").isoformat()[:-7].replace("T", "  ")
                if self.is_file(): rep += f" | {self.size()} Mb"
        elif "http" in str(self): rep += " URL " + self.as_url_str()
        else: rep += " Relative " + "'" + str(self) + "'"  # not much can be said about a relative path.
        return rep

    # %% ===================================== File Specs =============================================================
    def size(self, units='mb'):
        sizes = List(['b', 'kb', 'mb', 'gb']); import numpy as np; factor = dict(zip(sizes + sizes.apply(lambda x: x.swapcase()), np.tile(1024 ** np.arange(len(sizes)), 2)))[units]
        total_size = self.stat().st_size if self.is_file() else sum([item.stat().st_size for item in self.rglob("*") if item.is_file()])
        return round(total_size / factor, 1)

    def time(self, which="m", **kwargs):
        """* ``m`` time of modifying file ``content``, i.e. the time_produced it was created.
            * ``c`` time of changing file status (its inode is changed like permissions, path etc, but not content)
            * ``a`` last time the file was accessed."""
        return datetime.fromtimestamp({"m": self.stat().st_mtime, "a": self.stat().st_atime, "c": self.stat().st_ctime}[which], **kwargs)

    def stats(self): return Struct(size=self.size(), content_mod_time=self.time(which="m"), attr_mod_time=self.time(which="c"), last_access_time=self.time(which="a"), group_id_owner=self.stat().st_gid, user_id_owner=self.stat().st_uid)
    # ================================ String Nature management ====================================
    def _type(self): return ("File" if self.is_file() else ("Dir" if self.is_dir() else "NotExist")) if self.absolute() else "Relative"
    def clickable(self, inlieu=False): return self._return(self.expanduser().resolve().as_uri(), inlieu)
    def as_url_str(self, inlieu=False): return self._return(self.as_posix().replace("https:/", "https://").replace("http:/", "http://"), inlieu)
    def as_url_obj(self, inlieu=False): return self._return(install_n_import("urllib3").connection_from_url(self), inlieu)
    def as_unix(self, inlieu=False): return self._return(P(str(self).replace('\\', '/').replace('//', '/')), inlieu)
    def get_num(self, astring=None): int("".join(filter(str.isdigit, str(astring or self.stem))))
    def validate_name(self, replace='_'): validate_name(self.trunk, replace=replace)
    # ========================== override =======================================
    def write_text(self, data: str, **kwargs): super(P, self).write_text(data, **kwargs); return self
    def read_text(self, encoding=None, lines=False): return super(P, self).read_text(encoding=encoding) if not lines else List(super(P, self).read_text(encoding=encoding).splitlines())
    def write_bytes(self, data: bytes): super(P, self).write_bytes(data); return self
    def touch(self, mode: int = 0o666, parents=True, exist_ok: bool = ...): self.parent.create(parents=parents) if parents else None; super(P, self).touch(mode=mode, exist_ok=exist_ok); return self

    def symlink_from(self, folder=None, file=None, verbose=False, overwrite=False):
        assert self.expanduser().exists(), "self must exist if this method is used."
        if file is not None: assert folder is None, "You can only pass source or source_dir, not both."; result = P(file).expanduser().absolute()
        else: result = P(folder or P.cwd()).expanduser().absolute() / self.name
        return result.symlink_to(self, verbose=verbose, overwrite=overwrite)

    def symlink_to(self, target=None, verbose=True, overwrite=False, orig=False):
        target = P(target).expanduser().resolve()
        assert target.exists(), f"Target path `{target}` doesn't exist. This will create a broken link."
        self.parent.create()
        if overwrite and (self.is_symlink() or self.exists()): self.delete(sure=True, verbose=verbose)
        from crocodile.meta import Terminal
        if __import__("platform").system() == "Windows" and not Terminal.is_user_admin():  # you cannot create symlink without priviliages.
            Terminal.run_code_as_admin(f" -c \"from pathlib import Path; Path(r'{self.expanduser()}').symlink_to(r'{str(target)}')\""); __import__("time").sleep(0.5)  # give time_produced for asynch process to conclude before returning response.
        else: super(P, self.expanduser()).symlink_to(str(target))
        return self._return(P(target), inlieu=False, inplace=False, orig=orig, verbose=verbose, msg=f"LINKED {repr(self)}")

    def resolve(self, strict=False):
        try: return super(P, self).resolve(strict=strict)
        except OSError: return self

    # ======================================== Folder management =======================================
    def search(self, pattern='*', r=False, files=True, folders=True, compressed=False, dotfiles=False, filters: list = None, not_in: list = None, exts=None, win_order=False):
        """
        :param pattern:  linux search pattern
        :param r: recursive search flag
        :param files: include files in search.
        :param folders: include directories in search.
        :param compressed: search inside compressed files.
        :param dotfiles: flag to indicate whether the search should include those or not.
        :param filters: list of filters
        :param not_in: list of strings that search results should not contain them (short for filter with simple lambda)
        :param exts: list of extensions to search for.
        :param win_order: return search results in the order of files as they appear on a Windows machine.
        """
        filters = filters or []
        if not_in is not None: filters += [lambda x: all([str(notin) not in str(x) for notin in not_in])]
        if exts is not None: filters += [lambda x: any([ext in x.name for ext in exts])]
        slf = self.expanduser().resolve()
        if compressed and slf.suffix == ".zip":
            with __import__("zipfile").ZipFile(str(slf)) as z: content = List(z.namelist())
            raw = content.filter(lambda x: __import__("fnmatch").fnmatch(x, pattern)).apply(lambda x: slf / x)
        elif dotfiles: raw = slf.glob(pattern) if not r else self.rglob(pattern)
        else: raw = __import__("glob").glob(str(slf / "**" / pattern), recursive=r) if r else __import__("glob").glob(str(slf.joinpath(pattern)))  # glob ignroes dot and hidden files
        if compressed:
            comp_files = List(raw).filter(lambda x: '.zip' in str(x))
            for comp_file in comp_files: raw += P(comp_file).search(pattern=pattern, r=r, files=files, folders=folders, compressed=compressed, dotfiles=dotfiles, filters=filters, not_in=not_in, win_order=win_order)
        processed = List([P(item) for item in raw if (lambda item_: all([item_.is_dir() if not files else True, item_.is_file() if not folders else True] + [afilter(item_) for afilter in filters]))(P(item))])
        return processed if not win_order else processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in __import__("re").split('([0-9]+)', x.stem)])

    def tree(self, level: int = -1, limit_to_directories: bool = False, length_limit: int = 1000, stats=False, desc=None):
        """Based on: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python"""
        space, branch, tee, last, dir_path, files, directories = '    ', '│   ', '├── ', '└── ', self, 0, 0

        def get_stats(apath):
            if stats or desc:
                sts = apath.stats(printit=False)
                result = f" {sts.size} MB. {sts.content_mod_time}. "
                if desc is not None: result += desc(apath)
                return result
            return ""

        def inner(apath: P, prefix: str = '', level_=-1):
            nonlocal files, directories
            if not level_: return  # 0, stop iterating
            content = apath.search("*", files=not limit_to_directories)
            pointers = [tee] * (len(content) - 1) + [last]
            for pointer, path in zip(pointers, content):
                if path.is_dir():
                    yield prefix + pointer + path.name + get_stats(path)
                    directories += 1
                    extension = branch if pointer == tee else space
                    yield from inner(path, prefix=prefix + extension, level_=level_ - 1)
                elif not limit_to_directories:
                    yield prefix + pointer + path.name + get_stats(path)
                    files += 1
        print(dir_path.name)
        iterator = inner(dir_path, level_=level)
        for line in __import__("itertools").islice(iterator, length_limit): print(line)
        if next(iterator, None): print(f'... length_limit, {length_limit}, reached, counted:')
        print(f'\n{directories} directories' + (f', {files} files' if files else ''))

    def find(self, *args, r=True, compressed=True, **kwargs):
        """short for the method ``search`` then pick first item from results. useful for superflous directories or zip archives containing a single file."""
        if compressed is False and self.is_file(): return self
        results = self.search(*args, r=r, compressed=compressed, **kwargs)
        if len(results) > 0: return results[0].unzip() if ".zip" in str(results[0]) else results[0]

    def create(self, parents=True, exist_ok=True, parent_only=False): self.parent.mkdir(parents=parents, exist_ok=exist_ok) if parent_only else self.mkdir(parents=parents, exist_ok=exist_ok); return self
    browse = property(lambda self: self.search("*").to_struct(key_val=lambda x: ("qq_" + validate_name(x), x)).clean_view)
    pwd = staticmethod(lambda: P.cwd())
    tempdir = staticmethod(lambda: P(__import__("tempfile").mktemp()))
    temp = staticmethod(lambda: P(__import__("tempfile").gettempdir()))
    tmpdir = staticmethod(lambda prefix="": P.tmp(folder=rf"tmp_dirs/{prefix + ('_' if prefix != '' else '') + randstr()}"))
    def chdir(self): __import__("os").chdir(str(self.expanduser())); return self
    def listdir(self): return List(__import__("os").listdir(self.expanduser().resolve())).apply(P)
    @staticmethod
    def tmpfile(name=None, suffix="", folder=None, tstamp=False): return P.tmp(file=(name or randstr()) + "_" + randstr() + (("_" + timestamp()) if tstamp else "") + suffix, folder=folder or "tmp_files")

    @staticmethod
    def tmp(folder=None, file=None, root="~/tmp_results", verbose=False):
        root = P(root).expanduser().create()
        if folder is not None: root = (root / folder).create()
        if file is not None: root = root / file
        if verbose: print(f"TMPDIR {repr(root)}. Parent: {repr(root.parent)}")
        return root

    # ====================================== Compression ===========================================
    def zip(self, path=None, folder=None, name=None, arcname=None, inplace=False, verbose=True, content=True, orig=False, **kwargs):
        """"""
        path = self._resolve_path(folder, name, path, self.name).expanduser().resolve()
        slf = self.expanduser().resolve()
        arcname = P(arcname or slf.name)
        if arcname.name != slf.name: arcname /= slf.name  # arcname has to start from somewhere and end with filename
        if slf.is_file(): Compression.zip_file(ip_path=slf, op_path=path + f".zip" if path.suffix != ".zip" else path, arcname=arcname, **kwargs)
        else:
            root_dir, base_dir = (slf, ".") if content else (slf.split(at=str(arcname[0]))[0], arcname)
            path = Compression.compress_folder(root_dir=root_dir, op_path=path, base_dir=base_dir, fmt='zip', **kwargs)
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"ZIPPED {repr(slf)} ==>  {repr(path)}")

    def unzip(self, folder=None, fname=None, verbose=True, content=False, inplace=False, orig=False, **kwargs):
        """
        :param orig:
        :param folder: directory where extracted files will live.
        :param fname: a specific file path to be extracted from the archive.
        :param verbose:
        :param content: if set to True, all contents of the zip archive will be scattered in path dir.
        If set to False, a directory with same path as the zip file will be created and will contain the results.
        :param inplace: delete the original zip file after successful extraction.
        :param kwargs:
        :return: path if content=False, else, path.parent. Default path = self.parent / self.stem
        """
        slf = zipfile = self.expanduser().resolve()
        if slf.suffix != ".zip":  # may be there is .zip somewhere in the path.
            if ".zip" not in str(slf): return slf
            zipfile, fname = slf.split(at=List(slf.parts).filter(lambda x: ".zip" in x)[0], sep=-1)
        folder = (zipfile.parent / zipfile.stem) if folder is None else P(folder).joinpath(zipfile.stem).expanduser().resolve()
        result = Compression.unzip(zipfile, folder if not content else folder.parent, fname, **kwargs)
        return self._return(result, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNZIPPED {repr(zipfile)} ==> {repr(result)}")

    def tar(self, path=None): return Compression.untar(self, op_path=path or (self + '.gz'))
    def untar(self, path, verbose=True): _ = self, path, verbose; return P()
    def gz(self, path, verbose=True): _ = self, path, verbose; return P()
    def ungz(self, path, verbose=True): _ = self, path, verbose; return P()
    def tar_gz(self): pass

    def untar_ungz(self, folder=None, inplace=False, verbose=True, orig=False):
        folder = folder or P(self.parent) / P(self.stem)
        intrem = self.ungz(path=folder, verbose=verbose)
        result = intrem.untar(path=folder, verbose=verbose)
        intrem.delete(sure=True, verbose=verbose)
        return self._return(result, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"UNTARED-UNGZED {repr(self)} ==>  {repr(result)}")

    def compress(self, path=None, base_dir=None, fmt="zip", inplace=False, **kwargs):
        fmts = ["zip", "tar", "gzip"]
        assert fmt in fmts, f"Unsupported format {fmt}. The supported formats are {fmts}"
        _ = self, path, base_dir, kwargs, inplace
        pass

    def decompress(self): pass

    def encrypt(self, key=None, pwd=None, folder=None, name=None, path=None, verbose=True, append="_encrypted", inplace=False, orig=False, use_7z=False):
        """see: https://stackoverflow.com/questions/42568262/how-to-encrypt-text-with-a-password-in-python & https://stackoverflow.com/questions/2490334/simple-way-to-encode-a-string-according-to-a-password"""
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.append(name=append).name)
        assert slf.is_file(), f"Cannot encrypt a directory. You might want to try `zip_n_encrypt`. {self}"
        if use_7z:
            import crocodile.environment as env
            path = path + '.7z' if not path.suffix == '.7z' else path
            if env.system == "Windows":
                program = env.ProgramFiles.joinpath("7-Zip/7z.exe"); env.tm.run('winget install --name "7-zip" --Id "7zip.7zip" --source winget', shell="powershell") if not program.exists() else None
                env.tm.run(f"&'{program}' a '{path}' '{self}' -p{pwd}", shell="powershell")
            else: raise NotImplementedError("7z not implemented for Linux")
        else: path.write_bytes(encrypt(msg=slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"ENCRYPTED: {repr(slf)} ==> {repr(path)}.")

    def decrypt(self, key=None, pwd=None, path=None, folder=None, name=None, verbose=True, append="_encrypted", inplace=False, orig=False):
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.switch(append, "").name).write_bytes(decrypt(slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"DECRYPTED: {repr(slf)} ==> {repr(path)}.")

    def zip_n_encrypt(self, key=None, pwd=None, inplace=False, verbose=True, orig=False): return self.zip(inplace=inplace, verbose=verbose).encrypt(key=key, pwd=pwd, verbose=verbose, inplace=True) if not orig else self
    def decrypt_n_unzip(self, key=None, pwd=None, inplace=False, verbose=True, orig=False): return self.decrypt(key=key, pwd=pwd, verbose=verbose, inplace=inplace).unzip(folder=None, inplace=True, content=False) if not orig else self

    # ========================== Helpers =========================================
    def _resolve_path(self, folder, name, path, default_name, rel2it=False):
        """From all arguments, figure out what is the final path.
        :param rel2it: `folder` or `path` are relative to `self` as opposed to cwd."""
        if path is not None:
            assert folder is None and name is None, f"If `path` is passed, `folder` and `name` cannot be passed."
            path = P(self.joinpath(path).resolve() if rel2it else path).expanduser().resolve()
            assert not path.is_dir(), f"`path` passed is a directory! it must not be that. If this is meant, pass it with `path` kwarg. {path}"
            folder, name = path.parent, path.name; _ = name, folder
        else:
            name, folder = (default_name if name is None else str(name)), (self.parent if folder is None else folder)  # good for edge cases of path with single part.  # means same directory, just different name
            path = P(self.joinpath(folder).resolve() if rel2it else folder).expanduser().resolve() / name
        return path


class Compression(object):
    """Provides consistent behaviour across all methods. Both files and folders when compressed, default is being under the root of archive."""
    @staticmethod
    def compress_folder(root_dir, op_path, base_dir, fmt='zip', **kwargs):
        """shutil works with folders nicely (recursion is done interally) # directory to be archived: root_dir\base_dir, unless base_dir is passed as absolute path. # when archive opened; base_dir will be found."""
        assert fmt in {"zip", "tar", "gztar", "bztar", "xztar"} and P(op_path).suffix != ".zip", f"Don't add zip extention to this method, it is added automatically."
        return P(__import__('shutil').make_archive(base_name=str(op_path), format=fmt, root_dir=str(root_dir), base_dir=str(base_dir), **kwargs))  # returned path possible have added extension.

    @staticmethod
    def zip_file(ip_path, op_path, arcname=None, password=None, **kwargs):
        """
        arcname determines the directory of the file being archived inside the archive. Defaults to same
        as original directory except for drive. When changed, it should still include the file path in its end.
        If arcname = filename without any path, then, it will be in the root of the archive.
        """
        import zipfile
        with zipfile.ZipFile(str(op_path), 'w') as jungle_zip:
            jungle_zip.setpassword(pwd=password) if password is not None else None
            jungle_zip.write(filename=str(ip_path), arcname=str(arcname) if arcname is not None else None, compress_type=zipfile.ZIP_DEFLATED, **kwargs)
        return op_path

    @staticmethod
    def unzip(ip_path, op_path, fname=None, password=None, **kwargs):
        with __import__("zipfile").ZipFile(str(ip_path), 'r') as zipObj:
            if fname is None: zipObj.extractall(op_path, pwd=password, **kwargs)
            else: zipObj.extract(member=str(fname), path=str(op_path), pwd=password); op_path = P(op_path) / fname
        return P(op_path)

    @staticmethod
    def gz(file, op_file):
        with open(file, 'rb') as f_in:
            with __import__("gzip").open(op_file, 'wb') as f_out:  __import__("shutil").copyfileobj(f_in, f_out)
        return op_file

    @staticmethod
    def ungz(self, op_path=None):
        with __import__("gzip").open(str(self), 'r') as f_in, open(op_path, 'wb') as f_out: __import__("shutil").copyfileobj(f_in, f_out)
        return P(op_path)

    @staticmethod
    def tar(self, op_path):
        with __import__("tarfile").open(op_path, "w:gz") as tar: tar.add(str(self), arcname=__import__("os").path.basename(str(self)))
        return op_path

    @staticmethod
    def untar(self, op_path, fname=None, mode='r', **kwargs):
        with __import__("tarfile").open(str(self), mode) as file:
            if fname is None: file.extractall(path=op_path, **kwargs)  # extract all files in the archive
            else: file.extract(fname, **kwargs)
        return P(op_path)


class MemoryDB:
    """This class holds the historical data. It acts like a database, except that is memory based."""
    def __init__(self, size=5): self.size, self.list = size, List()
    def __repr__(self): return f"MemoryDB. Size={self.size}. Current length = {self.len}"
    def __getitem__(self, item): return self.list[item]
    len = property(lambda self: len(self.list))
    def append(self, item): self.list.append(item); self.list = self.list[-self.size:] if self.len > self.size else self.list  # take latest frames and drop the older ones.


class Fridge:
    """This class helps to accelrate access to latest data coming from expensive function. The class has two flavours, memory-based and disk-based variants."""
    def __init__(self, source_func, expire="1m", time_produced=None, logger=None, path=None, save=Save.pickle, read=Read.read):
        self.cache = None  # fridge content
        self.time_produced = time_produced or datetime.now()  # init time_produced
        self.source_func = source_func  # function which when called returns a fresh object to be frozen.
        self.path = P(path) if path else None  # if path is passed, it will function as disk-based flavour.
        self.save, self.read, self.logger, self.expire = save, read, logger, expire

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.path is not None: state["path"] = self.path.rel2home()  # With this implementation, instances can be pickled and loaded up in different machine and still works.
        return state

    def __setstate__(self, state): self.__dict__.update(state); self.path = P.home() / self.path if self.path is not None else self.path
    age = property(lambda self: datetime.now() - self.time_produced if self.path is None else datetime.now() - self.path.stats().content_mod_time)
    def reset(self): self.time_produced = datetime.now()

    def __call__(self, fresh=False):
        if self.path is None:  # Memory Fridge
            if self.cache is None or fresh is True or self.age > str2timedelta(self.expire):
                self.cache, self.time_produced = self.source_func(), datetime.now()
                if self.logger: self.logger.debug(f"Updating / Saving data from {self.source_func}")
            elif self.logger: self.logger.debug(f"Using cached values. Lag = {self.age}.")
        elif fresh or not self.path.exists() or self.age > str2timedelta(self.expire):  # disk fridge
            if self.logger: self.logger.debug(f"Updating & Saving {self.path} ...")
            self.cache = self.source_func()  # fresh order, never existed or exists but expired.
            self.save(obj=self.cache, path=self.path)
        elif self.age < str2timedelta(self.expire) and self.cache is None: self.cache = self.read(self.path)  # this implementation favours reading over pulling fresh at instantiation.  # exists and not expired. else # use the one in memory self.cache
        return self.cache


if __name__ == '__main__':
    pass
