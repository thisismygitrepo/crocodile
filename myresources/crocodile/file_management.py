
from crocodile.core import Struct, List, timestamp, randstr, validate_name, str2timedelta, Save, Path, install_n_import
from datetime import datetime


# %% =============================== Security ================================================
def obscure(msg: bytes) -> bytes: return __import__("base64").urlsafe_b64encode(__import__("zlib").compress(msg, 9))
def unobscure(obscured: bytes) -> bytes: return __import__("zlib").decompress(__import__("base64").urlsafe_b64decode(obscured))
def pwd2key(password: str, salt=None, iterations=None) -> bytes:  # Derive a secret key from a given password and salt"""
    if salt is None: m = __import__("hashlib").sha256(); m.update(password.encode("utf-8")); return __import__("base64").urlsafe_b64encode(m.digest())  # make url-safe bytes required by Ferent.
    from cryptography.hazmat.primitives import hashes; from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    return __import__("base64").urlsafe_b64encode(PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=None).derive(password.encode()))
def encrypt(msg: bytes, key=None, pwd: str = None, salted=True, iteration: int = None) -> bytes:
    salt = None  # silence the linter.
    if pwd is not None:  # generate it from password
        assert (key is None) and (type(pwd) is str), f"You can either pass key or pwd, or none of them, but not both."
        salt, iteration = (__import__('secrets').token_bytes(16), iteration or __import__('secrets').randbelow(1_000_000)) if salted else (None, None); key = pwd2key(pwd, salt, iteration)
    elif key is None: key = __import__("cryptography.fernet").__dict__["fernet"].Fernet.generate_key(); print(f"KEY SAVED @ {repr(P.tmpdir().joinpath('key.bytes').write_bytes(key))}")  # discouraged, make your keys/pwd before invoking the func. use random bytes, more secure but no string representation
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # a path to a key file was passed, read it:
    elif type(key) is bytes: pass  # key passed explicitly
    else: raise TypeError(f"Key must be either a path, bytes object or None.")
    code = __import__("cryptography.fernet").__dict__["fernet"].Fernet(key).encrypt(msg)
    return __import__("base64").urlsafe_b64encode(b'%b%b%b' % (salt, iteration.to_bytes(4, 'big'), __import__("base64").urlsafe_b64decode(code))) if pwd is not None and salted is True else code
def decrypt(token: bytes, key=None, pwd: str = None, salted=True) -> bytes:
    if pwd is not None:
        assert key is None, f"You can either pass key or pwd, or none of them, but not both."
        if salted:
            decoded = __import__("base64").urlsafe_b64decode(token); salt, iterations, token = decoded[:16], decoded[16:20], __import__("base64").urlsafe_b64encode(decoded[20:])
            key = pwd2key(pwd, salt, int.from_bytes(iterations, 'big'))
        else: key = pwd2key(pwd)  # trailing `;` prevents IPython from caching the result.
    if type(key) is bytes: pass   # passsed explicitly
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # passed a path to a file containing kwy
    else: raise TypeError(f"Key must be either str, P, Path or bytes.")
    return __import__("cryptography.fernet").__dict__["fernet"].Fernet(key).decrypt(token)


# %% =================================== File ============================================
def read(path, **kwargs):
    suffix = Path(path).suffix[1:]
    try: return getattr(Read, suffix)(str(path), **kwargs)
    except AttributeError as err:
        if "type object 'Read' has no attribute" not in str(err): raise AttributeError(err)
        if suffix in ('eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'): return __import__("matplotlib").pyplot.imread(path, **kwargs)  # from: plt.gcf().canvas.get_supported_filetypes().keys():
        raise AttributeError(f"Unknown file type. failed to recognize the suffix `{suffix}`")
def json(path, r=False, **kwargs):
    try: mydict = __import__("json").loads(P(path).read_text(), **kwargs)
    except Exception: mydict = install_n_import("pyjson5").loads(P(path).read_text(), **kwargs)  # file has C-style comments.
    return Struct.recursive_struct(mydict) if r else Struct(mydict)
def yaml(path, r=False):
    import yaml as yaml_
    with open(str(path), "r") as file: mydict = yaml_.load(file, Loader=yaml_.FullLoader)
    return Struct(mydict) if not r else Struct.recursive_struct(mydict)
def npy(path, **kwargs): data = (np := __import__("numpy")).load(str(path), allow_pickle=True, **kwargs); data = data.item() if data.dtype == np.object else data; return Struct(data) if type(data) is dict else data
def mat(path, remove_meta=False, **kwargs): res = Struct(__import__("scipy.io").__dict__["io"].loadmat(path, **kwargs)); List(res.keys()).filter("x.startswith('__')").apply(lambda x: res.__delattr__(x)) if remove_meta else None; return res
def csv(path, **kwargs): return __import__("pandas").read_csv(path, **kwargs)
def py(path): return Struct(__import__("runpy").run_path(path))
def pickles(bytes_obj): return __import__("dill").loads(bytes_obj)  # handles imports automatically provided that saved object was from an imported class (not in defined in __main__)
def pickle(path, **kwargs): obj = __import__("dill").loads(P(path).read_bytes(), **kwargs); return Struct(obj) if type(obj) is dict else obj
def pkl(*args, **kwargs): return pickle(*args, **kwargs)
class Read: read = read; mat = mat; json = json; yaml = yaml; npy = npy; csv = csv; pkl = pkl; py = py; pickle = pickle; txt = lambda path, encoding=None: P(path).read_text(encoding=encoding)


class P(type(Path()), Path):
    # ============= Path management ==================
    """ The default behaviour of methods acting on underlying disk object is to perform the action and return a new path referring to the mutated object in disk drive.
    However, there is a flag `orig` that makes the function return orignal path object `self` as opposed to the new one pointing to new object.
    Additionally, the fate of the original object can be decided by a flag `inplace` which means `replace` it defaults to False and in essence, it deletes the original underlying object.
    This can be seen in `zip` and `encrypt` but not in `copy`, `move`, `retitle` because the fate of original file is dictated already.
    Furthermore, those methods are accompanied with print statement explaining what happened to the object."""
    def delete(self, sure=False, verbose=True):
        slf = self  # slf = self.expanduser().resolve() don't resolve symlinks.
        if not sure: print(f"Did NOT DELETE because user is not sure. file: {repr(slf)}.") if verbose else None; return self
        if not slf.exists(): slf.unlink(missing_ok=True); print(f"Could NOT DELETE nonexisting file {repr(slf)}. ") if verbose else None; return slf  # broken symlinks exhibit funny existence behaviour, catch them here.
        slf.unlink(missing_ok=True) if slf.is_file() or slf.is_symlink() else __import__("shutil").rmtree(slf, ignore_errors=False); print(f"DELETED {repr(slf)}.") if verbose else None; return self
    def send2trash(self, verbose=True):
        if self.exists(): install_n_import("send2trash").send2trash(self.resolve().str); print(f"TRASHED {repr(self)}") if verbose else None  # do not expand user symlinks.
        elif verbose: print(f"Could NOT trash {self}"); return self
    def move(self, folder=None, name=None, path=None, rel2it=False, overwrite=False, verbose=True, parents=True, content=False):
        path = self._resolve_path(folder=folder, name=name, path=path, default_name=self.absolute().name, rel2it=rel2it)
        path.parent.create(parents=True, exist_ok=True) if parents else None; slf = self.expanduser().resolve()
        if content:
            assert self.is_dir(), NotADirectoryError(f"When `content` flag is set to True, path must be a directory. It is not: `{repr(self)}`")
            self.search("*").apply(lambda x: x.move(folder=path.parent, content=False, overwrite=overwrite)); return path  # contents live within this directory.
        if overwrite: tmp_path = slf.rename(path.parent.absolute() / randstr()); path.delete(sure=True, verbose=verbose); tmp_path.rename(path)  # works if moving a path up and parent has same name
        else: slf.rename(path)  # self._return(res=path, inplace=True, operation='rename', orig=False, verbose=verbose, strict=True, msg='')
        print(f"MOVED {repr(self)} ==> {repr(path)}`") if verbose else None; return path
    def copy(self, folder=None, name=None, path=None, content=False, verbose=True, append=f"_copy_{randstr()}", overwrite=False, orig=False):  # tested %100  # TODO: replace `content` flag with ability to interpret "*" in resolve method.
        dest = self._resolve_path(folder=folder, name=name, path=path, default_name=self.name, rel2it=False)
        dest, slf = dest.expanduser().resolve().create(parents_only=True), self.expanduser().resolve(); dest = self.append(append) if dest == self else dest
        dest.delete(sure=True) if not content and overwrite and dest.exists() else None
        if not content and not overwrite and dest.exists(): raise FileExistsError(f"Destination already exists: {repr(dest)}")
        if slf.is_file(): __import__("shutil").copy(str(slf), str(dest)); print(f"COPIED {repr(slf)} ==> {repr(dest)}") if verbose else None
        elif slf.is_dir(): dest = dest.parent if content else dest; __import__("distutils.dir_util").__dict__["dir_util"].copy_tree(str(slf), str(dest)); print(f"COPIED {'Content of ' if False else ''} {repr(slf)} ==> {repr(dest)}") if verbose else None
        else: print(f"Could NOT COPY. Not a file nor a path: {repr(slf)}.")
        return dest if not orig else self
    # ======================================= File Editing / Reading ===================================
    def readit(self, reader=None, strict=True, notfound=None, verbose=False, **kwargs):
        if not self.exists():
            if strict: raise FileNotFoundError(f"`{self}` is no where to be found!")
            else: return notfound
        filename = self.unzip(folder=self.tmp(folder="tmp_unzipped"), verbose=verbose) if '.zip' in str(self) else self
        try: return Read.read(filename, **kwargs) if reader is None else reader(str(filename), **kwargs)
        except IOError: raise IOError
    def start(self, opener=None):
        if str(self).startswith("http") or str(self).startswith("www"): __import__("webbrowser").open(str(self)); return self
        if __import__("sys").platform == "win32":  # double quotes fail with cmd. # __import__("os").startfile(filename)  # works for files and folders alike, but if opener is given, e.g. opener="start"
            __import__("subprocess").Popen(f"powershell start '{self.expanduser().resolve().str}'" if opener is None else rf'powershell {opener} \'{self}\''); return self  # fails for folders. Start must be passed, but is not defined.
        elif __import__("sys").platform == 'linux': __import__("subprocess").call(["xdg-open", self.expanduser().resolve().str]); return self  # works for files and folders alike
        else:  __import__("subprocess").call(["open", self.expanduser().resolve().str]); return self  # works for files and folders alike  # mac
    def __call__(self, *args, **kwargs): self.start(*args, **kwargs); return self
    def append_text(self, appendix): self.write_text(self.read_text() + appendix); return self
    def read_fresh_from(self, source_func, expire="1w", save=Save.pickle, reader=Read.read, **kwargs): return Fridge(source_func=source_func, path=self, expire=expire, save=save, reader=reader, **kwargs)
    def modify_text(self, txt, alt, newline=False, notfound_append=False, encoding=None):
        if not self.exists(): self.create(parents_only=True).write_text(txt)
        lines, bingo = self.read_text(encoding=encoding).split("\n"), False
        for idx, line in enumerate(lines):
            if txt in line: lines[idx], bingo = (alt if type(alt) is str else alt(line)) if newline is True else line.replace(txt, alt if type(alt) is str else alt(line)), True
        if bingo is False and notfound_append is True: lines.append(alt)  # txt not found, add it anyway.
        return self.write_text("\n".join(lines), encoding=encoding)
    def download(self, directory=None, name=None, memory=False, allow_redirects=True, params=None):
        response = __import__("requests").get(self.as_url_str(), allow_redirects=allow_redirects, params=params)  # Alternative: from urllib import request; request.urlopen(url).read().decode('utf-8').
        return response if memory else (P.home().joinpath("Downloads") if directory is None else P(directory)).joinpath(name or self.name).create(parents_only=True).write_bytes(response.content)  # r.contents is bytes encoded as per docs of requests.
    def _return(self, res, inlieu=False, inplace=False, operation=None, overwrite=False, orig=False, verbose=False, strict=True, msg="", __delayed_msg__=""):
        if inlieu: self._str = str(res)
        if inplace:
            assert self.exists(), f"`inplace` flag is only relevant if the path exists. It doesn't {self}"
            if operation == "rename":
                if overwrite and res.exists(): res.delete(sure=True, verbose=verbose)
                if not overwrite and res.exists():
                    if strict: raise FileExistsError(f"File {res} already exists.")
                    else: print(f"SKIPPED RENAMING {repr(self)} ==> {repr(res)} because FileExistsError and scrict=False policy.") if verbose else None; return self if orig else res
                self.rename(res); msg = msg or f"RENAMED {repr(self)} ==> {repr(res)}"
            elif operation == "delete": self.delete(sure=True, verbose=False);  __delayed_msg__ = f"DELETED {repr(self)}."
        print(msg) if verbose and msg != "" else None; print(__delayed_msg__) if verbose and __delayed_msg__ != "" else None; return self if orig else res
    # ================================ Path Object management ===========================================
    """ Distinction between Path object and the underlying file on disk that the path may refer to. Two distinct flags are used:
        `inplace`: the operation on the path object will affect the underlying file on disk if this flag is raised, otherwise the method will only alter the string.
        `inliue`: the method acts on the path object itself instead of creating a new one if this flag is raised.
        `orig`: whether the method returns the original path object or a new one."""
    def prepend(self, prefix, suffix=None, **kwargs): return self._return(self.parent.joinpath(prefix + self.trunk + (suffix or ''.join(('bruh'+self).suffixes))), operation="rename", **kwargs)  # Path('.ssh').suffix fails, 'bruh' fixes it.
    def append(self, name='', suffix=None, **kwargs): return self._return(self.parent.joinpath(self.trunk + (name or "_" + timestamp()) + (suffix or ''.join(('bruh'+self).suffixes))), operation="rename", **kwargs)
    def with_trunk(self, name, **kwargs): return self._return(self.parent.joinpath(name + "".join(self.suffixes)), operation="rename", **kwargs)  # Complementary to `with_stem` and `with_suffix`
    def with_name(self, name, verbose=True, **kwargs): assert type(name) is str, "name must be a string."; return self._return(self.parent / name, verbose=verbose, operation="rename", **kwargs)
    def switch(self, key: str, val: str, **kwargs): return self._return(P(str(self).replace(key, val)), operation="rename", **kwargs)  # Like string replce method, but `replace` is an already defined method."""
    def switch_by_index(self, idx: int, val: str, **kwargs): return self._return(P(*[val if index == idx else value for index, value in enumerate(self.parts)]), operation="rename", **kwargs)
    # ============================= attributes of object ======================================
    trunk = property(lambda self: self.name.split('.')[0])  # """ useful if you have multiple dots in file path where `.stem` fails."""
    len = property(lambda self: self.__len__()); items = property(lambda self: List(self.parts)); str = property(lambda self: str(self))  # or self._str
    def __len__(self): return len(self.parts)
    def __contains__(self, item): return item in self.parts
    def __iter__(self): return self.parts.__iter__()
    def __deepcopy__(self): return P(str(self))
    def __getstate__(self): return str(self)
    def __setstate__(self, state): self._str = str(state)
    def __add__(self, other): return self.parent.joinpath(self.name + str(other))  # used append and prepend if the addition wanted to be before suffix.
    def __radd__(self, other): return self.parent.joinpath(str(other) + self.name)  # other + P and `other` doesn't know how to make this addition.
    def __sub__(self, other): res = P(str(self).replace(str(other), "")); return res[1:] if str(res[0]) in {"\\", "/"} else res  # paths starting with "/" are problematic. e.g ~ / "/path" doesn't work.
    def rel2cwd(self, inlieu=False): return self._return(P(self.expanduser().absolute().relative_to(Path.cwd())), inlieu)
    def rel2home(self, inlieu=False): return self._return(P(self.expanduser().absolute().relative_to(Path.home())), inlieu)  # very similat to collapseuser but without "~" being added so its consistent with rel2cwd.
    def collapseuser(self, strict=True): assert str(P.home()) in str(self.expanduser()), ValueError(f"{str(P.home())} is not in the subpath of {str(self)}") if strict else None; return self if "~" in self else self._return("~" / (self - P.home()))    # opposite of `expanduser`
    def __getitem__(self, slici): return P(*[self[item] for item in slici]) if type(slici) is list else (P(*self.parts[slici]) if type(slici) is slice else P(self.parts[slici]))  # it is an integer
    def __setitem__(self, key: str or int or slice, value: str or Path):
        fullparts, new = list(self.parts), list(P(value).parts)
        if type(key) is str: idx = fullparts.index(key); fullparts.remove(key); fullparts = fullparts[:idx] + new + fullparts[idx + 1:]
        elif type(key) is int: fullparts = fullparts[:key] + new + fullparts[key + 1:]
        elif type(key) is slice: fullparts = fullparts[:(0 if key.start is None else key.start)] + new + fullparts[(len(fullparts) if key.stop is None else key.stop):]
        self._str = str(P(*fullparts))  # similar attributes: # self._parts # self._pparts # self._cparts # self._cached_cparts
    def split(self, at: str = None, index: int = None, sep=[-1, 0, 1][-1], strict=True):
        if index is None and (at is not None):  # at is provided  # ====================================   Splitting
            if not strict:  # bevaes like split method of string
                one, two = (items := str(self).split(sep=str(at)))[0], items[1]; one, two = P(one[:-1]) if one.endswith("/") else P(one), P(two[1:]) if two.startswith("/") else P(two)
            else:  # "strict": # raises an error if exact match is not found.
                index = self.parts.index(str(at)); one, two = self[0:index], self[index + 1:]  # both one and two do not include the split item.
        elif index is not None and (at is None):  # index is provided
            one, two = self[:index], P(*self.parts[index + 1:]); at = self[index]  # this is needed below.
        else: raise ValueError("Either `index` or `at` can be provided. Both are not allowed simulatanesouly.")
        if sep == 0: return one, two  # neither of the portions get the sperator appended to it. # ================================  appending `at` to one of the portions
        elif sep == 1: return one, at / two   # append it to right portion
        elif sep == -1: return one / at, two  # append it to left portion.
        else: raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")
    def __repr__(self):  # this is useful only for the console
        if self.is_symlink():
            try: target = self.resolve()  # broken symolinks are funny, and almost always fail `resolve` method.
            except Exception: target = "BROKEN LINK " + str(self)  # avoid infinite recursions for broken links.
            return "P: Symlink '" + str(self) + "' ==> " + repr(str(target) if target == self else target)
        elif self.is_absolute(): return "P: " + self._type() + " '" + self.clickable() + "'" + (" | " + self.time(which="c").isoformat()[:-7].replace("T", "  ") if self.exists() else "") + (f" | {self.size()} Mb" if self.is_file() else "")
        elif "http" in str(self): return "P: URL " + self.as_url_str()
        else: return "P: Relative " + "'" + str(self) + "'"  # not much can be said about a relative path.
    def size(self, units='mb'):  # ===================================== File Specs ==========================================================================================
        total_size = self.stat().st_size if self.is_file() else sum([item.stat().st_size for item in self.rglob("*") if item.is_file()])
        return round(total_size / dict(zip(List(['b', 'kb', 'mb', 'gb']).eval("self+self.swapcase()"), 2 * [1024 ** item for item in range(4)]))[units], 1)
    def time(self, which=["m", "c", "a"][0], **kwargs): return datetime.fromtimestamp({"m": self.stat().st_mtime, "a": self.stat().st_atime, "c": self.stat().st_ctime}[which], **kwargs)  # m last mofidication of content, i.e. the time it was created. c last status change (its inode is changed, permissions, path, but not content) a: last access
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
    def write_text(self, data: str, **kwargs) -> 'Path': super(P, self).write_text(data, **kwargs); return self
    def read_text(self, encoding=None, lines=False, printit=False): res = super(P, self).read_text(encoding=encoding) if not lines else List(super(P, self).read_text(encoding=encoding).splitlines()); print(res) if printit else None; return res
    def write_bytes(self, data: bytes): res = super(P, self).write_bytes(data); print(f"Could not save file on disk.") if res == 0 else None; return self
    def touch(self, mode: int = 0o666, parents=True, exist_ok: bool = ...): self.parent.create(parents=parents) if parents else None; super(P, self).touch(mode=mode, exist_ok=exist_ok); return self
    def symlink_from(self, folder=None, file=None, verbose=False, overwrite=False):
        assert self.expanduser().exists(), "self must exist if this method is used."
        if file is not None: assert folder is None, "You can only pass source or source_dir, not both."; result = P(file).expanduser().absolute()
        else: result = P(folder or P.cwd()).expanduser().absolute() / self.name
        return result.symlink_to(self, verbose=verbose, overwrite=overwrite)
    def symlink_to(self, target=None, verbose=True, overwrite=False, orig=False):
        self.parent.create(); assert (target := P(target).expanduser().resolve()).exists(), f"Target path `{target}` doesn't exist. This will create a broken link."
        if overwrite and (self.is_symlink() or self.exists()): self.delete(sure=True, verbose=verbose)
        if __import__("platform").system() == "Windows" and not (tm := __import__("crocodile").meta.Terminal).is_user_admin():  # you cannot create symlink without priviliages.
            tm.run_code_as_admin(f" -c \"from pathlib import Path; Path(r'{self.expanduser()}').symlink_to(r'{str(target)}')\""); __import__("time").sleep(0.5)  # give time_produced for asynch process to conclude before returning response.
        else: super(P, self.expanduser()).symlink_to(str(target))
        return self._return(P(target), inlieu=False, inplace=False, orig=orig, verbose=verbose, msg=f"LINKED {repr(self)}")
    def resolve(self, strict=False):
        try: return super(P, self).resolve(strict=strict)
        except OSError: return self
    # ======================================== Folder management =======================================
    def search(self, pattern='*', r=False, files=True, folders=True, compressed=False, dotfiles=False, filters: list = None, not_in: list = None, exts=None, win_order=False):
        filters = (filters or []) + ([lambda x: all([str(notin) not in str(x) for notin in not_in])] if not_in is not None else []) + ([lambda x: any([ext in x.name for ext in exts])] if exts is not None else [])
        if (slf := self.expanduser().resolve()).suffix == ".zip" and compressed:
            with __import__("zipfile").ZipFile(str(slf)) as z: content = List(z.namelist())
            raw = content.filter(lambda x: __import__("fnmatch").fnmatch(x, pattern)).apply(lambda x: slf / x)
        elif dotfiles: raw = slf.glob(pattern) if not r else self.rglob(pattern)
        else: raw = __import__("glob").glob(str(slf / "**" / pattern), recursive=r) if r else __import__("glob").glob(str(slf.joinpath(pattern)))  # glob ignroes dot and hidden files
        if compressed:
            comp_files = List(raw).filter(lambda x: '.zip' in str(x))
            for comp_file in comp_files: raw += P(comp_file).search(pattern=pattern, r=r, files=files, folders=folders, compressed=compressed, dotfiles=dotfiles, filters=filters, not_in=not_in, win_order=win_order)
        processed = List([P(item) for item in raw if (lambda item_: all([item_.is_dir() if not files else True, item_.is_file() if not folders else True] + [afilter(item_) for afilter in filters]))(P(item))])
        return processed if not win_order else processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in __import__("re").split('([0-9]+)', x.stem)])
    def tree(self, *args, **kwargs): return __import__("crocodile").msc.odds.__dict__['tree'](self, *args, **kwargs)
    def find(self, *args, r=True, compressed=True, **kwargs):  # short for the method ``search`` then pick first item from results. useful for superflous directories or zip archives containing a single file."""
        if compressed is False and self.is_file(): return self
        if len(results := self.search(*args, r=r, compressed=compressed, **kwargs)) > 0: return results[0].unzip() if ".zip" in str(results[0]) else results[0]
    browse = property(lambda self: self.search("*").to_struct(key_val=lambda x: ("qq_" + validate_name(x), x)).clean_view)
    def create(self, parents=True, exist_ok=True, parents_only=False): self.parent.mkdir(parents=parents, exist_ok=exist_ok) if parents_only else self.mkdir(parents=parents, exist_ok=exist_ok); return self
    def chdir(self): __import__("os").chdir(str(self.expanduser())); return self
    def listdir(self): return List(__import__("os").listdir(self.expanduser().resolve())).apply(P)
    pwd = staticmethod(lambda: P.cwd())
    tempdir = staticmethod(lambda: P(__import__("tempfile").mktemp()))
    temp = staticmethod(lambda: P(__import__("tempfile").gettempdir()))
    tmpdir = staticmethod(lambda prefix="": P.tmp(folder=rf"tmp_dirs/{prefix + ('_' if prefix != '' else '') + randstr()}"))
    tmpfile = staticmethod(lambda name=None, suffix="", folder=None, tstamp=False: P.tmp(file=(name or randstr()) + "_" + randstr() + (("_" + timestamp()) if tstamp else "") + suffix, folder=folder or "tmp_files"))
    tmp = staticmethod(lambda folder=None, file=None, root="~/tmp_results": P(root).expanduser().create().joinpath(folder or "").joinpath(file or "").create(parents_only=True if file else False))
    env = staticmethod(lambda: __import__("crocodile.environment").environment)
    # ====================================== Compression & Encryption ===========================================
    def zip(self, path=None, folder=None, name=None, arcname=None, inplace=False, verbose=True, content=True, orig=False, **kwargs):
        path, slf = self._resolve_path(folder, name, path, self.name).expanduser().resolve(), self.expanduser().resolve()
        arcname = P(arcname or slf.name)
        if arcname.name != slf.name: arcname /= slf.name  # arcname has to start from somewhere and end with filename
        if slf.is_file(): path = Compression.zip_file(ip_path=slf, op_path=path + f".zip" if path.suffix != ".zip" else path, arcname=arcname, **kwargs)
        else:
            root_dir, base_dir = (slf, ".") if content else (slf.split(at=str(arcname[0]))[0], arcname)
            path = Compression.compress_folder(root_dir=root_dir, op_path=path, base_dir=base_dir, fmt='zip', **kwargs)
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"ZIPPED {repr(slf)} ==>  {repr(path)}")
    def unzip(self, folder=None, fname=None, verbose=True, content=False, inplace=False, orig=False, **kwargs):
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
        assert fmt in (fmts := ["zip", "tar", "gzip"]), f"Unsupported format {fmt}. The supported formats are {fmts}"
        _ = self, path, base_dir, kwargs, inplace
        pass
    def decompress(self): pass
    def encrypt(self, key=None, pwd=None, folder=None, name=None, path=None, verbose=True, append="_encrypted", inplace=False, orig=False, use_7z=False):  # see: https://stackoverflow.com/questions/42568262/how-to-encrypt-text-with-a-password-in-python & https://stackoverflow.com/questions/2490334/simple-way-to-encode-a-string-according-to-a-password"""
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.append(name=append).name); assert slf.is_file(), f"Cannot encrypt a directory. You might want to try `zip_n_encrypt`. {self}"
        if use_7z and (env := P.env()).system == "Windows":
            env.tm.run('winget install --name "7-zip" --Id "7zip.7zip" --source winget', shell="powershell") if not (program := env.ProgramFiles.joinpath("7-Zip/7z.exe")).exists() else None
            path = path + '.7z' if not path.suffix == '.7z' else path; env.tm.run(f"&'{program}' a '{path}' '{self}' -p{pwd}", shell="powershell")
        elif use_7z: raise NotImplementedError("7z not implemented for Linux")
        else: path.write_bytes(encrypt(msg=slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, inlieu=False, inplace=inplace, operation="delete", orig=orig, verbose=verbose, msg=f"ENCRYPTED: {repr(slf)} ==> {repr(path)}.")
    def decrypt(self, key=None, pwd=None, path=None, folder=None, name=None, verbose=True, append="_encrypted", **kwargs):
        slf = self.expanduser().resolve(); path = self._resolve_path(folder, name, path, slf.name.replace(append, "") if "_encrypted" in slf.name else "decrypted_" + slf.name).write_bytes(decrypt(slf.read_bytes(), key=key, pwd=pwd))
        return self._return(path, operation="delete", verbose=verbose, msg=f"DECRYPTED: {repr(slf)} ==> {repr(path)}.", **kwargs)
    def zip_n_encrypt(self, key=None, pwd=None, inplace=False, verbose=True, orig=False): return self.zip(inplace=inplace, verbose=verbose).encrypt(key=key, pwd=pwd, verbose=verbose, inplace=True) if not orig else self
    def decrypt_n_unzip(self, key=None, pwd=None, inplace=False, verbose=True, orig=False): return self.decrypt(key=key, pwd=pwd, verbose=verbose, inplace=inplace).unzip(folder=None, inplace=True, content=False) if not orig else self
    def _resolve_path(self, folder, name, path, default_name, rel2it=False):  # From all arguments, figure out what is the final path.
        """:param rel2it: `folder` or `path` are relative to `self` as opposed to cwd. This is used when resolving '../dir'"""
        if path is not None:
            path = P(self.joinpath(path).resolve() if rel2it else path).expanduser().resolve()
            assert folder is None and name is None, f"If `path` is passed, `folder` and `name` cannot be passed."; assert not path.is_dir(), f"`path` passed is a directory! it must not be that. If this is meant, pass it with `folder` kwarg. `{path}`"
            return path
        name, folder = (default_name if name is None else str(name)), (self.parent if folder is None else folder)  # good for edge cases of path with single part.  # means same directory, just different name
        return P(self.joinpath(folder).resolve() if rel2it else folder).expanduser().resolve() / name


def compress_folder(root_dir, op_path, base_dir, fmt='zip', **kwargs):  # shutil works with folders nicely (recursion is done interally) # directory to be archived: root_dir\base_dir, unless base_dir is passed as absolute path. # when archive opened; base_dir will be found."""
    assert fmt in {"zip", "tar", "gztar", "bztar", "xztar"} and P(op_path).suffix != ".zip", f"Don't add zip extention to this method, it is added automatically."
    return P(__import__('shutil').make_archive(base_name=str(op_path), format=fmt, root_dir=str(root_dir), base_dir=str(base_dir), **kwargs))  # returned path possible have added extension.
def zip_file(ip_path, op_path, arcname=None, password=None, **kwargs):
    """arcname determines the directory of the file being archived inside the archive. Defaults to same as original directory except for drive.
    When changed, it should still include the file path in its end. If arcname = filename without any path, then, it will be in the root of the archive."""
    import zipfile
    with zipfile.ZipFile(str(op_path), 'w') as jungle_zip:
        jungle_zip.setpassword(pwd=password) if password is not None else None
        jungle_zip.write(filename=str(ip_path), arcname=str(arcname) if arcname is not None else None, compress_type=zipfile.ZIP_DEFLATED, **kwargs)
    return P(op_path)
def unzip(ip_path, op_path, fname=None, password=None, **kwargs):
    with __import__("zipfile").ZipFile(str(ip_path), 'r') as zipObj:
        if fname is None: zipObj.extractall(op_path, pwd=password, **kwargs)
        else: zipObj.extract(member=str(fname), path=str(op_path), pwd=password); op_path = P(op_path) / fname
    return P(op_path)
def gz(file, op_file):
    with open(file, 'rb') as f_in:
        with __import__("gzip").open(op_file, 'wb') as f_out:  __import__("shutil").copyfileobj(f_in, f_out)
    return P(op_file)
def ungz(self, op_path=None):
    with __import__("gzip").open(str(self), 'r') as f_in, open(op_path, 'wb') as f_out: __import__("shutil").copyfileobj(f_in, f_out)
    return P(op_path)
def tar(self, op_path):
    with __import__("tarfile").open(op_path, "w:gz") as tar_: tar_.add(str(self), arcname=__import__("os").path.basename(str(self)))
    return P(op_path)
def untar(self, op_path, fname=None, mode='r', **kwargs):
    with __import__("tarfile").open(str(self), mode) as file:
        if fname is None: file.extractall(path=op_path, **kwargs)  # extract all files in the archive
        else: file.extract(fname, **kwargs)
    return P(op_path)
class Compression: compress_folder = compress_folder; zip_file = zip_file; unzip = unzip; gz = gz; ungz = ungz; targ = tar; untar = untar  # Provides consistent behaviour across all methods. Both files and folders when compressed, default is being under the root of archive."""


class Fridge:  # This class helps to accelrate access to latest data coming from expensive function. The class has two flavours, memory-based and disk-based variants."""
    def __init__(self, source_func, expire="1m", logger=None, path=None, save=Save.pickle, reader=Read.read):
        self.cache = None  # fridge content
        self.source_func = source_func  # function which when called returns a fresh object to be frozen.
        self.path = P(path) if path else None  # if path is passed, it will function as disk-based flavour.
        self.time_produced = None
        self.save, self.reader, self.logger, self.expire = save, reader, logger, expire
    age = property(lambda self: datetime.now() - self.time_produced if self.path is None else datetime.now() - self.path.stats().content_mod_time)
    def __setstate__(self, state): self.__dict__.update(state); self.path = P.home() / self.path if self.path is not None else self.path
    def __getstate__(self): state = self.__dict__.copy(); state["path"] = self.path.rel2home() if self.path is not None else state["path"]; return state  # With this implementation, instances can be pickled and loaded up in different machine and still works.
    def __call__(self, fresh=False):
        if self.path is None:  # Memory Fridge
            if self.cache is None or fresh is True or self.age > str2timedelta(self.expire): self.cache, self.time_produced = self.source_func(), datetime.now(); self.logger.debug(f"Updating / Saving data from {self.source_func}") if self.logger else None
            elif self.logger: self.logger.debug(f"Using cached values. Lag = {self.age}.")
        elif fresh or not self.path.exists() or self.age > str2timedelta(self.expire):  # disk fridge
            if self.logger: self.logger.debug(f"Updating & Saving {self.path} ...")
            self.cache = self.source_func(); self.save(obj=self.cache, path=self.path)  # fresh order, never existed or exists but expired.
        elif self.age < str2timedelta(self.expire) and self.cache is None: self.cache = self.reader(self.path)  # this implementation favours reading over pulling fresh at instantiation.  # exists and not expired. else # use the one in memory self.cache
        return self.cache


if __name__ == '__main__':
    P(r"C:\Users\Alex\Downloads\encrypted_data").move(folder=r'C:/Users/Alex/data/crypto', content=True)
