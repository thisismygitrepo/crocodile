import time

from crocodile.core import Struct, np, os, sys, List, datetime, timestamp, randstr, validate_name, str2timedelta,\
    Save, Path, install_n_import, dill


# =============================== Security ================================================

def obscure(msg: bytes) -> bytes:
    # if type(msg) is str: msg = msg.encode()
    import zlib
    from base64 import urlsafe_b64encode as b64e
    return b64e(zlib.compress(msg, 9))


def unobscure(obscured: bytes) -> bytes:
    import zlib
    from base64 import urlsafe_b64decode as b64d
    return zlib.decompress(b64d(obscured))


def pwd2key(password: str, salt=None, iterations=None) -> bytes:
    """Derive a secret key from a given password and salt"""
    import base64
    if salt is None:
        import hashlib
        m = hashlib.sha256()  # converts anything to fixed length 32 bytes
        m.update(password.encode("utf-8"))
        return base64.urlsafe_b64encode(m.digest())  # make url-safe bytes required by Ferent.
    """Adding salt and iterations to the password. The salt and iteration numbers will be stored explicitly
    with the final encrypted message, i.e. known publicly. The benefit is that they makes trying very hard.
    The machinery below that produces the final key is very expensive (as large as iteration)"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=None)
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt(msg: bytes, key=None, pwd: str = None, salted=True, iteration: int = None) -> bytes:
    """Encryption Tips:
    * Be careful of key being stored unintendedly in console or terminal history, e.g. don't use IPython.
    * It behoves you to try decrypting it to err on the side of safety.
    * Don't forget to delete OR store key file safely.
    """
    # if type(msg) is str: msg = msg.encode()
    from cryptography.fernet import Fernet
    salt = None  # silence the linter.
    # step 1: get the key:
    if pwd is not None:  # generate it from password
        assert key is None, f"You can either pass key or pwd, or none of them, but not both."
        assert type(pwd) is str
        if salted:   # strengthen the password by adding random characters to it. The characters will be
            # explicitly mentioned in final encryption and used at decryption. They help to make problem more expensive
            # to brute force, hence reducing hances of finding a password even though it has few characters.
            import secrets
            if iteration is None: iteration = secrets.randbelow(1_000_000)
            salt = secrets.token_bytes(16)
        else: salt = iteration = None
        key = pwd2key(pwd, salt, iteration)
    elif key is None:  # generate a new key: discouraged, always make your keys/pwd before invoking the func.
        key = Fernet.generate_key()  # uses random bytes, more secure but no string representation
        print(f"KEY SAVED @ {repr(P.tmpdir().joinpath('key.bytes').write_bytes(key))}")  # without verbosity check:
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # a path to a key file was passed, read it:
    elif type(key) is bytes: pass  # key passed explicitly
    else: raise TypeError(f"Key must be either a path, bytes object or None.")
    # if type(msg) is str: msg = msg.encode("utf-8")
    code = Fernet(key).encrypt(msg)
    if pwd is not None and salted is True:
        from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d
        return b64e(b'%b%b%b' % (salt, iteration.to_bytes(4, 'big'), b64d(code)))
    else: return code


def decrypt(token: bytes, key=None, pwd: str = None, salted=True) -> bytes:
    from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d
    from cryptography.fernet import Fernet
    if pwd is not None:
        assert key is None, f"You can either pass key or pwd, or none of them, but not both."
        assert type(pwd) is str
        if salted:
            decoded = b64d(token)
            salt, iterations, token = decoded[:16], decoded[16:20], b64e(decoded[20:])
            key = pwd2key(pwd, salt, int.from_bytes(iterations, 'big'))
        else: key = pwd2key(pwd)
    if type(key) is bytes: pass   # passsed explicitly
    elif type(key) in {str, P, Path}: key = P(key).read_bytes()  # passed a path to a file containing kwy
    else: raise TypeError(f"Key must be either str, P, Path or bytes.")
    return Fernet(key).decrypt(token)


# =================================== File ============================================

class Read(object):
    @staticmethod
    def read(path, **kwargs):
        suffix = Path(path).suffix[1:]
        # if suffix in ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']:
        #     # plt.gcf().canvas.get_supported_filetypes().keys():
        #     return plt.imread(path, **kwargs)
        try: reader = getattr(Read, suffix)
        except AttributeError: raise AttributeError(f"Unknown file type. failed to recognize the suffix {suffix}")
        return reader(str(path), **kwargs)

    @staticmethod
    def npy(path, **kwargs):
        data = np.load(str(path), allow_pickle=True, **kwargs)
        if data.dtype == np.object: data = data.item()
        return Struct(data) if type(data) is dict else data

    @staticmethod
    def mat(path, remove_meta=False, **kwargs):
        from scipy.io import loadmat
        res = Struct(loadmat(path, **kwargs))
        if remove_meta: List(res.keys()).filter("x.startswith('__')").apply(lambda x: res.__delattr__(x))
        return res

    @staticmethod
    def json(path, r=False, **kwargs):
        """Returns a Structure"""
        import json
        try:
            with open(str(path), "r") as file:
                mydict = json.load(file, **kwargs)
        except Exception:  # file has C-style comments.
            with open(str(path), "r") as file:
                mydict = install_n_import("pyjson5").load(file, **kwargs)
        return Struct.recursive_struct(mydict) if r else Struct(mydict)

    @staticmethod
    def yaml(path, r=False):
        import yaml
        with open(str(path), "r") as file: mydict = yaml.load(file, Loader=yaml.FullLoader)
        return Struct(mydict) if not r else Struct.recursive_struct(mydict)

    @staticmethod
    def py(path): import runpy; return Struct(runpy.run_path(path))
    @staticmethod
    def csv(path, **kwargs): import pandas as pd; return pd.read_csv(path, **kwargs)
    @staticmethod
    def pkl(*args, **kwargs): return Read.pickle(*args, **kwargs)
    @staticmethod
    def pickles(bytes_obj): return dill.loads(bytes_obj)

    @staticmethod
    def pickle(path, **kwargs):
        with open(path, 'rb') as file: obj = dill.load(file, **kwargs)
        return Struct(obj) if type(obj) is dict else obj


class P(type(Path()), Path):
    """Path Class: Designed with one goal in mind: any operation on paths MUST NOT take more than one line of code.
    It offers:
    * methods act on the underlying object in the disk drive: move, move_up, copy, encrypt, zip and delete.
    * methods act on the path object: parent, joinpath, switch, prepend, append
    * attributes of path: stem, trunk, size, date etc.
    """
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
            else:
                import shutil
                shutil.rmtree(slf, ignore_errors=True)
            if verbose: print(f"DELETED {repr(slf)}.")
        elif verbose: print(f"Did NOT DELETE because user is not sure. file: {repr(slf)}.")
        return self

    def send2trash(self, verbose=True):
        if self.exists():
            install_n_import("send2trash").send2trash(self.resolve().str)  # do not expand user symlinks.
            if verbose: print(f"TRASHED {repr(self)}")
        elif verbose: print(f"Could NOT trash {self}")
        return self

    def move(self, folder=None, name=None, path=None, rel2it=False, overwrite=False, verbose=True, parents=True,
             content=False):
        """
        :param folder: directory
        :param name: fname of the file
        :param path: full path, that includes directory and file fname.
        :param rel2it:
        :param overwrite:
        :param parents:
        :param content:
        :param verbose:
        :return:
        """
        path = self._resolve_path(folder=folder, name=name, path=path, default_name=self.absolute().name, rel2it=rel2it)
        name, folder = path.name, path.parent
        if parents: folder.create(parents=True, exist_ok=True)
        slf = self.expanduser().resolve()
        if content:
            assert self.is_dir(), NotADirectoryError(f"When `content` flag is set to True, path must be a directory. "
                                                     f"It is not: `{repr(self)}`")
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
        if verbose: print(f"RENAMED {repr(self)} ==> {repr(new_path)}")
        return new_path if not orig else self

    def copy(self, folder=None, name=None, path=None, content=False, verbose=True, append=f"_copy_{randstr()}",
             overwrite=False, orig=False):
        """
        :param orig:
        :param overwrite:
        :param name:
        :param append:
        :param folder: copy the file to this directory (filename remains the same).
        :param path: full path of destination (including -potentially different file name).
        :param content: copy the parent directory or its content (relevant only if copying a directory)
        :param verbose:
        :return: path to copied file or directory.

        .. wanring:: Do not confuse this with ``copy`` module that creates clones of Python objects.
        """ # tested %100
        if folder is not None and path is None:
            if name is None: dest = P(folder).expanduser().resolve().create()
            else:
                dest = P(folder).expanduser().resolve() / name
                content = True
        elif path is not None and folder is None:
            dest = P(path)
            content = True  # this way, the destination will be filled with contents of `self`
        elif path is None and folder is None: dest = self.with_name(str(name)) if name is not None else self.append(append)
        else: raise NotImplementedError
        dest = dest.expanduser().resolve()
        slf = self.expanduser().resolve()
        dest.parent.create()
        if overwrite and dest.exists(): dest.delete(sure=True)
        if slf.is_file():
            import shutil
            shutil.copy(str(slf), str(dest))  # str() only there for Python < (3.6)
            if verbose: print(f"COPIED {repr(slf)} ==> {repr(dest)}")
        elif slf.is_dir():
            from distutils.dir_util import copy_tree
            if content: copy_tree(str(slf), str(dest))
            else: copy_tree(str(slf), str(P(dest).joinpath(slf.name).create()))
            if verbose:
                preface = "Content of " if content else ""
                print(f"COPIED {preface} {repr(slf)} ==> {repr(dest)}")
        else: print(f"Could NOT COPY. Not a file nor a path: {repr(slf)}.")
        return dest / slf.name if not orig else self

    # ======================================= File Editing / Reading ===================================
    def readit(self, reader=None, notfound=FileNotFoundError, readerror=IOError, verbose=False, **kwargs):
        """
        :param reader: function that reads this file format, if not passed it will be inferred from extension.
        :param notfound: behaviour when file ``self`` to be read doesn't actually exist. Default: throw an error.
                can be set to return `False` or any other value that will be returned if file not found.
        :param verbose:
        :param readerror:
        :param kwargs:
        :return:
        """
        _ = readerror
        if not self.exists():
            if notfound is FileNotFoundError: raise FileNotFoundError(f"`{self}` is no where to be found!")
            else: return notfound
        filename = self
        if '.zip' in str(self): filename = self.unzip(folder=self.tmp(folder="unzipped"), verbose=verbose)
        try: return Read.read(filename, **kwargs) if reader is None else reader(str(filename), **kwargs)
        except IOError: raise IOError

    def start(self, opener=None):  # explore folders.
        import subprocess
        if str(self).startswith("http") or str(self).startswith("www"): import webbrowser; webbrowser.open(str(self)); return self
        filename = self.expanduser().resolve().str
        if sys.platform == "win32":
            if opener is None: tmp = f"powershell start '{filename}'"  # double quotes fail with cmd.
            else: tmp = rf'powershell {opener} \'{self}\''
            # os.startfile(filename)  # works for files and folders alike, but if opener is given, e.g. opener="start"
            subprocess.Popen(tmp)  # fails for folders. Start must be passed, but is not defined.
        elif sys.platform == 'linux':
            opener = "xdg-open"
            subprocess.call([opener, filename])  # works for files and folders alike
        else:  subprocess.call(["open", filename])  # works for files and folders alike  # mac
        return self

    def __call__(self, *args, **kwargs): self.start(*args, **kwargs); return self
    def append_text(self, appendix): self.write_text(self.read_text() + appendix); return self

    def modify_text(self, txt, alt, newline=False, notfound_append=False, encoding="utf-8"):
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
        self.parent.create()
        if not self.exists(): self.write_text(txt)
        lines = self.read_text(encoding=encoding).split("\n")
        bingo = False
        for idx, line in enumerate(lines):
            if txt in line:
                bingo = True
                if newline is True: lines[idx] = alt if type(alt) is str else alt(line)
                else: lines[idx] = line.replace(txt, alt if type(alt) is str else alt(line))
        if bingo is False and notfound_append is True: lines.append(alt)  # txt not found, add it anyway.
        self.write_text("\n".join(lines), encoding=encoding)
        return self

    def download(self, directory=None, name=None, memory=False, allow_redirects=True, params=None):
        """Assuming URL points to anything but html page."""
        import requests
        response = requests.get(self.as_url_str(), allow_redirects=allow_redirects, params=params)
        if memory is False:
            directory = P.home().joinpath("Downloads") if directory is None else P(directory)
            directory = directory.joinpath(name or self.name)
            directory.write_bytes(response.content)  # r.contents is bytes encoded as per docs of requests.
            return directory
        else: return response.content
        # Alternative: from urllib import request; request.urlopen(url).read().decode('utf-8')

    def read_fresh_from(self, source_func, expire="1w", save=Save.pickle, read=Read.read):
        return Fridge(source_func=source_func, path=self, expire=expire, save=save, read=read)

    # ================================ Path Object management ===========================================
    """ The default behaviour of methods that mutate the path object:
        Those methods do not perform an action on objects in disk. Instead, only manipulate strings in memory.
        The new object returned by these methods can be a new one (Default) or mutation of `self`, which is
        achieved by using `inliue` flag. A different name for this flag is chosen to distinguish it from `inpalce`
        used by the aforementioned category of methods. Furthermore, this gives extra controllability for user
        to dictate the exact behaviour wanted.
        `Inplace` flag might still be offered in some of those methods and is relevant only if path exists
        The default for this flag is False in those methods.
    """
    def prepend(self, prefix, suffix=None, inlieu=False, inplace=False):
        """Add extra text before file path
        e.g: blah\blah.extenion ==> becomes ==> blah/name_blah.extension.
        notice that `__add__` method removes the extension, while this one preserves it.
        """
        if suffix is None: suffix = ''.join(self.suffixes)
        result = self._return(self.parent.joinpath(prefix + self.trunk + suffix), inlieu)
        if inplace:
            assert self.exists(), f"`inplace` flag is only relevant if the path exists. It doesn't {self}"
            self.retitle(result.name)
        return result

    def append(self, name='', suffix=None, inplace=False, inlieu=False):
        """Add extra text after file path, and optionally add extra suffix. e.g: blah\blah.extenion ==> becomes ==> blah/blah_name.extension"""
        result = self._return(self.parent.joinpath(self.trunk + name + suffix or ''.join(self.suffixes)), inlieu)
        if inplace and self.exists(): self.retitle(result.name)
        return result

    def with_trunk(self, name, inlieu=False, inplace=False):
        """Complementary to `with_stem` and `with_suffic`"""
        res = self.parent.joinpath(name + "".join(self.suffixes))
        if inplace and self.exists(): self.retitle(name=res.name)
        return self._return(res, inlieu)

    def append_time_stamp(self, fmt=None, inlieu=False, inplace=False):
        result = self._return(self.append(name="_" + timestamp(fmt=fmt)), inlieu)
        if inplace and self.exists(): self.retitle(result.name)
        return result

    def switch(self, key: str, val: str, inlieu=False, inplace=False):
        """Changes a given part of the path to another given one. `replace` is an already defined method."""
        result = self._return(P(str(self).replace(key, val)), inlieu)
        if inplace and self.exists(): self.retitle(result)
        return result

    def switch_by_index(self, key: int, val: str, inplace=False, inlieu=False):
        """Changes a given index of the path to another given one"""
        fullparts = list(self.parts)
        fullparts[key] = val
        result = self._return(P(*fullparts), inlieu)
        if inplace and self.exists(): self.retitle(result)
        return result

    def _return(self, res, inlieu: bool):
        """
        :param res: result path, could exists or not.
        :params inlieu: decides on whether the current object `self` is mutated to be the result `res`
        or the result is returned as a separate object.
        """
        if not inlieu: return res
        else:
            if self.exists(): self.rename(str(res))
            self._str = str(res)
            return self

    # ============================= attributes of object ======================================
    @property
    def trunk(self):
        """ useful if you have multiple dots in file path where `.stem` fails."""
        return self.name.split('.')[0]

    @property
    def len(self): return self.__len__()
    @property
    def str(self): return str(self)  # or self._str
    @property
    def items(self): return List(self.parts)
    def __len__(self): return len(self.parts)
    def __contains__(self, item): return item in self.parts
    def __iter__(self): return self.parts.__iter__()
    def __deepcopy__(self): return P(str(self))
    def __getstate__(self): return str(self)
    def __setstate__(self, state): self._str = str(state)
    def __add__(self, other): return self.parent.joinpath(self.stem + str(other))
    def __radd__(self, other): return self.parent.joinpath(str(other) + self.stem)  # other + P and `other` doesn't know how to make this addition.

    def __sub__(self, other):
        """removes all similar characters from the string form of the path"""
        res = P(str(self).replace(str(other), ""))
        if str(res[0]) in {"\\", "/"}: res = res[1:]  # paths starting with "/" are problematic. e.g ~ / "/path" doesn't work.
        return res

    # def __rtruediv__(self, other):
    #     tmp = str(self)
    #     if tmp[0] == "/":  # if dir starts with this, all Path methods fail.
    #         tmp = tmp[1:]
    #     return P(other) / tmp

    def rel2cwd(self, inlieu=False): return self._return(P(self.relative_to(Path.cwd())), inlieu)
    def rel2home(self, inlieu=False): return self._return(P(self.relative_to(Path.home())), inlieu)  # opposite of `expanduser`

    def collapseuser(self, strict=True, inlieu=False):
        """same as rel2home except that it adds the tilde `~` to indicated home at the beginning.
         Thus, it is a self-contained absolute path, bar a `expanduser` method."""
        if "~" in self: return self
        if strict:
            assert str(P.home()) in str(self), ValueError(f"{str(P.home())} is not in the subpath of {str(self)}"
                                                          f" OR one path is relative and the other is absolute.")
        return self._return("~" / (self - P.home()), inlieu)

    def split(self, at: str = None, index: int = None, sep: int = 1, mode=["strict", "lenient"][0]):
        """Splits a path at a given string or index
        :param at: string telling where to split.
        :param index: integer telling at which index to split.
        :param sep: can be either [-1, 0, 1]. Determines where the separator is going to live with:
               left portion, none or right portion.
        :param mode: "lenient" mode makes `split` method behaves like split method of string. This can produce
            unwanted behaviour due to e.g. patial matches. 'strict' mode is the default which only splits
             at exact match.
        :return: two paths
        """
        # ====================================   Splitting
        if index is None and (at is not None):  # at is provided
            if mode == "lenient":
                items = str(self).split(sep=str(at))
                one, two = items[0], items[1]
                one = one[:-1] if one.endswith("/") else one
                two = two[1:] if two.startswith("/") else two
            else:  # "strict"
                index = self.parts.index(str(at))  # raises an error if exact match is not found.
                one, two = self[0:index], self[index + 1:]  # both one and two do not include the split item.
            one, two = P(one), P(two)
        elif index is not None and (at is None):  # index is provided
            one = self[:index]
            two = P(*self.parts[index + 1:])
            at = self[index]  # this is needed below.
        else: raise ValueError("Either `index` or `at` can be provided. Both are not allowed simulatanesouly.")

        # ================================  appending `at` to one of the portions
        if sep == 0: pass  # neither of the portions get the sperator appended to it.
        elif sep == 1: two = at / two   # append it to right portion
        elif sep == -1: one = one / at  # append it to left portion.
        else: raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")
        return one, two

    def __getitem__(self, slici):  # tested.
        if type(slici) is slice: return P(*self.parts[slici])
        elif type(slici) is list or type(slici) is np.ndarray: return P(*[self[item] for item in slici])
        else: return P(self.parts[slici])  # it is an integer

    def __setitem__(self, key, value):
        """
        :param key: typing.Union[str, int, slice]
        :param value: typing.Union[str, Path]
        :return:
        """
        fullparts = list(self.parts)
        new = list(P(value).parts)
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
        sizes = List(['b', 'kb', 'mb', 'gb'])
        factor = dict(zip(sizes + sizes.apply("x.swapcase()"), np.tile(1024 ** np.arange(len(sizes)), 2)))[units]
        if self.is_file(): total_size = self.stat().st_size
        elif self.is_dir(): total_size = sum([item.stat().st_size for item in self.rglob("*") if item.is_file()])
        else: raise FileNotFoundError(self.absolute().as_uri())
        return round(total_size / factor, 1)

    def time(self, which="m", **kwargs):
        """Meaning of ``which values``
            * ``m`` time_produced of modifying file ``content``, i.e. the time_produced it was created.
            * ``c`` time_produced of changing file status (its inode is changed like permissions, path etc, but not content)
            * ``a`` last time_produced the file was accessed.

        :param which: Determines which time_produced to be returned. Three options are availalable:
        :param kwargs:
        """
        timestamp_ = {"m": self.stat().st_mtime, "a": self.stat().st_atime, "c": self.stat().st_ctime}[which]
        return datetime.fromtimestamp(timestamp_, **kwargs)

    def stats(self):
        """A variant of `stat` method that returns a structure with human-readable values."""
        return Struct(size=self.size(), content_mod_time=self.time(which="m"), attr_mod_time=self.time(which="c"),
                      last_access_time=self.time(which="a"), group_id_owner=self.stat().st_gid, user_id_owner=self.stat().st_uid)

    # ================================ String Nature management ====================================
    def clickable(self, inlieu=False): return self._return(self.expanduser().resolve().as_uri(), inlieu)
    def as_url_str(self, inlieu=False): return self._return(self.as_posix().replace("https:/", "https://").replace("http:/", "http://"), inlieu)
    def as_url_obj(self, inlieu=False): return self._return(install_n_import("urllib3").connection_from_url(self), inlieu)
    def as_unix(self, inlieu=False): return self._return(P(str(self).replace('\\', '/').replace('//', '/')), inlieu)
    def get_num(self, astring=None): int("".join(filter(str.isdigit, str(astring or self.stem))))
    def validate_name(self, replace='_'): validate_name(self.trunk, replace=replace)

    # ========================== override =======================================
    def symlink_from(self, folder=None, file=None, verbose=False, overwrite=False):
        assert self.expanduser().exists(), "self must exist if this method is used."
        if file is not None:
            assert folder is None, "You can only pass source or source_dir, not both."
            result = P(file).expanduser().absolute()
        else: result = P(folder or P.cwd()).expanduser().absolute() / self.name
        return result.symlink_to(self, verbose=verbose, overwrite=overwrite)

    def symlink_to(self, target=None, verbose=True, overwrite=False, orig=False):
        """
        Creates a symlink to the target.
        :param target:
        :param verbose:
        :param overwrite: If True, overwrites existing symlink (self). Target path is not changed.
        :param orig:
        """
        target = P(target).expanduser().resolve()
        assert target.exists(), f"Target path `{target}` doesn't exist. This will create a broken link."
        self.parent.create()
        if overwrite:
            if self.is_symlink() or self.exists():
                # self.exists() is False for broken links even though they exist
                self.delete(sure=True, verbose=verbose)
        import platform
        from crocodile.meta import Terminal
        if platform.system() == "Windows" and not Terminal.is_user_admin():  # you cannot create symlink without priviliages.
            Terminal.run_code_as_admin(f" -c \"from pathlib import Path; Path(r'{self.expanduser()}').symlink_to(r'{str(target)}')\"")
            time.sleep(0.5)  # give time_produced for asynch process to conclude before returning response.
        else: super(P, self.expanduser()).symlink_to(str(target))
        if verbose: print(f"LINKED {repr(self)}")
        return P(target) if not orig else self
    
    def resolve(self, strict=False):
        try: return super(P, self).resolve(strict=strict)
        except OSError: return self

    def write_text(self, data: str, **kwargs): super(P, self).write_text(data, **kwargs); return self
    def read_text(self, encoding="utf-8", lines=False): return super(P, self).read_text(encoding=encoding) if not lines else List(super(P, self).read_text(encoding=encoding).splitlines())
    def write_bytes(self, data: bytes): super(P, self).write_bytes(data); return self

    def touch(self, mode: int = 0o666, parents=True, exist_ok: bool = ...):
        if parents: self.parent.create(parents=parents)
        super(P, self).touch(mode=mode, exist_ok=exist_ok)
        return self

    # ======================================== Folder management =======================================
    def create(self, parents=True, exist_ok=True, parent_only=False):
        if parent_only: self.parent.mkdir(parents=parents, exist_ok=exist_ok)
        else: self.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    @property
    def browse(self): return self.search("*").to_struct(key_val=lambda x: ("qq_" + validate_name(x), x)).clean_view

    def search(self, pattern='*', r=False, generator=False, files=True, folders=True, compressed=False,
               dotfiles=False, filters: list = None, not_in: list = None, exts=None, win_order=False):
        """
        :param pattern:  linux search pattern
        :param r: recursive search flag
        :param generator: output format, list or generator.
        :param files: include files in search.
        :param folders: include directories in search.
        :param compressed: search inside compressed files.
        :param dotfiles: flag to indicate whether the search should include those or not.
        :param filters: list of filters
        :param not_in: list of strings that search results should not contain them (short for filter with simple lambda)
        :param exts: list of extensions to search for.
        :param win_order: return search results in the order of files as they appear on a Windows machine.

        :return: search results.
        """
        # ================= Get concrete values for default arguments ========================================
        filters = filters or []
        if not_in is not None: filters += [lambda x: all([str(notin) not in str(x) for notin in not_in])]
        if exts is not None: filters += [lambda x: any([ext in x.name for ext in exts])]
        # ============================ get generator of search results ========================================
        slf = self.expanduser().resolve()
        if compressed and slf.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(str(slf)) as z: content = List(z.namelist())
            from fnmatch import fnmatch
            raw = content.filter(lambda x: fnmatch(x, pattern)).apply(lambda x: slf / x)
        elif dotfiles: raw = slf.glob(pattern) if not r else self.rglob(pattern)
        else:  # glob ignroes dot and hidden files
            from glob import glob
            raw = glob(str(slf / "**" / pattern), recursive=r) if r else glob(str(slf.joinpath(pattern)))

        if compressed:
            comp_files = List(raw).filter(lambda x: '.zip' in str(x))
            for comp_file in comp_files:
                raw += P(comp_file).search(pattern=pattern, r=r, generator=generator, files=files, folders=folders,
                                           compressed=compressed,
                                           dotfiles=dotfiles, filters=filters, not_in=not_in, win_order=win_order)

        def run_filter(item_):
            flags = [item_.is_dir() if not files else True, item_.is_file() if not folders else True]
            return all(flags + [afilter(item_) for afilter in filters])

        if generator:
            def gen():
                flag = False
                while not flag:
                    item_ = next(raw)
                    flag = P(item_) if run_filter(P(item_)) else None
                    if flag: yield item_
            return gen
        else:  # unpack the generator and vet the items (the function also returns P objects)
            processed = [P(item) for item in raw if run_filter(P(item))]
            if not processed: return List(processed)  # if empty, don't proceeed
            if win_order:  # this option only supported in non-generator mode.
                import re
                processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', x.stem)])
            return List(processed)

    def tree(self, level: int = -1, limit_to_directories: bool = False,
             length_limit: int = 1000, stats=False, desc=None):
        """Given a directory Path object print a visual tree structure
        Based on: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python"""
        from itertools import islice
        space = '    '
        branch = '│   '
        tee = '├── '
        last = '└── '
        dir_path = self
        files = 0
        directories = 0

        def get_stats(apath):
            if stats or desc:
                sts = apath.stats(printit=False)
                result = f" {sts.size} MB. {sts.content_mod_time}. "
                if desc is not None: result += desc(apath)
                return result
            else: return ""

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
        for line in islice(iterator, length_limit): print(line)
        if next(iterator, None): print(f'... length_limit, {length_limit}, reached, counted:')
        print(f'\n{directories} directories' + (f', {files} files' if files else ''))

    def find(self, *args, r=True, compressed=True, **kwargs):
        """short for the method ``search`` then pick first item from results.
        useful for superflous directories or zip archives containing a single file.

        Behaviour:
        * if path (self) is a file, it returns itself.
        * unless this path is an archive file, in which case it is parsed as a directory.
        * Return None in case the directory is empty.
        """
        if compressed is False and self.is_file(): return self
        results = self.search(*args, r=r, compressed=compressed, **kwargs)
        if len(results) > 0:
            result = results[0]
            if ".zip" in str(result): return result.unzip()
            return result
        else: return None

    @staticmethod
    def pwd(): return P.cwd()
    @staticmethod
    def tempdir(): import tempfile; return P(tempfile.mktemp())
    @staticmethod
    def temp(): import tempfile; return P(tempfile.gettempdir())
    @staticmethod
    def tmpdir(prefix=""): return P.tmp(folder=rf"tmp_dirs/{prefix + ('_' if prefix != '' else '') + randstr()}")
    def chdir(self): os.chdir(str(self.expanduser())); return self
    def listdir(self): return List(os.listdir(self.expanduser().resolve())).apply(P)
    @staticmethod
    def tmpfile(name=None, suffix="", folder=None, tstamp=False): return P.tmp(file=(name or randstr()) + "_" + randstr() + (("_" + timestamp()) if tstamp else "") + suffix, folder=folder or "tmp_files")

    @staticmethod
    def tmp(folder=None, file=None, path="home", verbose=False):
        """
        :param folder: this param is created automatically.
        :param file: this param is appended to path, but not created.
        :param path:
        :param verbose:
        :return:
        """
        if str(path) == "home":
            path = P.home() / f"tmp_results"
            path.mkdir(exist_ok=True, parents=True)
        if folder is not None:
            path = path / folder
            path.mkdir(exist_ok=True, parents=True)
        if file is not None:  path = path / file
        if verbose: print(f"TMPDIR {repr(path)}. Parent: {repr(path.parent)}")
        return path

    # ====================================== Compression ===========================================
    def zip(self, path=None, folder=None, name=None, arcname=None, inplace=False, verbose=True, content=True,
            orig=False, **kwargs):
        """
        """
        path = self._resolve_path(folder, name, path, self.name).expanduser().resolve()
        slf = self.expanduser().resolve()
        arcname = P(arcname or slf.name)
        if arcname.name != slf.name: arcname /= slf.name  # arcname has to start from somewhere and end with filename
        if slf.is_file():
            if path.suffix != ".zip": path = path + f".zip"
            path = Compression.zip_file(ip_path=slf, op_path=path, arcname=arcname, **kwargs)
        else:
            root_dir, base_dir = (slf, ".") if content else slf.split(at=str(arcname[0]))[0], arcname
            path = Compression.compress_folder(root_dir=root_dir, op_path=path, base_dir=base_dir, fmt='zip', **kwargs)
        if verbose: print(f"ZIPPED {repr(slf)} ==>  {repr(path)}")
        if inplace: slf.delete(sure=True, verbose=verbose)
        return path if not orig else self

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
        slf = self.expanduser().resolve()
        zipfile = slf
        if slf.suffix != ".zip":  # may be there is .zip somewhere in the path.
            if ".zip" not in str(slf): return slf
            zipfile, fname = slf.split(at=List(slf.parts).filter(lambda x: ".zip" in x)[0], sep=-1)
        folder = (zipfile.parent / zipfile.stem) if folder is None else P(folder).joinpath(zipfile.stem).expanduser().resolve()
        if content: folder = folder.parent
        result = Compression.unzip(zipfile, folder, fname, **kwargs)
        if verbose: print(f"UNZIPPED {repr(zipfile)} ==> {repr(result)}")
        if inplace: slf.delete(sure=True, verbose=verbose)
        return result if not orig else self

    def tar(self, path=None):
        if path is None: path = self + '.gz'
        result = Compression.untar(self, op_path=path)
        return result

    def untar(self, path, verbose=True):
        _ = self, path, verbose
        return P()

    def gz(self, path, verbose=True):
        _ = self, path, verbose
        return P()

    def ungz(self, path, verbose=True):
        _ = self, path, verbose
        return P()

    def tar_gz(self): pass

    def untar_ungz(self, folder=None, inplace=False, verbose=True):
        folder = folder or P(self.parent) / P(self.stem)
        intrem = self.ungz(path=folder, verbose=verbose)
        result = intrem.untar(path=folder, verbose=verbose)
        intrem.delete(sure=True, verbose=verbose)
        if inplace: self.delete(sure=True, verbose=verbose)
        return result

    def compress(self, path=None, base_dir=None, fmt="zip", inplace=False, **kwargs):
        fmts = ["zip", "tar", "gzip"]
        assert fmt in fmts, f"Unsupported format {fmt}. The supported formats are {fmts}"
        _ = self, path, base_dir, kwargs, inplace
        pass

    def decompress(self):
        pass

    def encrypt(self, key=None, pwd=None, folder=None, name=None, path=None, verbose=True,
                append="_encrypted", inplace=False, orig=False, use_7z=False):
        """
        see: https://stackoverflow.com/questions/42568262/how-to-encrypt-text-with-a-password-in-python
        https://stackoverflow.com/questions/2490334/simple-way-to-encode-a-string-according-to-a-password
        """
        slf = self.expanduser().resolve()
        assert slf.is_file(), f"Cannot encrypt a directory. You might want to try `zip_n_encrypt`. {self}"
        path = self._resolve_path(folder, name, path, slf.append(name=append).name)
        if use_7z:
            import crocodile.environment as env
            path = path + '.7z'
            if env.system == "Windows":
                program = env.ProgramFiles.joinpath("7-Zip/7z.exe")
                if not program.exists(): env.tm.run('winget install --name "7-zip" --Id "7zip.7zip" --source winget', shell="powershell")
                env.tm.run(f"&'{program}' a '{path}' '{self}' -p{pwd}", shell="powershell")
            else: raise NotImplementedError("7z not implemented for Linux")
            return path
        code = encrypt(msg=slf.read_bytes(), key=key, pwd=pwd)
        path.write_bytes(code)  # Fernet(key).encrypt(self.read_bytes()))
        if verbose: print(f"ENCRYPTED: {repr(slf)} ==> {repr(path)}.")
        if inplace: slf.delete(sure=True, verbose=verbose)
        return path if not orig else self

    def decrypt(self, key=None, pwd=None, path=None, folder=None, name=None, verbose=True, append="_encrypted",
                inplace=False, orig=False):
        slf = self.expanduser().resolve()
        path = self._resolve_path(folder, name, path, slf.switch(append, "").name)
        path.write_bytes(decrypt(slf.read_bytes(), key=key, pwd=pwd))  # Fernet(key).decrypt(self.read_bytes()))
        if verbose: print(f"DECRYPTED: {repr(slf)} ==> {repr(path)}.")
        if inplace: slf.delete(sure=True, verbose=verbose)
        return path if not orig else self

    def zip_n_encrypt(self, key=None, pwd=None, inplace=False, verbose=True, orig=False):
        zipped = self.zip(inplace=inplace, verbose=verbose)
        zipped_secured = zipped.encrypt(key=key, pwd=pwd, verbose=verbose, inplace=True)
        return zipped_secured if not orig else self

    def decrypt_n_unzip(self, key=None, pwd=None, inplace=False, verbose=True, orig=False):
        deciphered = self.decrypt(key=key, pwd=pwd, verbose=verbose, inplace=inplace)
        unzipped = deciphered.unzip(folder=None, inplace=True, content=False)
        return unzipped if not orig else self

    # ========================== Helpers =========================================
    def _resolve_path(self, folder, name, path, default_name, rel2it=False):
        """From all arguments, figure out what is the final path.
        :param rel2it: `folder` or `path` are relative to `self` as opposed to cwd.
        """
        if path is not None:
            assert folder is None and name is None, f"If `path` is passed, `folder` and `name` cannot be passed."
            if rel2it: path = self.joinpath(path).resolve()
            path = P(path).expanduser().resolve()
            assert not path.is_dir(), f"`path` passed is a directory! it must not be that. If this is meant, pass it with `path` kwarg. {path}"
            folder = path.parent
            name = path.name
            _ = name, folder
        else:
            if name is None: name = default_name
            else: name = str(name)  # good for edge cases of path with single part.
            if folder is None: folder = self.parent  # means same directory, just different name
            if rel2it: folder = self.joinpath(folder).resolve()
            path = P(folder).expanduser().resolve() / name
        return path

    def _type(self):
        if self.absolute():
            if self.is_file(): return "File"
            elif self.is_dir(): return "Dir"
            else: return "NotExist"  # there is no tell if it is a file or directory.
        else: return "Relative"      # there is no tell if it is a file or directory.


class Compression(object):
    """Provides consistent behaviour across all methods ...
    Both files and folders when compressed, default is being under the root of archive."""
    @staticmethod
    def compress_folder(root_dir, op_path, base_dir, fmt='zip', **kwargs):
        """
        shutil works with folders nicely (recursion is done interally)
        # directory to be archived: root_dir\base_dir, unless base_dir is passed as absolute path.
        # when archive opened; base_dir will be found.
        """
        assert fmt in {"zip", "tar", "gztar", "bztar", "xztar"}
        assert P(op_path).suffix != ".zip", f"Don't add zip extention to this method, it is added automatically."
        import shutil
        result_path = shutil.make_archive(base_name=op_path, format=fmt, root_dir=str(root_dir), base_dir=str(base_dir), **kwargs)
        return P(result_path)  # same as path but (possibly) with format extension

    @staticmethod
    def zip_file(ip_path, op_path, arcname=None, password=None, **kwargs):
        """
        arcname determines the directory of the file being archived inside the archive. Defaults to same
        as original directory except for drive. When changed, it should still include the file path in its end.
        If arcname = filename without any path, then, it will be in the root of the archive.
        """
        import zipfile
        jungle_zip = zipfile.ZipFile(str(op_path), 'w')
        if password is not None: jungle_zip.setpassword(pwd=password)
        jungle_zip.write(filename=str(ip_path), arcname=str(arcname) if arcname is not None else None, compress_type=zipfile.ZIP_DEFLATED, **kwargs)
        jungle_zip.close()
        return op_path

    @staticmethod
    def unzip(ip_path, op_path, fname=None, password=None, **kwargs):
        from zipfile import ZipFile
        with ZipFile(str(ip_path), 'r') as zipObj:
            if fname is None: zipObj.extractall(op_path, pwd=password, **kwargs)
            else:
                zipObj.extract(member=str(fname), path=str(op_path), pwd=password)
                op_path = P(op_path) / fname
        return P(op_path)

    @staticmethod
    def gz(file, op_file):
        import gzip
        import shutil
        with open(file, 'rb') as f_in:
            with gzip.open(op_file, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        return op_file

    @staticmethod
    def ungz(self, op_path=None):
        import gzip
        import shutil
        with gzip.open(str(self), 'r') as f_in, open(op_path, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        return P(op_path)

    @staticmethod
    def tar(self, op_path):
        import tarfile
        with tarfile.open(op_path, "w:gz") as tar: tar.add(str(self), arcname=os.path.basename(str(self)))
        return op_path

    @staticmethod
    def untar(self, op_path, fname=None, mode='r', **kwargs):
        import tarfile
        with tarfile.open(str(self), mode) as file:
            if fname is None: file.extractall(path=op_path, **kwargs)  # extract all files in the archive
            else: file.extract(fname, **kwargs)
        return P(op_path)


class MemoryDB:
    """This class holds the historical data. It acts like a database, except that is memory based."""

    def __init__(self, size=5, ):
        self.size = size
        self.list = List()

    def __repr__(self): return f"MemoryDB. Size={self.size}. Current length = {self.len}"
    def __getitem__(self, item): return self.list[item]
    @property
    def len(self): return len(self.list)

    def append(self, item):
        self.list.append(item)
        if self.len > self.size: self.list = self.list[-self.size:]  # take latest frames and drop the older ones.


class Fridge:
    """This class helps to accelrate access to latest data coming from a rather expensive function,
    Thus, if multiple methods from superior classses requested this within 0.1 seconds,
    there will be no problem of API being a bottleneck reducing running time_produced to few seconds
    The class has two flavours, memory-based and disk-based variants."""
    def __init__(self, source_func, expire="1m", time_produced=None, logger=None, path=None, save=Save.pickle, read=Read.read):
        """
        :param source_func: function that returns data
        :param expire: time_produced after which the data is considered expired.
        :param time_produced: creation time. If not provided, it will be taken from the source_func.
        :param logger: logger to use.
        :param path: path to save the data.
        :param save: save method.
        :param read: read method.
        """
        self.cache = None  # fridge content
        self.time_produced = time_produced or datetime.now()  # init time_produced
        self.expire = expire  # how much time_produced elapsed before
        self.source_func = source_func  # function which when called returns a fresh object to be frozen.
        self.logger = logger
        self.path = P(path) if path else None  # if path is passed, it will function as disk-based flavour.
        self.save = save
        self.read = read

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.path is not None: state["path"] = self.path.rel2home()  # With this implementation, instances can be pickled and loaded up in different machine and still works.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.path is not None: self.path = P.home() / self.path

    @property
    def age(self): return datetime.now() - self.time_produced if self.path is None else datetime.now() - self.path.stats().content_mod_time
    def reset(self): self.time_produced = datetime.now()

    def __call__(self, fresh=False):
        """"""
        if self.path is None:  # Memory Fridge
            if self.cache is None or fresh is True or self.age > str2timedelta(self.expire):
                self.cache = self.source_func()
                self.time_produced = datetime.now()
                if self.logger: self.logger.debug(f"Updating / Saving data from {self.source_func}")
            else:
                if self.logger: self.logger.debug(f"Using cached values. Lag = {self.age}.")
            return self.cache
        else:  # disk fridge
            if fresh or not self.path.exists() or self.age > str2timedelta(self.expire):
                if self.logger: self.logger.debug(f"Updating & Saving {self.path} ...")
                # print(datetime.now() - self.path.stats().content_mod_time, str2timedelta(self.expire))
                self.cache = self.source_func()  # fresh order, never existed or exists but expired.
                self.save(obj=self.cache, path=self.path)
            elif self.age < str2timedelta(self.expire):
                if self.cache is None:  # this implementation favours reading over pulling fresh at instantiation.
                    self.cache = self.read(self.path)  # exists and not expired.
                else:  # use the one in memory self.cache
                    pass
            return self.cache


if __name__ == '__main__':
    pass
