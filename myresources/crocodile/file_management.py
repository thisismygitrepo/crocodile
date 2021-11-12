
from crocodile.core import Struct, pd, np, os, List, datetime, get_time_stamp, Base
# Typing
import re
import typing
import string
# Path
import sys
import shutil
from glob import glob
from pathlib import Path


class Read(object):
    @staticmethod
    def read(path, **kwargs):
        suffix = Path(path).suffix[1:]
        # if suffix in ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']:
        #     # plt.gcf().canvas.get_supported_filetypes().keys():
        #     return plt.imread(path, **kwargs)
        # else:
        try:
            reader = getattr(Read, suffix)
        except AttributeError:
            raise AttributeError(f"Unknown file type. failed to recognize the suffix {suffix}")
        return reader(str(path), **kwargs)

    @staticmethod
    def npy(path, **kwargs):
        """returns Structure if the object loaded is a dictionary"""
        data = np.load(str(path), allow_pickle=True, **kwargs)
        if data.dtype == np.object:
            data = data.item()
        if type(data) is dict:
            data = Struct(data)
        return data

    @staticmethod
    def mat(path, remove_meta=False, **kwargs):
        """
        :param remove_meta:
        :param path:
        :return: Structure object
        """
        from scipy.io import loadmat
        res = Struct(loadmat(path, **kwargs))
        if remove_meta:
            res.keys().filter("x.startswith('__')").apply(lambda x: res.__delattr__(x))
        return res

    @staticmethod
    def json(path, r=False, **kwargs):
        """Returns a Structure"""
        import json
        with open(str(path), "r") as file:
            mydict = json.load(file, **kwargs)
        if r:
            return Struct.recursive_struct(mydict)
        else:
            return Struct(mydict)

    @staticmethod
    def yaml(path, r=False):
        import yaml
        with open(str(path), "r") as file:
            mydict = yaml.load(file, Loader=yaml.FullLoader)
        if r:
            return Struct.recursive_struct(mydict)
        else:
            return Struct(mydict)

    @staticmethod
    def csv(path, **kwargs):
        # w = P(path).append(".dtypes").readit(reader=pd.read_csv, notexist=None)
        # w = dict(zip(w['index'], w['dtypes'])) if w else w
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def pickle(path, **kwargs):
        # import pickle
        # dill = Experimental.assert_package_installed("dill")
        # removed for disentanglement
        import dill
        with open(path, 'rb') as file:
            obj = dill.load(file, **kwargs)
        if type(obj) is dict:
            obj = Struct(obj)
        return obj

    @staticmethod
    def pkl(*args, **kwargs):
        return Read.pickle(*args, **kwargs)


class P(type(Path()), Path, Base):
    """Path Class: Designed with one goal in mind: any operation on paths MUST NOT take more than one line of code.
    """

    # %% ===================================== File Specs =============================================================
    def size(self, units='mb'):
        sizes = List(['b', 'kb', 'mb', 'gb'])
        factor = dict(zip(sizes + sizes.apply("x.swapcase()"),
                          np.tile(1024 ** np.arange(len(sizes)), 2)))[units]
        if self.is_file():
            total_size = self.stat().st_size
        elif self.is_dir():
            results = self.rglob("*")
            total_size = 0
            for item in results:
                if item.is_file():
                    total_size += item.stat().st_size
        else:
            raise TypeError("This thing is not a file nor a folder.")
        return round(total_size / factor, 1)

    def time(self, which="m", **kwargs):
        """Meaning of ``which values``
            * ``m`` time of modifying file ``content``, i.e. the time it was created.
            * ``c`` time of changing file status (its inode is changed like permissions, name etc, but not contents)
            * ``a`` last time the file was accessed.

        :param which: Determines which time to be returned. Three options are availalable:
        :param kwargs:
        :return:
        """
        time = {"m": self.stat().st_mtime, "a": self.stat().st_atime, "c": self.stat().st_ctime}[which]
        return datetime.fromtimestamp(time, **kwargs)

    def stats(self, printit=True):
        """A variant of `stat` method that returns a structure with human-readable values."""
        res = Struct(size=self.size(),
                     content_mod_time=self.time(which="m"),
                     attr_mod_time=self.time(which="c"),
                     last_access_time=self.time(which="a"),
                     group_id_owner=self.stat().st_gid,
                     user_id_owner=self.stat().st_uid)
        if printit:
            res.print()
        return res

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
                if desc is not None:
                    result += desc(apath)
                return result
            else:
                return ""

        def inner(apath: P, prefix: str = '', level_=-1):
            nonlocal files, directories
            if not level_: return  # 0, stop iterating
            contents = apath.search("*", files=not limit_to_directories)
            pointers = [tee] * (len(contents) - 1) + [last]
            for pointer, path in zip(pointers, contents):
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
        for line in islice(iterator, length_limit):
            print(line)
        if next(iterator, None):
            print(f'... length_limit, {length_limit}, reached, counted:')
        print(f'\n{directories} directories' + (f', {files} files' if files else ''))

    # ================================ Path Object management ===========================================
    @property
    def trunk(self):
        """ useful if you have multiple dots in file name where .stem fails.
        """
        return self.name.split('.')[0]

    def __add__(self, name):
        """Behaves like adding strings"""
        return self.parent.joinpath(self.stem + name)

    def __sub__(self, other):
        """removes all similar characters from the string form of the path"""
        return P(str(self).replace(str(other), ""))

    # def __rtruediv__(self, other):
    #     tmp = str(self)
    #     if tmp[0] == "/":  # if dir starts with this, all Path methods fail.
    #         tmp = tmp[1:]
    #     return P(other) / tmp

    def prepend(self, prefix, stem=False):
        """Add extra text before file name
        e.g: blah\blah.extenion ==> becomes ==> blah/name_blah.extension.
        notice that `__add__` method removes the extension, while this one preserves it.
        """
        if stem:
            return self.parent.joinpath(prefix + self.stem)
        else:
            return self.parent.joinpath(prefix + self.name)

    def append(self, name='', suffix=None):
        """Add extra text after file name, and optionally add extra suffix.
        e.g: blah\blah.extenion ==> becomes ==> blah/blah_name.extension
        """
        if suffix is None:
            suffix = ''.join(self.suffixes)
        return self.parent.joinpath(self.stem + name + suffix)

    def append_time_stamp(self, ft=None):
        return self.append(name="-" + get_time_stamp(ft=ft))

    def absolute_from(self, reference=None):
        """As opposed to ``relative_to`` which takes two abolsute paths and make ``self`` relative to ``reference``,
        this one takes in two relative paths, and return an absolute version of `self` the reference
        for which is ``reference``.

        :param reference: a directory `name` from which the current relative path ``self`` is defined.
            Default value of reference is current directory name, making the method act like ``absolute`` method

        .. warning:: ``reference`` should be within working directory, otherwise it raises an error.

        .. note:: If you have the full path of the reference, then this method would give the same result as
            agoing with `reference / self`
        """
        if reference is None:
            reference = P.cwd()[-1].string
        return P.cwd().split(at=reference)[0] / reference / self

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
                items = str(self).split(sep=at)
                one, two = items[0], items[1]
                one = one[:-1] if one.endswith("/") else one
                two = two[1:] if two.startswith("/") else two

            else:  # "strict"
                index = self.parts.index(at)  # raises an error if exact match is not found.
                one, two = self[0:index], self[index + 1:]  # both one and two do not include the split item.

            one, two = P(one), P(two)

        elif index is not None and (at is None):  # index is provided
            one = self[:index]
            two = P(*self.parts[index + 1:])
            at = self[index]  # this is needed below.

        else:
            raise ValueError("Either `index` or `at` can be provided. Both are not allowed simulatanesouly.")

        # ================================  appending `at` to one of the portions
        if sep == 0:
            pass  # neither of the portions get the sperator appended to it.
        elif sep == 1:  # append it to right portion
            two = at / two
        elif sep == -1:  # append it to left portion.
            one = one / at
        else:
            raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")

        return one, two

    def __getitem__(self, slici):  # tested.
        if type(slici) is slice:
            return P(*self.parts[slici])
        elif type(slici) is list or type(slici) is np.ndarray:
            return P(*[self[item] for item in slici])
        else:
            return P(self.parts[slici])

    def __setitem__(self, key: typing.Union[str, int, slice], value: typing.Union[str, Path]):
        fullparts = list(self.parts)
        new = list(P(value).parts)
        if type(key) is str:
            idx = fullparts.index(key)
            fullparts.remove(key)
            fullparts = fullparts[:idx] + new + fullparts[idx + 1:]
        elif type(key) is int or type(key) is slice:
            if type(key) is int:  # replace this entry
                fullparts = fullparts[:key] + new + fullparts[key + 1:]
            elif type(key) is slice:
                if key.stop is None:
                    key = slice(key.start, len(fullparts), key.step)
                if key.start is None:
                    key = slice(0, key.stop, key.step)
                fullparts = fullparts[:key.start] + new + fullparts[key.stop:]
        obj = P(*fullparts)
        self._str = str(obj)
        # TODO: Do we need to update those as well?
        # self._parts
        # self._pparts
        # self._cparts
        # self._cached_cparts

    def __len__(self):
        return len(self.parts)

    @property
    def len(self):
        return self.__len__()

    def switch(self, key: str, val: str):
        """Changes a given part of the path to another given one. `replace` is an already defined method."""
        return P(str(self).replace(key, val))

    def switch_by_index(self, key: int, val: str):
        """Changes a given index of the path to another given one"""
        fullparts = list(self.parts)
        fullparts[key] = val
        return P(*fullparts)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            _ = {}
        return P(str(self))

    # ================================ String Nature management ====================================
    def __repr__(self):  # this is useful only for the console
        return "P: " + self.__str__()

    @property
    def string(self):  # this method is used by other functions to get string representation of path
        return str(self)

    def get_num(self, astring=None):
        if astring is None:
            astring = self.stem
        return int("".join(filter(str.isdigit, str(astring))))

    def make_valid_filename(self, replace='_'):
        """Converts arbitrary filename into proper variable name as per Python."""
        return self.make_valid_filename_(self.trunk, replace=replace)

    @staticmethod
    def make_valid_filename_(astring, replace='_'):
        return re.sub(r'^(?=\d)|\W', replace, str(astring))

    @staticmethod
    def random(length=10, pool=None):
        if pool is None:
            pool = string.ascii_letters
        import random
        result_str = ''.join(random.choice(pool) for _ in range(length))
        return result_str

    def as_unix(self):
        return P(str(self).replace('\\', '/').replace('//', '/'))

    # ==================================== File management =========================================
    def delete(self, are_you_sure=False):
        if are_you_sure:
            if self.is_file():
                self.unlink()  # missing_ok=True added in 3.8
            else:

                shutil.rmtree(self, ignore_errors=True)
                # self.rmdir()  # dir must be empty
        else:
            print("File not deleted because user is not sure.")

    def send2trash(self):
        # send2trash = Experimental.assert_package_installed("send2trash")
        # removed for disentanglement
        import send2trash
        send2trash.send2trash(self.string)

    def move(self, new_path):
        new_path = P(new_path)
        temp = self.absolute()
        temp.rename(new_path.absolute() / temp.name)
        return new_path

    def renameit(self, new_file_name):
        assert type(new_file_name) is str, "New new should be a string representing file name alone."
        new_path = self.parent / new_file_name
        self.rename(new_path)
        return new_path

    def copy(self, target_dir=None, target_name=None, contents=False, verbose=False):
        """

        :param target_dir: copy the file to this directory (filename remains the same).
        :param target_name: full path of destination (including -potentially different-  file name).
        :param contents: copy the parent directory or its contents (relevant only if copying a directory)
        :param verbose:
        :return: path to copied file or directory.

        .. wanring:: Do not confuse this with ``copy`` module that creates clones of Python objects.

        """
        dest = None  # destination.

        if target_dir is not None:
            assert target_name is None, f"You can either pass target_dir or target_name but not both"
            dest = P(target_dir).create()  # / self.name

        if target_name is not None:
            assert target_dir is None, f"You can either pass target_dir or target_name but not both"
            target_name = P(target_name)
            target_name.parent.create()
            dest = target_name

        if dest is None:
            dest = self.append(f"_copy__{get_time_stamp()}")

        if self.is_file():

            shutil.copy(str(self), str(dest))  # str() only there for Python < (3.6)
            if verbose: print(f"File \n{self}\ncopied successfully to: \n{dest}")
        elif self.is_dir():
            from distutils.dir_util import copy_tree
            if contents:
                copy_tree(str(self), str(dest))
            else:
                copy_tree(str(self), str(P(dest).joinpath(self.name).create()))
            if verbose:
                preface = "Contents of " if contents else ""
                print(f"{preface}\n{self.as_uri()}\ncopied successfully to: \n{dest.as_uri()}")
        else:
            print(f"Could not copy this thing. {self.as_uri()}. Not a file nor a folder.")
        return dest / self.name

    def clean(self):
        """removes contents on a folder, rather than deleting the folder."""
        contents = self.listdir()
        for content in contents:
            self.joinpath(content).send2trash()
        return self

    def readit(self, reader=None, notfound=FileNotFoundError, verbose=False, **kwargs):
        """

        :param reader: function that reads this file format, if not passed it will be inferred from extension.
        :param notfound: behaviour when file ``self`` to be read doesn't actually exist. Default: throw an error.
                can be set to return `False` or any other value that will be returned if file not found.
        :param verbose:
        :param kwargs:
        :return:
        """
        filename = self
        if '.zip' in str(self):
            filename = self.unzip(op_path=self.tmp("unzipped"))
            if verbose:
                print(f"File {self} was uncompressed to {filename}")

        if str(filename).startswith("http") or str(filename).startswith("www"):
            import webbrowser
            webbrowser.open(str(filename))
            return self
        try:
            if reader is None:  # infer the reader
                return Read.read(filename, **kwargs)
            else:
                return reader(str(filename), **kwargs)
        except FileNotFoundError:
            if notfound is FileNotFoundError:
                raise notfound
            else:
                return notfound
        # for other errors, we do not know how to handle them, thus, they will be raised automatically.

    def explore(self):  # explore folders.
        # os.startfile(os.path.realpath(self))
        filename = self.absolute().string
        if sys.platform == "win32":
            os.startfile(filename)  # works for files and folders alike
        elif sys.platform == 'linux':
            import subprocess
            opener = "xdg-open"
            subprocess.call([opener, filename])  # works for files and folders alike
        else:  # mac
            # os.system(f"open {filename}")
            import subprocess
            subprocess.call(["open", filename])  # works for files and folders alike

    # ======================================== Folder management =======================================
    def create(self, parents=True, exist_ok=True, parent_only=False):
        """Creates directory while returning the same object
        """
        if parent_only:
            self.parent.mkdir(parents=parents, exist_ok=exist_ok)
        else:
            self.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    @property
    def browse(self):
        return self.search("*").to_struct(key_val=lambda x: ("qq_" + x.make_valid_filename(), x)).clean_view

    def search(self, pattern='*', r=False, generator=False, files=True, folders=True, compressed=False,
               dotfiles=False,
               absolute=True, filters: list = None, not_in: list = None, win_order=False):
        """
        :param pattern:  linux search pattern
        :param r: recursive search flag
        :param generator: output format, list or generator.
        :param files: include files in search.
        :param folders: include directories in search.
        :param compressed: search inside compressed files.
        :param dotfiles: flag to indicate whether the search should include those or not.
        :param filters: list of filters
        :param absolute: return relative paths or abosolute ones.
        :param not_in: list of strings that search results should not contain them (short for filter with simple lambda)
        :param win_order: return search results in the order of files as they appear on a Windows machine.

        :return: search results.

        # :param visible: exclude hidden files and folders (Windows)
        """
        # ================= Get concrete values for default arguments ========================================
        if filters is None:
            filters = []
        else:
            pass

        if not_in is not None:
            for notin in not_in:
                filters += [lambda x: str(notin) not in str(x)]

        # ============================ get generator of search results ========================================

        if self.suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(str(self)) as z:
                contents = List(z.namelist())
            from fnmatch import fnmatch
            raw = contents.filter(lambda x: fnmatch(x, pattern)).apply(lambda x: self / x)

        elif dotfiles:
            raw = self.glob(pattern) if not r else self.rglob(pattern)
        else:
            if r:
                path = self / "**" / pattern
                raw = glob(str(path), recursive=r)
            else:
                path = self.joinpath(pattern)
                raw = glob(str(path))

        if compressed:
            comp_files = List(raw).filter(lambda x: '.zip' in str(x))
            for comp_file in comp_files:
                raw += P(comp_file).search(pattern=pattern, r=r, generator=generator, files=files, folders=folders,
                                           compressed=compressed,
                                           dotfiles=dotfiles,
                                           absolute=absolute, filters=filters, not_in=not_in, win_order=win_order)

            # if os.name == 'nt':

        #     import win32api, win32con

        # def folder_is_hidden(p):
        #     if os.name == 'nt':
        #         attribute = win32api.GetFileAttributes(p)
        #         return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)

        def run_filter(item_):
            flags = [True]
            if not files:
                flags.append(item_.is_dir())
            if not folders:
                flags.append(item_.is_file())
            for afilter in filters:
                flags.append(afilter(item_))
            return all(flags)

        def do_screening(item_):
            item_ = P(item_)  # because some filters needs advanced functionalities of P objects.
            if absolute:
                item_ = item_.absolute()

            if run_filter(item_):
                return item_
            else:
                return None

        if generator:
            def gen():
                flag = False
                while not flag:
                    item_ = next(raw)
                    flag = do_screening(item_)
                    if flag:
                        yield item_

            return gen
        else:
            # unpack the generator and vet the items (the function also returns P objects)
            # processed = [result for item in raw if (result := do_screening(item))]
            processed = []
            for item in raw:
                result = do_screening(item)
                if result:
                    processed.append(result)

            if not processed:  # if empty, don't proceeed
                return List(processed)
            if win_order:  # this option only supported in non-generator mode.
                processed.sort(key=lambda x: [int(k) if k.isdigit() else k for k in re.split('([0-9]+)', x.stem)])
            return List(processed)

    def listdir(self):
        return List(os.listdir(self)).apply(P)

    def find(self, *args, r=True, **kwargs):
        """short for the method ``search`` then pick first item from results.

        .. note:: it is delibrately made to return None in case and object is not found.
        """
        results = self.search(*args, r=r, **kwargs)
        return results[0] if len(results) > 0 else None

    # def open_with_system(self):
    #     self.explore()  # if it is a file, it will be opened with its default program.

    @staticmethod
    def tmp(folder=None, fn=None, path="home", ex=False):
        """
        folder is created.
        file name is not created, only appended.
        """
        if str(path) == "home":
            path = P.home() / f"tmp_results"
            path.mkdir(exist_ok=True, parents=True)
        if folder is not None:
            path = path / folder
            path.mkdir(exist_ok=True, parents=True)
        if fn is not None:
            path = path / fn
        if ex:
            print(path.as_uri())
            print(path.parent.as_uri())
        return path

    @staticmethod
    def tmp_fname(name=None):
        return P.tmp(fn=(name or P.random()) + "-" + get_time_stamp())

    # ====================================== Compression ===========================================
    def zip(self, op_path=None, arcname=None, **kwargs):
        """
        """
        op_path = op_path or self
        arcname = arcname or self.name
        arcname = P(self.evalstr(arcname, expected="self"))
        op_path = P(self.evalstr(op_path, expected="self"))
        if arcname.name != self.name:
            arcname /= self.name  # arcname has to start from somewhere and end with filename
        if self.is_file():
            op_path = Compression.zip_file(ip_path=self, op_path=op_path, arcname=arcname, **kwargs)
        else:
            op_path = Compression.compress_folder(ip_path=self, op_path=op_path,
                                                  arcname=arcname, format_='zip', **kwargs)
        return op_path

    def unzip(self, op_path=None, fname=None, **kwargs):
        zipfile = self

        if self.suffix != ".zip":  # may be there is .zip somewhere in the path.
            assert ".zip" in str(self), f"Not a zip archive."
            zipfile, fname = self.split(at=".zip", sep=0)
            zipfile += ".zip"

        if op_path is None:
            op_path = zipfile.parent / zipfile.stem
        else:
            op_path = P(self.evalstr(op_path, expected="self"))
        return Compression.unzip(zipfile, op_path, fname, **kwargs)

    def compress(self, op_path=None, base_dir=None, format_="zip", **kwargs):
        formats = ["zip", "tar", "gzip"]
        assert format_ in formats, f"Unsupported format {format_}. The supported formats are {formats}"
        _ = self, op_path, base_dir, kwargs
        pass

    def decompress(self):
        pass


class Compression(object):
    """Provides consistent behaviour across all methods ...
    Both files and folders when compressed, default is being under the root of archive."""

    def __init__(self):
        pass

    @staticmethod
    def compress_folder(ip_path, op_path, arcname, format_='zip', **kwargs):
        """Explanation of Shutil parameters:

        * ``base_dir`` (here referred to as ``ip_path``) is what is going to be acturally archived.
            When provided, it **has to** be relevant to ``root_dir`` (here referred to as ``arcname``).
        * ``root_dir`` is where the archive is going to start from. It will create all the necessary subfolder till
            it reaches the ``base_dir`` where archiving actually starts.
        * Example: If you want to compress a folder in ``Downloads/myfolder/compress_this``
            Then, say that your rootdir is where you want the archive structure to include,
            then mention the folder you want to actually archive relatively to that root.

        .. note:: ``format_`` can only be one of ``zip, tar, gztar, bztar, xztar``.
        """
        root_dir = ip_path.split(at=arcname[0])[0]  # shutil works with folders nicely (recursion is done interally)
        result_path = shutil.make_archive(base_name=op_path, format=format_,
                                          root_dir=str(root_dir), base_dir=str(arcname), **kwargs)
        return P(result_path)  # same as op_path but (possibly) with format extension

    @staticmethod
    def zip_file(ip_path, op_path, arcname, **kwargs):
        """
        arcname determines the directory of the file being archived inside the archive. Defaults to same
        as original directory except for drive. When changed, it should still include the file name in its end.
        If arcname = filename without any path, then, it will be in the root of the archive.
        """
        import zipfile
        if op_path.suffix != ".zip":
            op_path = op_path + f".zip"
        jungle_zip = zipfile.ZipFile(str(op_path), 'w')
        jungle_zip.write(filename=str(ip_path), arcname=str(arcname), compress_type=zipfile.ZIP_DEFLATED, **kwargs)
        jungle_zip.close()
        return op_path

    @staticmethod
    def unzip(ip_path, op_path, fname=None, **kwargs):
        from zipfile import ZipFile
        with ZipFile(str(ip_path), 'r') as zipObj:
            if fname is None:  # extract all:
                zipObj.extractall(op_path, **kwargs)
            else:
                zipObj.extract(str(fname), str(op_path), **kwargs)
                op_path = P(op_path) / fname
        return P(op_path)

    @staticmethod
    def gz(file):
        import gzip
        with open(file, 'rb') as f_in:
            with gzip.open(str(file) + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def ungz(self, op_path=None):
        import gzip
        fn = str(self)
        op_path = op_path or self.parent / self.stem
        with gzip.open(fn, 'r') as f_in, open(op_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return P(op_path)

    @staticmethod
    def tar():
        # import tarfile
        pass

    @staticmethod
    def untar(self, fname=None, extract_dir='.', mode='r', **kwargs):
        import tarfile
        file = tarfile.open(str(self), mode)
        if fname is None:  # extract all files in the archive
            file.extractall(path=extract_dir, **kwargs)
        else:
            file.extract(fname, **kwargs)
        file.close()
        return fname


if __name__ == '__main__':
    pass
