"""
A collection of classes extending the functionality of Python's builtins.
email programmer@usa.com
"""

import re
import typing
import string
import enum
import os
import sys
from glob import glob
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% ========================== File Management  =========================================


class Base:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_saved(cls, path, *args, reader=None, **kwargs):
        """Whether the save format is .json, .mat, .pickle or .npy, Reader returns Structure
        For best experience, make sure that your subclass can be initialized with no or only fake inputs.
        """
        inst = cls(*args, **kwargs)
        if reader is None:
            data = Read.read(path)
        else:
            data = reader(path)
        # return inst, data
        new_data = data.dict if type(data) is Struct else data  # __setitem__ should be defined.
        inst.__dict__.update(new_data)
        return inst

    def save_npy(self, path, **kwargs):
        np.save(path, self.__dict__, **kwargs)

    def save_pickle(self, path, itself=False, **kwargs):
        """
        :param path:
        :param itself: determiens whether to save the weights only or the entire class.
        """
        if not itself:
            Save.pickle(path, self.__dict__, **kwargs)
        else:
            Save.pickle(path, self, **kwargs)

    def save_json(self, path, *args, **kwargs):
        """Use case: json is good for simple dicts, e.g. settings.
        Advantage: human-readable from file explorer."""
        _ = args
        Save.json(path, self.__dict__, **kwargs)
        return self

    def save_mat(self, path, *args, **kwargs):
        """for Matlab compatibility."""
        _ = args
        Save.mat(path, self.__dict__, **kwargs)
        return self

    def get_attributes(self):
        attrs = list(filter(lambda x: ('__' not in x) and not x.startswith("_"), dir(self)))
        return attrs
        # [setattr(Path, name, getattr(MyPath, name)) for name in funcs]

    # def get_methods(self):

    # def get_dict(self):
    #     return list(self.__dict__.keys())

    def __deepcopy__(self, *args, **kwargs):
        """Literally creates a new copy of values of old object, rather than referencing them.
        similar to copy.deepcopy()"""
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(copy.deepcopy(self.__dict__))
        return obj

    def __copy__(self, *args, **kwargs):
        """Shallow copy. New object, but the keys of which are referencing the values from the old object.
        Does similar functionality to copy.copy"""
        obj = self.__class__(*args, **kwargs)
        obj.__dict__.update(self.__dict__.copy())
        return obj

    def evalstr(self, string_, expected='self'):
        _ = self
        if type(string_) is str:
            if expected == 'func':
                return eval("lambda x: " + string_)
            elif expected == 'self':
                if "self" in string_:
                    return eval(string_)
                else:
                    return string_
        else:
            return string_


class P(type(Path()), Path, Base):
    """Path Class: Designed with one goal in mind: any operation on paths MUST NOT take more than one line of code.
    """

    # ===================================== File Specs ================================================================
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
        from datetime import datetime
        return datetime.fromtimestamp(time, **kwargs)

    # ================================ Path Object management ===========================================
    @property
    def trunk(self):
        """ useful if you have multiple dots in file name where .stem fails.
        """
        return self.name.split('.')[0]

    def __add__(self, name):
        return self.parent.joinpath(self.stem + name)

    def __sub__(self, other):
        return P(str(self).replace(str(other), ""))

    # def __rtruediv__(self, other):
    #     tmp = str(self)
    #     if tmp[0] == "/":  # if dir starts with this, all Path methods fail.
    #         tmp = tmp[1:]
    #     return P(other) / tmp

    def prepend(self, prefix, stem=False):
        """Add extra text before file name
        e.g: blah\blah.extenion ==> becomes ==> blah/name_blah.extension
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

    def split(self, at : str =None, index : int =None, sep: int= 1):
        """Splits a path at a given string or index

        :param self:
        :param at:
        :param index:
        :param sep: can be either [-1, 0, 1]. Determines where the separator is going to live with:
               left portion, none or right portion.
        :return: two paths
        """
        if index is None:  # at is provided
            items = str(self).split(sep=at)
            one, two = items[0], items[1]

            one = one[:-1] if one.endswith("/") else one
            two = two[1:] if two.startswith("/") else two

            one, two = P(one), P(two)

        else:
            one = self[:index]
            two = P(*self.parts[index + 1:])

        # appending `at` to one of the portions
        if sep == 0:
            pass  # neither of the portions get the sperator appended to it.
        elif sep == 1:  # append it to right portion
            two = at / two
        elif sep == -1:  # append it to left portion.
            one = one / at
        else:
            raise ValueError(f"`sep` should take a value from the set [-1, 0, 1] but got {sep}")

        return one, two

    def __getitem__(self, slici):
        if type(slici) is slice:
            return P(*self.parts[slici])
        elif type(slici) is list or type(slice) is np.ndarray:
            return P(*[self[item] for item in slici])
        else:
            return P(self.parts[slici])

    def __len__(self):
        return len(self.parts)

    @property
    def len(self):
        return self.__len__()

    def __setitem__(self, key, value):
        fullparts = list(self.parts)
        fullparts[key] = value
        return P(*fullparts)  # TODO: how to change self[-1]

    def switch(self, key: str, val: str):
        """Changes a given part of the path to another given one"""
        return P(str(self).replace(key, val))

    def switch_index(self, key: int, val: str):
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
        return "AlexPath(" + self.__str__() + ")"

    @property
    def string(self):  # this method is used by other functions to get string representation of path
        return str(self)

    def get_num(self, astring=None):
        if astring is None:
            astring = self.stem
        return int("".join(filter(str.isdigit, str(astring))))

    def make_valid_filename(self, replace='_'):
        return self.make_valid_filename_(self.trunk, replace=replace)

    @staticmethod
    def make_valid_filename_(astring, replace='_'):
        return re.sub(r'^(?=\d)|\W', replace, str(astring))

    @staticmethod
    def get_random_string(length=10, pool=None):
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
                import shutil
                shutil.rmtree(self, ignore_errors=True)
                # self.rmdir()  # dir must be empty
        else:
            print("File not deleted because user is not sure.")

    def send2trash(self):
        send2trash = Experimental.assert_package_installed("send2trash")
        send2trash.send2trash(self.string)

    def move(self, new_path):
        new_path = P(new_path)
        temp = self.absolute()
        temp.rename(new_path.absolute() / temp.name)
        return new_path

    def renameit(self, new_name):
        new_path = self.parent / new_name
        self.rename(new_path)
        return new_path

    def copy(self, target_dir=None, target_name=None, contents=False, verbose=False):
        """

        :param target_dir: copy the file to this directory.
        :param target_name: full path of destination (including -potentially different-  file name).
        :param contents: copy the parent directory or its contents (relevant only if copying a directory)
        :param verbose:
        :return: path to copied file or directory.

        .. wanring:: Do not confuse this with ``copy`` module that creates clones of Python objects.

        """
        dest = None  # destination.

        if target_dir is not None:
            assert target_name is None, f"You can either pass target_dir or target_name but not both"
            dest = P(target_dir).create()

        if target_name is not None:
            assert target_dir is None, f"You can either pass target_dir or target_name but not both"
            target_name = P(target_name)
            target_name.parent.create()
            dest = target_name

        if dest is None:
            dest = self.append(f"_copy__{get_time_stamp()}")

        if self.is_file():

            import shutil
            shutil.copy(str(self), str(dest))  # str() only there for Python < (3.6)
            if verbose:
                print(f"File \n{self}\ncopied successfully to: \n{dest}")
        elif self.is_dir():
            from distutils.dir_util import copy_tree
            if contents:
                copy_tree(str(self), str(dest))
            else:
                copy_tree(str(self), str(P(dest).joinpath(self.name).create()))
        else:
            print("Could not copy this thing. Not a file nor a folder.")
        return dest

    def clean(self):
        """removes contents on a folder, rather than deleting the folder."""
        contents = self.listdir()
        for content in contents:
            self.joinpath(content).send2trash()
        return self

    def readit(self, reader=None, notexist=None, **kwargs):
        filename = self
        if '.zip' in str(self):
            filename = self.unzip(op_path=tb.tmp("unzipped"))

        def func():
            if reader is None:
                return Read.read(filename, **kwargs)
            else:
                return reader(str(filename), **kwargs)

        if notexist is None:
            return func()
        else:
            try:
                return func()
            except Exception:
                return notexist

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

    def search(self, pattern='*', r=False, generator=False, files=True, folders=True, dotfiles=False,
               absolute=True, filters: list = None, not_in: list = None, win_order=False):
        """
        :param pattern:  linux search pattern
        :param r: recursive search flag
        :param generator: output format, list or generator.
        :param files: include files in search.
        :param folders: include directories in search.
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
                contents = L(z.namelist())
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

            # if os.name == 'nt':

        #     import win32api, win32con

        # def folder_is_hidden(p):
        #     if os.name == 'nt':
        #         attribute = win32api.GetFileAttributes(p)
        #         return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)

        def run_filter(item):
            flags = [True]
            if not files:
                flags.append(item.is_dir())
            if not folders:
                flags.append(item.is_file())
            for afilter in filters:
                flags.append(afilter(item))
            return all(flags)

        def do_screening(item):
            item = P(item)  # because some filters needs advanced functionalities of P objects.
            if absolute:
                item = item.absolute()

            if run_filter(item):
                return item
            else:
                return None

        if generator:
            def gen():
                flag = False
                while not flag:
                    item = next(raw)
                    flag = do_screening(item)
                    if flag:
                        yield item

            return gen
        else:
            # unpack the generator and vet the items (the function also returns P objects)
            processed = [result for item in raw if (result := do_screening(item))]
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
    def tmp(folder=None, fn=None, path="home"):
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
        return path

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


tmp = P.tmp


class Compression:
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
        root_dir = ip_path.split(at=arcname[0])[0]
        import shutil  # shutil works with folders nicely (recursion is done interally)
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
        import shutil
        with open(file, 'rb') as f_in:
            with gzip.open(str(file) + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def ungz(self, op_path=None):
        import shutil
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


class Read:

    @staticmethod
    def read(path, **kwargs):
        suffix = P(path).suffix[1:]
        # if suffix in ['eps', 'jpg', 'jpeg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']:
        #     # plt.gcf().canvas.get_supported_filetypes().keys():
        #     return plt.imread(path, **kwargs)
        # else:
        reader = getattr(Read, suffix)
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
    def mat(path, **kwargs):
        """
        :param path:
        :return: Structure object
        """
        from scipy.io import loadmat
        return Struct(loadmat(path, **kwargs))

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
        w = P(path).append(".dtypes").readit(reader=pd.read_csv, notexist=None)
        w = dict(zip(w['index'], w['dtypes'])) if w else w
        return pd.read_csv(path, dtypes=w, **kwargs)

    @staticmethod
    def pickle(path, **kwargs):
        # import pickle
        dill = Experimental.assert_package_installed("dill")
        with open(path, 'rb') as file:
            obj = dill.load(file, **kwargs)
        if type(obj) is dict:
            obj = Struct(obj)
        return obj

    @staticmethod
    def pkl(*args, **kwargs):
        return Read.pickle(*args, **kwargs)

    @staticmethod
    def csv(path, *args, **kwargs):
        return pd.read_csv(path, *args, **kwargs)


class Save:
    @staticmethod
    def csv(path, obj):
        obj.to_frame('dtypes').reset_index().to_csv(P(path).append(".dtypes").string)

    @staticmethod
    def mat(path=P.tmp(), mdict=None, **kwargs):
        """
        .. note::
            Avoid using mat for saving results because of incompatiblity:

            * `None` type is not accepted.
            * Scalars are conveteed to [1 x 1] arrays.
            * etc. As such, there is no gaurantee that you restore what you saved.

            Unless you want to pass the results to Matlab animals, avoid this format.
        """
        from scipy.io import savemat
        if '.mat' not in str(path):
            path += '.mat'
        path.parent.mkdir(exist_ok=True, parents=True)
        for key, value in mdict.items():
            if value is None:
                mdict[key] = []
        savemat(str(path), mdict, **kwargs)

    @staticmethod
    def json(path, obj, **kwargs):
        """This format is **compatible** with simple dictionaries that hold strings or numbers
         but nothing more than that.
        E.g. arrays or any other structure. An example of that is settings dictionary. It is useful because it can be
        inspected using any text editor."""
        import json
        if not str(path).endswith(".json"):
            path = str(path) + ".json"
        with open(str(path), "w") as file:
            json.dump(obj, file, default=lambda x: x.__dict__, **kwargs)

    @staticmethod
    def yaml(path, obj, **kwargs):
        import yaml
        if not str(path).endswith(".yaml"):
            path = str(path) + ".yaml"
        with open(str(path), "w") as file:
            yaml.dump(obj, file, **kwargs)

    # @staticmethod
    # def pickle(path, obj, **kwargs):
    #     if ".pickle" not in str(path):
    #         path = path + ".pickle"
    #     import pickle
    #     with open(str(path), 'wb') as file:
    #         pickle.dump(obj, file, **kwargs)

    @staticmethod
    def pickle(path, obj, **kwargs):
        dill = Experimental.assert_package_installed("dill")
        with open(str(path), 'wb') as file:
            dill.dump(obj, file, **kwargs)


def accelerate(func, ip):
    """ Conditions for this to work:
    * Must run under __main__ context
    * func must be defined outside that context.


    To accelerate IO-bound process, use multithreading. An example of that is somthing very cheap to process,
    but takes a long time to be obtained like a request from server. For this, multithreading launches all threads
    together, then process them in an interleaved fashion as they arrive, all will line-up for same processor,
    if it happens that they arrived quickly.

    To accelerate processing-bound process use multiprocessing, even better, use Numba.
    Method1 use: multiprocessing / multithreading.
    Method2: using joblib (still based on multiprocessing)
    from joblib import Parallel, delayed
    Fast method using Concurrent module
    """
    split = np.array_split(ip, os.cpu_count())
    # make each thread process multiple inputs to avoid having obscene number of threads with simple fast
    # operations

    # vectorize the function so that it now accepts lists of ips.
    # def my_func(ip):
    #     return [func(tmp) for tmp in ip]

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        op = executor.map(func, split)
        op = list(op)  # convert generator to list
    op = np.concatenate(op, axis=0)
    # op = self.reader.assign_resize(op, f=0.8, nrp=56, ncp=47, interpolation=True)
    return op


# %% ========================== Object Management ==============================================


class List(list, Base):
    """Use this class to keep items of the same type.
    """

    # =============================== Constructor Methods ====================
    def __init__(self, obj_list=None):
        super().__init__()
        self.list = list(obj_list) if obj_list is not None else []

    def __bool__(self):
        return bool(self.list)

    @classmethod
    def from_copies(cls, obj, count):
        return cls([copy.deepcopy(obj) for _ in range(count)])

    @classmethod
    def from_replicating(cls, func, *args, replicas=None, **kwargs):
        """
        :param args: could be one item repeated for all instances, or iterable. If iterable, it can by a Cycle object.
        :param kwargs: those could be structures:
        :param replicas:
        :param func:

        """
        if not args and not kwargs:  # empty args list and kwargs list
            return cls([func() for _ in range(replicas)])
        else:
            result = []
            for params in zip(*(args + tuple(kwargs.values()))):
                an_arg = params[:len(args)]
                a_val = params[len(args):]
                a_kwarg = dict(zip(kwargs.keys(), a_val))
                result.append(func(*an_arg, **a_kwarg))
            return cls(result)

    def save_items(self, directory, names=None, saver=None):
        if saver is None:
            saver = Save.pickle
        if names is None:
            names = range(len(self))
        for name, item in zip(names, self.list):
            saver(path=directory / name, obj=item)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
            _ = memodict
        return List([copy.deepcopy(i) for i in self.list])

    def __copy__(self):
        return List(self.list.copy())

    def __getstate__(self):
        return self.list

    def __setstate__(self, state):
        self.list = state

    # ================= call methods =====================================
    def method(self, name, *args, **kwargs):
        return List([getattr(i, name)(*args, **kwargs) for i in self.list])

    def attr(self, name):
        return List([getattr(i, name) for i in self.list])

    # def __getattribute__(self, item):
    #     # you can dispense with this method. Its only purpose is to make eaisr experience qwith the linter
    #     # obj = object.__getattribute__(self, "list")[0]
    #     # try:
    #     #     attr = object.__getattribute__(self, item)
    #     #     if hasattr(obj, item):
    #     #         return self.__getattr__(item)
    #     #     else:
    #     #         return attr
    #     # except AttributeError:
    #     #     return self.__getattr__(item)
    #     if item == "list":  # grant special access to this attribute.
    #         return object.__getattribute__(self, "list")
    #     if item in object.__getattribute__(self, "__dict__").keys():
    #         return self.__getattr__(item)
    #     else:
    #         return object.__getattribute__(self, item)

    def __getattr__(self, name):  # fallback position when normal mechanism fails.
        # this is called when __getattribute__ raises an error or call this explicitly.
        result = List([getattr(i, name) for i in self.list])
        return result

    def __call__(self, *args, lest=True, **kwargs):
        if lest:
            return List([i(*args, **kwargs) for i in self.list])
        else:
            return [i(*args, **kwargs) for i in self.list]

    # ======================== Access Methods ==========================================
    def __getitem__(self, key):
        if type(key) is list or type(key) is np.ndarray:  # to allow fancy indexing like List[1, 5, 6]
            return List([self[item] for item in key])

        # behaves similarly to Numpy A[1] vs A[1:2]
        result = self.list[key]  # return the required item only (not a List)
        if type(key) is not slice:
            return result  # choose one item
        else:
            return List(result)

    def __setitem__(self, key, value):
        self.list[key] = value

    def sample(self, size=1):
        return self[np.random.choice(len(self), size)]

    def to_struct(self, key_val=None):
        """
        :param key_val: function that returns (key, value) pair.
        :return:
        """
        if key_val is None:
            def key_val(x):
                return str(x), x
        else:
            key_val = self.evalstr(key_val)
        return Struct.from_keys_values_pairs(self.apply(key_val))

    # def find(self, patt, match="fnmatch"):
    #     """Looks up the string representation of all items in the list and finds the one that partially matches
    #     the argument passed. This method is a short for ``self.filter(lambda x: string_ in str(x))`` If you need more
    #     complicated logic in the search, revert to filter method.
    #     """
    #

        # if match == "string" or None:
        #     for idx, item in enumerate(self.list):
        #         if patt in str(item):
        #             return item
        # elif match == "fnmatch":
        #     import fnmatch
        #     for idx, item in enumerate(self.list):
        #         if fnmatch.fnmatch(str(item), patt):
        #             return item
        # else:  # "regex"
        #     # escaped = re.escape(string_)
        #     compiled = re.compile(patt)
        #     for idx, item in enumerate(self.list):
        #         if compiled.search(str(item)) is not None:
        #             return item
        # return None

    def index(self, func):
        """ A generalization of the `.index` method of `list`. It takes in a function rather than an
         item to find its index. Additionally, it returns full list of results, not just the first result.

        :param func:
        :return: List of indices of items where the function returns `True`.
        """
        func = self.evalstr(func)
        res = []
        for idx, x in enumerate(self.list):
            if func(x):
                res.append(idx)
        return res

    # ======================= Modify Methods ===============================
    def combine(self):
        res = self.list[0]
        for item in self.list[1:]:
            res = res + item
        return res

    def append(self, obj):
        self.list.append(obj)

    def __add__(self, other):
        return List(self.list + other.list)

    def __repr__(self):
        if len(self.list) > 0:
            tmp1 = f"AlexList object with {len(self.list)} elements. One example of those elements: \n"
            tmp2 = f"{self.list[0].__repr__()}"
            return tmp1 + tmp2
        else:
            return f"An Empty AlexList []"

    def __len__(self):
        return len(self.list)

    @property
    def len(self):
        return self.list.__len__()

    def __iter__(self):
        return iter(self.list)

    def apply(self, func, *args, lest=None, jobs=None, depth=1, verbose=False, **kwargs):
        """
        :param jobs:
        :param func: func has to be a function, possibly a lambda function. At any rate, it should return something.
        :param args:
        :param lest:
        :param verbose:
        :param depth: apply the function to inner Lists
        :param kwargs: a list of outputs each time the function is called on elements of the list.
        :return:
        """
        if depth > 1:
            depth -= 1
            # assert type(self.list[0]) == List, "items are not Lists".
            self.apply(lambda x: x.apply(func, *args, lest=lest, jobs=jobs, depth=depth, **kwargs))

        func = self.evalstr(func, expected='func')

        tqdm = 0
        if verbose or jobs:
            Experimental.assert_package_installed("tqdm")
            from tqdm import tqdm

        if lest is None:
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(i, *args, **kwargs) for i in tqdm(self.list)))
            else:
                iterator = self.list if not verbose else tqdm(self.list)
                return List([func(x, *args, **kwargs) for x in iterator])
        else:
            if jobs:
                from joblib import Parallel, delayed
                return List(Parallel(n_jobs=jobs)(delayed(func)(x, y) for x, y in tqdm(zip(self.list, lest))))
            else:
                iterator = zip(self.list, lest) if not verbose else tqdm(zip(self.list, lest))
                return List([func(x, y) for x, y in iterator])

    def modify(self, func, lest=None):
        """Modifies objects rather than returning new list of objects, hence the name of the method.
        :param func: a string that will be executed, assuming idx, x and y are given.
        :param lest:
        :return:
        """
        if lest is None:
            for x in self.list:
                _ = x
                exec(func)
        else:
            for idx, (x, y) in enumerate(zip(self.list, lest)):
                _ = idx, x, y
                exec(func)
        return self

    def sort(self, *args, **kwargs):
        self.list.sort(*args, **kwargs)
        return self

    def sorted(self, *args, **kwargs):
        return List(sorted(self.list, *args, **kwargs))

    def filter(self, func):
        if type(func) is str:
            func = eval("lambda x: " + func)
        result = List()
        for item in self.list:
            if func(item):
                result.append(item)
        return result

    def print(self, nl=1, sep=False, style=repr):
        for idx, item in enumerate(self.list):
            print(f"{idx:2}- {style(item)}", end=' ')
            for _ in range(nl):
                print('', end='\n')
            if sep:
                print(sep * 100)

    def to_dataframe(self, names=None, minimal=True):
        DisplayData.set_display()
        columns = ['object'] + list(self.list[0].__dict__.keys())
        df = pd.DataFrame(columns=columns)
        if minimal:
            return df
        for i, obj in enumerate(self.list):
            if names is None:
                name = [obj]
            else:
                name = [names[i]]
            df.loc[i] = name + list(self.list[i].__dict__.values())
        return df

    def to_numpy(self):
        return self.np

    @property
    def np(self):
        return np.array(self.list)


L = List


class Struct(Base):
    """Use this class to keep bits and sundry items.
    Combines the power of dot notation in classes with strings in dictionaries to provide Pandas-like experience
    """

    def __init__(self, dictionary=None, **kwargs):
        """
        :param dictionary: a dict, a Struct, None or an object with __dict__ attribute.
        """
        super(Struct, self).__init__()
        if type(dictionary) is Struct:
            dictionary = dictionary.dict
        if dictionary is None:  # only kwargs were passed
            final_dict = kwargs
        elif not kwargs:  # only dictionary was passed
            final_dict = dictionary if type(dictionary) is dict else dictionary.__dict__
        else:  # both were passed
            final_dict = dictionary if type(dictionary) is dict else dictionary.__dict__
            final_dict.update(kwargs)
        self.__dict__ = final_dict

    def __bool__(self):
        return bool(self.__dict__)

    @staticmethod
    def recursive_struct(mydict):
        struct = Struct(mydict)
        for key, val in struct.items():
            if type(val) is dict:
                struct[key] = Struct.recursive_struct(val)
        return struct

    @staticmethod
    def recursive_dict(struct):
        mydict = struct.dict
        for key, val in mydict.items():
            if type(val) is Struct:
                mydict[key] = Struct.recursive_dict(val)
        return mydict

    @classmethod
    def from_keys_values(cls, keys: list, values: list):
        return cls(dict(zip(keys, values)))

    @classmethod
    def from_keys_values_pairs(cls, my_list):
        res = dict()
        for k, v in my_list:
            res[k] = v
        return cls(res)

    @classmethod
    def from_names(cls, *names, default_=None):  # Mimick NamedTuple and defaultdict
        if default_ is None:
            default_ = [None] * len(names)
        return cls.from_keys_values(names, values=default_)

    def get_values(self, keys):
        return List([self[key] for key in keys])

    @property
    def clean_view(self):

        class Temp:
            pass

        temp = Temp()
        temp.__dict__ = self.__dict__
        return temp

    def __repr__(self):
        repr_string = ""
        for key in self.keys().list:
            repr_string += str(key) + ", "
        return "Structure, with following keys:\n" + repr_string

    def print(self, sep=20, yaml=False):
        if yaml:
            self.save_yaml(P.tmp(fn="__tmp.yaml"))
            txt = P.tmp(fn="__tmp.yaml").read_text()
            print(txt)
            return None
        repr_string = ""
        repr_string += "Structure, with following entries:\n"
        repr_string += "Key" + " " * sep + "Item Type" + " " * sep + "Item Details\n"
        repr_string += "---" + " " * sep + "---------" + " " * sep + "------------\n"
        for key in self.keys().list:
            key_str = str(key)
            type_str = str(type(self[key])).split("'")[1]
            val_str = DisplayData.get_repr(self[key])
            repr_string += key_str + " " * abs(sep - len(key_str)) + " " * len("Key")
            repr_string += type_str + " " * abs(sep - len(type_str)) + " " * len("Item Type")
            repr_string += val_str + "\n"
        print(repr_string)

    def __str__(self):
        mystr = str(self.__dict__)
        mystr = mystr[1:-1].replace(":", " =").replace("'", "")
        return mystr

    def __getitem__(self, item):  # allows indexing into entries of __dict__ attribute
        return self.__dict__[item]  # thus, gives both dot notation and string access to elements.

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, item):  # this works better with the linter.
        try:
            self.__dict__[item]
        except KeyError:
            # try:
            # super(Struct, self).__getattribute__(item)
            # object.__getattribute__(self, item)
            # except AttributeError:
            raise AttributeError(f"Could not find the attribute `{item}` in object `{self.__class__}`")

    def __getstate__(self):  # serialize
        return self.__dict__

    def __setstate__(self, state):  # deserialize
        self.__dict__ = state

    def __iter__(self):
        return iter(self.dict.items())

    def save_yaml(self, path):
        Save.yaml(path, self.recursive_dict(self))

    @property
    def dict(self):  # allows getting dictionary version without accessing private memebers explicitly.
        return self.__dict__

    @dict.setter
    def dict(self, adict):
        self.__dict__ = adict

    def update(self, *args, **kwargs):
        """Accepts dicts and keyworded args
        """
        new_struct = Struct(*args, **kwargs)
        self.__dict__.update(new_struct.__dict__)
        return self

    def apply(self, func):
        func = self.evalstr(func)
        for key, val in self.items():
            self[key] = func(val)
        return self

    def inverse(self):
        return Struct({v: k for k, v in self.dict.items()})

    def append_values(self, *others, **kwargs):
        """ """
        return Struct(self.concat_dicts(*((self.dict,) + others), **kwargs))

    @staticmethod
    def concat_values(*dicts, method=None, lenient=True, collect_items=False, clone=True):
        if method is None:
            method = list.__add__
        if not lenient:
            keys = dicts[0].keys()
            for i in dicts[1:]:
                assert i.keys() == keys
        # else if lenient, take the union
        if clone:
            total_dict = copy.deepcopy(dicts[0])  # take first dict in the tuple
        else:
            total_dict = dicts[0]  # take first dict in the tuple
        if collect_items:
            for key, val in total_dict.item():
                total_dict[key] = [val]

            def method(tmp1, tmp2):
                return tmp1 + [tmp2]

        if len(dicts) > 1:  # are there more dicts?
            for adict in dicts[1:]:
                for key in adict.keys():  # get everything from this dict
                    try:  # may be the key exists in the total dict already.
                        total_dict[key] = method(total_dict[key], adict[key])
                    except KeyError:  # key does not exist in total dict
                        if collect_items:
                            total_dict[key] = [adict[key]]
                        else:
                            total_dict[key] = adict[key]
        return Struct(total_dict)

    def keys(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.keys())

    def values(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.values())

    def items(self):
        """Same behaviour as that of `dict`, except that is doesn't produce a generator."""
        return List(self.dict.items())

    def to_dataframe(self, *args, **kwargs):
        # return self.values().to_dataframe(names=self.keys())
        return pd.DataFrame(self.__dict__, *args, **kwargs)

    def spawn_from_values(self, values):
        """From the same keys, generate a new Struct with different values passed."""
        return self.from_keys_values(self.keys(), self.evalstr(values, expected='self'))

    def spawn_from_keys(self, keys):
        """From the same values, generate a new Struct with different keys passed."""
        return self.from_keys_values(self.evalstr(keys, expected="self"), self.values())

    def plot(self, artist=None, xdata=None):
        if artist is None:
            artist = Artist(figname='Structure Plot')
        for key, val in self:
            if xdata is None:
                xdata = np.arange(len(val))
            artist.plot(xdata, val, label=key)
        try:
            artist.fig.legend()
        except AttributeError:
            pass
        return artist


class Cycle:
    def __init__(self, c, name=''):
        self.c = c
        self.index = -1
        self.name = name

    def __str__(self):
        return self.name

    def next(self):
        self.index += 1
        if self.index >= len(self.c):
            self.index = 0
        return self.c[self.index]

    def previous(self):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.c) - 1
        return self.c[self.index]

    def set(self, value):
        self.index = self.c.index(value)

    def get(self):
        return self.c[self.index]

    def get_index(self):
        return self.index

    def set_index(self, index):
        self.index = index

    def sample(self, size=1):
        return np.random.choice(self.c, size)

    def __add__(self, other):
        pass  # see behviour of matplotlib cyclers.


class Experimental:
    class Log:
        def __init__(self, path=None):
            if path is None:
                path = P('console_output')
            self.path = path + '.log'
            sys.stdout = open(self.path, 'w')

        def finish(self):
            sys.stdout.close()
            print(f"Finished ... have a look @ \n {self.path}")

    @staticmethod
    def assert_package_installed(package):
        try:
            pkg = __import__(package)
            return pkg
        except ImportError:
            # import pip
            # pip.main(['install', package])
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        pkg = __import__(package)
        return pkg

    @staticmethod
    def generate_readme(path, obj=None, meta=None, save_source_code=True):
        """Generates a readme file to contextualize any binary files.

        :param path: directory or file path.
        :param obj: Python module, class, method or function used to generate the result (not the result or an
            instance of any class)
        :param meta:
        :param save_source_code:
        """
        import inspect
        path = P(path)
        readmepath = path / f"README.md" if path.is_dir() else path

        separator = "\n" + "-----" + "\n\n"
        text = "# Meta\n"
        if meta is not None:
            text = text + meta
        text += separator

        if obj is not None:
            lines = inspect.getsource(obj)
            text += f"# Code to generate the result\n" + "```python\n" + lines + "\n```" + separator
            text += f"# Source code file generated me was located here: \n'{inspect.getfile(obj)}'\n" + separator

        readmepath.write_text(text)
        print(f"Successfully generated README.md file. Checkout:\n", readmepath.as_uri())

        if save_source_code:
            P(inspect.getmodule(obj).__file__).zip(op_path=readmepath.with_name("source_code.zip"))
            print(readmepath.with_name("source_code.zip").as_uri())

    @staticmethod
    def load_from_source_code(directory, obj=None):
        """Does the following:

        * Globs directory passed for ``source_code`` module.
        * Loads the directory to the memroy.
        * Returns either the package or a piece of it as indicated by ``obj``
        """
        P(directory).find("source_code*", r=True).unzip(tmpdir := P.tmp() / get_time_stamp(name="tmp_sourcecode"))
        sys.path.insert(0, str(tmpdir))
        sourcefile = __import__(tmpdir.find("*").stem)
        if obj is not None:
            loaded = getattr(sourcefile, obj)
            return loaded
        else:
            return sourcefile

    @staticmethod
    def get_locals(func):
        exec(Experimental.convert_to_global(func))  # run the function here.
        return Struct(vars())

    @staticmethod
    def update_globals(local_dict):
        sys.modules['__main__'].__dict__.update(local_dict)

    @staticmethod
    def in_main(func):  # a decorator
        def wrapper():  # a wrapper that remembers the function func because it was in the closure when construced.
            local_dict = Experimental.get_locals(func)
            Experimental.update_globals(local_dict)

        return wrapper

    @staticmethod
    def run_globally(func):
        exec(Experimental.convert_to_global(func))
        globals().update(vars())

    @staticmethod
    def convert_to_global(name):
        """Takes in a function name, reads it source code and returns a new version of it that can be run in the main.
        This is useful to debug functions and class methods alike.
        """
        import inspect
        import textwrap

        codelines = inspect.getsource(name)
        # remove def func_name() line from the list
        idx = codelines.find("):\n")
        header = codelines[:idx]
        codelines = codelines[idx + 3:]

        # remove any indentation (4 for funcs and 8 for classes methods, etc)
        codelines = textwrap.dedent(codelines)

        # remove return statements
        codelines = codelines.split("\n")
        codelines = [code + "\n" for code in codelines if not code.startswith("return ")]

        code_string = ''.join(codelines)  # convert list to string.

        temp = inspect.getfullargspec(name)
        arg_string = """"""
        # if isinstance(type(name), types.MethodType) else tmp.args
        if temp.defaults:  # not None
            for key, val in zip(temp.args[1:], temp.defaults):
                arg_string += f"{key} = {val}\n"
        if "*args" in header:
            arg_string += "args = (,)\n"
        if "**kwargs" in header:
            arg_string += "kwargs = {}\n"
        result = arg_string + code_string

        clipboard = Experimental.assert_package_installed("clipboard")
        clipboard.copy(result)
        print("code to be run \n", result, "=" * 100)
        return result  # ready to be run with exec()

    @staticmethod
    def edit_source(module, *edits):
        sourcelines = P(module.__file__).read_text().split("\n")
        for edit_idx, edit in enumerate(edits):
            line_idx = 0
            for line_idx, line in enumerate(sourcelines):
                if f"here{edit_idx}" in line:
                    new_line = line.replace(edit[0], edit[1])
                    print(f"Old Line: {line}\nNew Line: {new_line}")
                    if new_line == line:
                        raise KeyError(f"Text Not found.")
                    sourcelines[line_idx] = new_line
                    break
            else:
                raise KeyError(f"No marker found in the text. Place the following: 'here{line_idx}'")
        newsource = "\n".join(sourcelines)
        P(module.__file__).write_text(newsource)
        import importlib
        importlib.reload(module)
        return module

    @staticmethod
    def monkey_patch(class_inst, func):  # lambda *args, **kwargs: func(class_inst, *args, **kwargs)
        setattr(class_inst.__class__, func.__name__, func)

    @staticmethod
    def run_cell(pointer, module=sys.modules[__name__]):
        # update the module by reading it again.
        # if type(module) is str:
        #     module = __import__(module)
        # import importlib
        # importlib.reload(module)
        # if type(module) is str:
        #     sourcecells = P(module).read_text().split("#%%")
        # else:
        sourcecells = P(module.__file__).read_text().split("#%%")

        for cell in sourcecells:
            if pointer in cell.split('\n')[0]:
                break  # bingo
        else:
            raise KeyError(f"The pointer `{pointer}` was not found in the module `{module}`")
        print(cell)
        clipboard = Experimental.assert_package_installed("clipboard")
        clipboard.copy(cell)
        return cell


class Manipulator:
    @staticmethod
    def merge_adjacent_axes(array, ax1, ax2):
        """Multiplies out two axes to generate reduced order array.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        """
        shape = array.shape
        # order = len(shape)
        sz1, sz2 = shape[ax1], shape[ax2]
        new_shape = shape[:ax1] + (sz1 * sz2,) + shape[ax2 + 1:]
        return array.reshape(new_shape)

    @staticmethod
    def merge_axes(array, ax1, ax2):
        """Brings ax2 next to ax1 first, then combine the two axes into one.
        :param array:
        :param ax1:
        :param ax2:
        :return:
        """
        array2 = np.moveaxis(array, ax2, ax1 + 1)  # now, previously known as ax2 is located @ ax1 + 1
        return Manipulator.merge_adjacent_axes(array2, ax1, ax1 + 1)

    @staticmethod
    def expand_axis(array, ax_idx, factor):
        """opposite functionality of merge_axes.
        While ``numpy.split`` requires the division number, this requies the split size.
        """
        total_shape = list(array.shape)
        size = total_shape.pop(ax_idx)
        new_shape = (int(size / factor), factor)
        for index, item in enumerate(new_shape):
            total_shape.insert(ax_idx + index, item)
        # should be same as return np.split(array, new_shape, ax_idx)
        return array.reshape(tuple(total_shape))

    @staticmethod
    def slicer(array, a_slice: slice, axis=0):
        lower_ = a_slice.start
        upper_ = a_slice.stop
        n = len(array)
        lower_ = lower_ % n  # if negative, you get the positive equivalent. If > n, you get principal value.
        roll = lower_
        lower_ = lower_ - roll
        upper_ = upper_ - roll
        array_ = np.roll(array, -roll, axis=axis)
        upper_ = upper_ % n
        new_slice = slice(lower_, upper_, a_slice.step)
        return array_[Manipulator.indexer(axis=axis, myslice=new_slice, rank=array.ndim)]

    @staticmethod
    def indexer(axis, myslice, rank=None):
        """
        Returns a tuple of slicers.
        """
        everything = slice(None, None, None)  # `:`
        if rank is not None:
            indices = [everything] * rank
            indices[axis] = myslice
            return tuple(indices)
        else:
            indices = [everything] * (axis + 1)
            indices[axis] = myslice
            return tuple(indices)


def batcher(func_type='function'):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, x, *args, per_instance_kwargs=None, **kwargs):
                output = []
                for counter, item in enumerate(x):
                    if per_instance_kwargs is not None:
                        mykwargs = {key: value[counter] for key, value in per_instance_kwargs.items()}
                    else:
                        mykwargs = {}
                    output.append(func(self, item, *args, **mykwargs, **kwargs))
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __init__(self, func):
                self.func = func

            def __call__(self, x, **kwargs):
                output = [self.func(item, **kwargs) for item in x]
                return np.array(output)

        return Batch


def batcherv2(func_type='function', order=1):
    if func_type == 'method':
        def batch(func):
            # from functools import wraps
            #
            # @wraps(func)
            def wrapper(self, *args, **kwargs):
                output = [func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

            return wrapper

        return batch
    elif func_type == 'class':
        raise NotImplementedError
    elif func_type == 'function':
        class Batch(object):
            def __int__(self, func):
                self.func = func

            def __call__(self, *args, **kwargs):
                output = [self.func(self, *items, *args[order:], **kwargs) for items in zip(*args[:order])]
                return np.array(output)

        return Batch


class DisplayData:
    def __init__(self, x):
        self.x = pd.DataFrame(x)

    @staticmethod
    def set_display():
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 200)
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.max_rows', 1000)

    @staticmethod
    def eng():
        pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
        pd.options.display.float_format = '{:, .5f}'.format
        pd.set_option('precision', 7)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    @staticmethod
    def get_repr(data):
        """A well-behaved repr function for all data types."""
        if type(data) is np.ndarray:
            string_ = f"shape = {data.shape}, dtype = {data.dtype}."
            return string_
        elif type(data) is str:
            return data
        elif type(data) is list:
            return f"length = {len(data)}. 1st item type = {type(data[0]) if len(data) > 0 else None}"
        else:
            return repr(data)

    @staticmethod
    def outline(array, name="Array", imprint=True):
        str_ = f"{name}. Shape={array.shape}. Dtype={array.dtype}"
        if imprint:
            print(str_)
        return str_


# %% ========================== Plot Helper funcs ========================================


class FigurePolicy(enum.Enum):
    close_create_new = 'Close the previous figure that has the same figname and create a new fresh one'
    add_new = 'Create a new figure with same name but with added suffix'
    same = 'Grab the figure of the same name'


def get_time_stamp(ft=None, name=None):
    if ft is None:  # this is better than putting the default non-None value above.
        ft = '%Y-%m-%d-%I-%M-%S-%p-%f'  # if another function using this internally and wants to expise those kwarg
        # then it has to worry about not sending None which will overwrite this defualt value.
    from datetime import datetime
    _ = datetime.now().strftime(ft)
    if name:
        name = name + '_' + _
    else:
        name = _
    return name


class FigureManager:
    """
    Handles figures of matplotlib.
    """

    def __init__(self, info_loc=None, figpolicy=FigurePolicy.same):
        self.figpolicy = figpolicy
        self.fig = self.ax = self.event = None
        self.cmaps = Cycle(plt.colormaps())
        import matplotlib.colors as mcolors
        self.mcolors = list(mcolors.CSS4_COLORS.keys())
        self.facecolor = Cycle(list(mcolors.CSS4_COLORS.values()))
        self.colors = Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        self.cmaps.set('viridis')
        self.index = self.pause = self.index_max = None
        self.auto_brightness = False
        self.info_loc = [0.8, 0.01] if info_loc is None else info_loc
        self.pix_vals = False
        self.help_menu = {'_-=+[{]}\\': {'help': "Adjust Vmin Vmax. Shift + key applies change to all axes. \\ "
                                                 "toggles auto-brightness ", 'func': self.adjust_brightness},
                          "/": {'help': 'Show/Hide info text', 'func': self.text_info},
                          "h": {'help': 'Show/Hide help menu', 'func': self.show_help},
                          "tyTY": {'help': 'Change color map', 'func': self.change_cmap},
                          '<>': {'help': 'Change figure face color', 'func': self.change_facecolor},
                          "v": {'help': 'Show/Hide pixel values (Toggle)', 'func': self.show_pix_val},
                          'P': {'help': 'Pause/Proceed (Toggle)', 'func': self.pause_func},
                          'r': {'help': 'Replay', 'func': self.replay},
                          '1': {'help': 'Previous Image', 'func': self.previous},
                          '2': {'help': 'Next Image', 'func': self.next},
                          'S': {'help': 'Save Object', 'func': self.save},
                          'c': {'help': 'Show/Hide cursor', 'func': self.show_cursor},
                          'aA': {'help': 'Show/Hide ticks and their labels', 'func': self.show_ticks},
                          'alt+a': {'help': 'Show/Hide annotations', 'func': self.toggle_annotate}}
        # IMPORTANT: add the 'alt/ctrl+key' versions of key after the key in the dictionary above, not before.
        # Otherwise the naked key version will statisfy the condition `is key in this`? in the parser.
        self.message = ''
        self.message_obj = self.cursor = None
        self.annot_flag = False  # one flag for all axes?
        self.boundaries_flag = True

    @staticmethod
    def grid(ax, factor=5, x_or_y='both', color='gray', alpha1=0.5, alpha2=0.25):
        if type(ax) in {list, List, np.ndarray}:
            for an_ax in ax:
                FigureManager.grid(an_ax, factor=factor, x_or_y=x_or_y, color=color, alpha1=alpha1, alpha2=alpha2)
            return None

        # Turning on major grid for both axes.
        ax.grid(which='major', axis='x', color='gray', linewidth=0.5, alpha=alpha1)
        ax.grid(which='major', axis='y', color='gray', linewidth=0.5, alpha=alpha1)
        if x_or_y in {'both', 'x'}:
            xt = ax.get_xticks()  # major ticks
            steps = (xt[1] - xt[0]) / factor
            ax.xaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='x', color=color, linewidth=0.5, alpha=alpha2)
        if x_or_y in {'both', 'y'}:
            yt = ax.get_yticks()  # major ticks
            steps = (yt[1] - yt[0]) / factor
            ax.yaxis.set_minor_locator(plt.MultipleLocator(steps))
            ax.grid(which='minor', axis='y', color=color, linewidth=0.5, alpha=alpha2)

    def maximize_fig(self):
        _ = self
        # plt.get_current_fig_manager().window.state('zoom')
        plt.get_current_fig_manager().full_screen_toggle()

    def toggle_annotate(self, event):
        self.annot_flag = not self.annot_flag
        if event.inaxes:
            if event.inaxes.images:
                event.inaxes.images[0].set_picker(True)
                self.message = f"Annotation flag is toggled to {self.annot_flag}"
        if not self.annot_flag:  # if it is off
            pass  # hide all annotations

    def annotate(self, event, axis=None, data=None):
        self.event = event
        e = event.mouseevent
        if axis is None:
            ax = e.inaxes
        else:
            ax = axis
        if ax:
            if not hasattr(ax, 'annot_obj'):  # first time
                ax.annot_obj = ax.annotate("", xy=(0, 0), xytext=(-30, 30),
                                           textcoords="offset points",
                                           arrowprops=dict(arrowstyle="->", color="w", connectionstyle="arc3"),
                                           va="bottom", ha="left", fontsize=10,
                                           bbox=dict(boxstyle="round", fc="w"), )
            else:
                ax.annot_obj.set_visible(self.annot_flag)

            x, y = int(np.round(e.xdata)), int(np.round(e.ydata))
            if data is None:
                z = e.inaxes.images[0].get_array()[y, x]
            else:
                z = data[y, x]
            ax.annot_obj.set_text(f'x:{x}\ny:{y}\nvalue:{z:.3f}')
            ax.annot_obj.xy = (x, y)
            self.fig.canvas.draw_idle()

    def save(self, event):
        _ = event
        Save.pickle('.', obj=self)

    def replay(self, event):
        _ = event
        self.pause = False
        self.index = 0
        self.message = 'Replaying'
        self.animate()

    def pause_func(self, event):
        _ = event
        self.pause = not self.pause
        self.message = f'Pause flag is set to {self.pause}'
        self.animate()

    def previous(self, event):
        _ = event
        self.index = self.index - 1 if self.index > 0 else self.index_max - 1
        self.message = f'Previous {self.index}'
        self.animate()

    def next(self, event):
        _ = event
        self.index = self.index + 1 if self.index < self.index_max - 1 else 0
        self.message = f'Next {self.index}'
        self.animate()

    def animate(self):
        pass  # a method of the artist child class that is inheriting from this class

    def text_info(self, event):
        _ = event
        self.message = ''

    def show_help(self, event):
        _ = event
        default_plt = {"q ": {'help': "Quit Figure."},
                       "Ll": {'help': "change x/y scale to log and back to linear (toggle)"},
                       "Gg": {'help': "Turn on and off x and y grid respectively."},
                       "s ": {'help': "Save Figure"},
                       "f ": {'help': "Toggle Full screen"},
                       "p ": {'help': "Select / Deselect Pan"}}
        figs = plt.get_figlabels()
        if "Keyboard shortcuts" in figs:
            plt.close("Keyboard shortcuts")  # toggle
        else:
            fig = plt.figure(num="Keyboard shortcuts")
            for i, key in enumerate(self.help_menu.keys()):
                fig.text(0.1, 1 - 0.05 * (i + 1), f"{key:30s} {self.help_menu[key]['help']}")
            print(pd.DataFrame([[val['help'], key] for key, val in self.help_menu.items()], columns=['Action', 'Key']))
            print(f"\nDefault plt Keys:\n")
            print(pd.DataFrame([[val['help'], key] for key, val in default_plt.items()], columns=['Action', 'Key']))

    def adjust_brightness(self, event):
        ax = event.inaxes
        if ax is not None and ax.images:
            message = 'None'
            if event.key == '\\':
                self.auto_brightness = not self.auto_brightness
                message = f"Auto-brightness flag is set to {self.auto_brightness}"
                if self.auto_brightness:  # this change is only for the current image.
                    im = self.ax.images[0]
                    im.norm.autoscale(im.get_array())
                    # changes to all ims take place in animate as in ImShow and Nifti methods animate.
            vmin, vmax = ax.images[0].get_clim()
            if event.key in '-_':
                message = 'increase vmin'
                vmin += 1
            elif event.key in '[{':
                message = 'decrease vmin'
                vmin -= 1
            elif event.key in '=+':
                message = 'increase vmax'
                vmax += 1
            elif event.key in ']}':
                message = 'decrease vmax'
                vmax -= 1
            self.message = message + '  ' + str(round(vmin, 1)) + '  ' + str(round(vmax, 1))
            if event.key in '_+}{':
                for ax in self.fig.axes:
                    if ax.images:
                        ax.images[0].set_clim((vmin, vmax))
            else:
                if ax.images:
                    ax.images[0].set_clim((vmin, vmax))

    def change_cmap(self, event):
        ax = event.inaxes
        if ax is not None:
            cmap = self.cmaps.next() if event.key in 'tT' else self.cmaps.previous()
            if event.key in 'TY':
                for ax in self.fig.axes:
                    for im in ax.images:
                        im.set_cmap(cmap)
            else:
                for im in ax.images:
                    im.set_cmap(cmap)
            self.message = f"Color map changed to {ax.images[0].cmap.name}"

    def change_facecolor(self, event):
        color = self.facecolor.next() if event.key == '>' else self.facecolor.previous()
        self.fig.set_facecolor(color)
        self.message = f"Figure facecolor was set to {self.mcolors[self.facecolor.get_index()]}"

    def show_pix_val(self, event):
        ax = event.inaxes
        if ax is not None:
            self.pix_vals = not self.pix_vals  # toggle
            self.message = f"Pixel values flag set to {self.pix_vals}"
            if self.pix_vals:
                self.show_pixels_values(ax)
            else:
                while len(ax.texts) > 0:
                    for text in ax.texts:
                        text.remove()

    def process_key(self, event):
        self.event = event  # useful for debugging.
        for key in self.help_menu.keys():
            if event.key in key:
                self.help_menu[key]['func'](event)
                break
        self.update_info_text(self.message)
        if event.key != 'q':  # for smooth quit without throwing errors
            fig = event.canvas.figure  # don't update if you want to quit.
            fig.canvas.draw()

    def update_info_text(self, message):
        if self.message_obj:
            self.message_obj.remove()
        self.message_obj = self.fig.text(*self.info_loc, message, fontsize=8)

    @staticmethod
    def get_nrows_ncols(num_plots, nrows=None, ncols=None):
        if not nrows and not ncols:
            nrows = int(np.floor(np.sqrt(num_plots)))
            ncols = int(np.ceil(np.sqrt(num_plots)))
            while nrows * ncols < num_plots:
                ncols += 1
        elif not ncols and nrows:
            ncols = int(np.ceil(num_plots / nrows))
        elif not nrows and ncols:
            nrows = int(np.ceil(num_plots / ncols))
        else:
            pass
        return nrows, ncols

    def show_cursor(self, event):
        ax = event.inaxes
        if ax:  # don't do this if c was pressed outside an axis.
            if hasattr(ax, 'cursor_'):  # is this the first time?
                if ax.cursor_ is None:
                    from matplotlib import widgets
                    ax.cursor_ = widgets.Cursor(ax=ax, vertOn=True, horizOn=True, color='red', lw=1.0)
                else:  # toggle the cursor.
                    ax.cursor_ = None
                self.message = f'Cursor flag set to {bool(ax.cursor_)}'
            else:  # first call
                ax.cursor_ = None
                self.show_cursor(event)

    def show_ticks(self, event):
        self.boundaries_flag = not self.boundaries_flag
        axis = event.inaxes
        if event.key == 'a':
            if axis:
                # event.inaxes.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(axis)
                self.message = f"Boundaries flag set to {self.boundaries_flag} in {axis}"

        else:
            for ax in self.ax:
                # ax.axis(['off', 'on'][self.boundaries_flag])
                self.toggle_ticks(ax)

    @staticmethod
    def toggle_ticks(an_ax, state=None):
        for line in an_ax.get_yticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklines():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_xticklabels():
            line.set_visible(not line.get_visible() if state is None else state)
        for line in an_ax.get_yticklabels():
            line.set_visible(not line.get_visible() if state is None else state)

    def clear_axes(self):
        for ax in self.ax:
            ax.cla()

    @staticmethod
    def show_pixels_values(ax):
        im = ax.images[0].get_array()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if ymin > ymax:  # default imshow settings
            ymin, ymax = ymax, ymin
        for (j, i), label in np.ndenumerate(im):
            if (xmin < i < xmax) and (ymin < j < ymax):
                ax.text(i, j, np.round(label).__int__(), ha='center', va='center', size=8)

    @staticmethod
    def update(figname, obj_name, data=None):
        """Fastest update ever. But, you need access to label name.
        Using this function external to the plotter. But inside the plotter you need to define labels to objects
        The other alternative is to do the update inside the plotter, but it will become very verbose.

        :param figname:
        :param obj_name:
        :param data:
        :return:
        """
        obj = FigureManager.findobj(figname, obj_name)
        if data is not None:
            obj.set_data(data)
            # update scale:
            # obj.axes.relim()
            # obj.axes.autoscale()

    @staticmethod
    def findobj(figname, obj_name):
        if type(figname) is str:
            fig = plt.figure(num=figname)
        else:
            fig = figname
        search_results = fig.findobj(lambda x: x.get_label() == obj_name)
        if len(search_results) > 0:  # list of length 1, 2 ...
            search_results = search_results[0]  # the first one is good enough.
        return search_results

    def get_fig(self, figname='', suffix=None, **kwargs):
        return FigureManager.get_fig_static(self.figpolicy, figname, suffix, **kwargs)

    @staticmethod
    def get_fig_static(figpolicy, figname='', suffix=None, **kwargs):
        """
        :param figpolicy:
        :param figname:
        :param suffix: only relevant if figpolicy is add_new
        :param kwargs:
        :return:
        """
        fig = None
        exist = True if figname in plt.get_figlabels() else False
        if figpolicy is FigurePolicy.same:
            fig = plt.figure(num=figname, **kwargs)
        elif figpolicy is FigurePolicy.add_new:
            if exist:
                new_name = get_time_stamp(name=figname) if suffix is None else figname + suffix
            else:
                new_name = figname
            fig = plt.figure(num=new_name, **kwargs)
        elif figpolicy is FigurePolicy.close_create_new:
            if exist:
                plt.close(figname)
            fig = plt.figure(num=figname, **kwargs)
        return fig

    def transperent_fig(self):
        self.fig.canvas.manager.window.attributes("-transparentcolor", "white")

    @staticmethod
    def set_ax_size(ax, w, h):
        """ w, h: width, height in inches """
        left = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - left)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    @staticmethod
    def get_ax_size(ax):  # returns axis size in inches.
        w, h = ax.figure.get_size_inches()
        width = ax.figure.subplotpars.right - ax.figure.subplotpars.left
        height = ax.figure.subplotpars.top - ax.figure.subplotpars.bottom
        # width, height = ax.figbox.extents[2:] - ax.figbox.extents[:2]
        return w * width, h * height

    @staticmethod
    def set_ax_to_real_life_size(ax, inch_per_unit=1 / 25.4):
        limit_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        limit_y = ax.get_ylim()[1] - ax.get_ylim()[0]
        FigureManager.set_ax_size(ax, limit_x * inch_per_unit, limit_y * inch_per_unit)

    @staticmethod
    def try_figure_size():
        fig, ax = plt.subplots()
        x = np.arange(0, 100, 0.01)
        y = np.sin(x) * 100
        ax.plot(x, y)
        ax.axis("square")
        ax.set_xlim(0, 100)
        ax.set_ylim(-100, 100)
        FigureManager.set_ax_to_real_life_size(ax)
        fig.savefig(P.tmp() / "trial.png", dpi=250)

    @staticmethod
    def write(txt, name="text", size=8, **kwargs):
        fig = plt.figure(figsize=(11.69, 8.27), num=name)
        FigureManager.maximize_fig(fig)
        fig.clf()
        fig.text(0.5, 0.5, txt, transform=fig.transFigure, size=size, ha="center", va='center', **kwargs)
        return fig

    @staticmethod
    def activate_latex(size=20):
        """Setting up matplotlib"""
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('axes', labelsize=size)
        plt.rc('axes', titlesize=size)
        plt.rc('legend', fontsize=size / 1.5)
        # rc('text', usetex=True)
        plt.rcParams['text.usetex'] = True
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    @staticmethod
    def set_linestyles_and_markers_and_colors(test=False):
        from cycler import cycler
        from matplotlib import lines
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = list(lines.lineMarkers.keys())[:-4]  # ignore the None
        linestyles = list(lines.lineStyles.keys())[:-3]  # ignore the Nones
        linestyles = (linestyles * 10)[:len(markers)]
        colors = (colors * 10)[:len(markers)]
        default_cycler = (cycler(linestyle=linestyles) + cycler(marker=markers) + cycler(color=colors))
        plt.rc('axes', prop_cycle=default_cycler)
        if test:
            temp = np.random.randn(10, 10)
            for idx, aq in enumerate(temp):
                plt.plot(aq + idx * 2)

    def close(self):
        plt.close(self.fig)


class SaveType:
    class GenericSave:
        """ You can either pass the figures to be tracked or, pass them dynamically at add method, or,
        add method will capture every figure and axis

        """
        stream = ['clear', 'accumulate', 'update'][0]

        def __init__(self, save_dir=None, save_name=None, watch_figs=None, max_calls=2000, delay=100, **kwargs):
            self.delay = delay
            self.watch_figs = watch_figs
            if watch_figs:
                assert type(watch_figs) is list, "This should be a list"
                if type(watch_figs[0]) is str:
                    self.watch_figs = [plt.figure(num=afig) for afig in watch_figs]

            save_dir = save_dir or P.tmp().string
            self.save_name = get_time_stamp(name=save_name)
            self.save_dir = save_dir
            self.kwargs = kwargs
            self.counter = 0
            self.max = max_calls

        def add(self, fignames=None, names=None, **kwargs):
            print(f"Saver added frame number {self.counter}", end='\r')
            self.counter += 1
            plt.pause(self.delay * 0.001)
            if self.counter > self.max:
                print('Turning off IO')
                plt.ioff()

            if fignames:  # name sent explicitly
                self.watch_figs = [plt.figure(figname) for figname in fignames]
            else:  # tow choices:
                if self.watch_figs is None:  # None exist ==> add all
                    figure_names = plt.get_figlabels()  # add all.
                    self.watch_figs = [plt.figure(k) for k in figure_names]
                else:  # they exist already.
                    pass

            if names is None:  # individual save name, useful for PNG.
                names = [get_time_stamp(name=a_figure.get_label()) for a_figure in self.watch_figs]

            for afig, aname in zip(self.watch_figs, names):
                self._save(afig, aname, **kwargs)

        def _save(self, *args, **kwargs):
            pass

    class Null(GenericSave):
        """ Use this when you do not want to save anything. This class will help plot to work faster
        by removing lines of previous plot, so you get live animation cheaply.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fname = self.save_dir

        def finish(self):
            print(f"Nothing saved by {self}")
            return self.fname

    class PDF(GenericSave):
        """For pdf, you just need any figure manager, [update, clear, accumalate], preferabbly fastest.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from matplotlib.backends.backend_pdf import PdfPages
            self.fname = os.path.join(self.save_dir, self.save_name + '.pdf')
            self.pp = PdfPages(self.fname)

        def _save(self, a_fig, a_name, bbox_inches='tight', pad_inches=0.3, **kwargs):
            self.pp.savefig(a_fig, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

        def finish(self, open_result=True):
            print(f"Saving results ...")
            self.pp.close()
            print(f"PDF Saved @", P(self.fname).absolute().as_uri())
            if open_result:
                import webbrowser as wb
                # chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe".replace('\\', '/')
                # wb.register('chrome', None, wb.BackgroundBrowser(chrome_path))
                # wb.get('chrome').open(self.fname)
                wb.open(self.fname)
            return self

    class PNG(GenericSave):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.save_dir = os.path.join(self.save_dir, self.save_name)
            os.makedirs(self.save_dir, exist_ok=True)
            self.fname = self.save_dir

        def _save(self, afigure, aname, dpi=150, **kwargs):
            aname = P(aname).make_python_name()
            afigure.savefig(os.path.join(self.save_dir, aname), bbox_inches='tight', pad_inches=0.3,
                            dpi=dpi, **kwargs)

        def finish(self):
            print(f"PNGs Saved @", Path(self.fname).absolute().as_uri())
            return self.fname

    class GIF(GenericSave):
        """Requirements: same axis must persist, only new objects are drawn inside it.
        This is not harsh as no one wants to add multiple axes on top of each other.
        Next, the objects drawn must not be removed, or updated, instead they should pile up in axis.

        # do not pass names in the add method. names will be extracted from figures.
        # usually it is smoother when adding animate=True to plot or imshow commands for GIF purpose

        Works for images only. Add more .imshow to the same axis, and that's it. imshow will conver up previous images.
        For lines, it will superimpose it and will look ugly.

        If you clear the axis, nothing will be saved. This should not happend.
        The class will automatically detect new lines by their "neo" labels.
        and add them then hide them for the next round.
        Limitation of ArtistAnimation: works on lines and images list attached to figure axes.
        Doesn't work on axes, unless you add large number of them. As such, titles are not incorporated etc.
        """

        def __init__(self, interval=100, **kwargs):
            super().__init__(**kwargs)
            from collections import defaultdict
            self.container = defaultdict(lambda: [])
            self.interval = interval
            self.fname = None  # determined at finish time.

        def _save(self, afigure, aname, cla=False, **kwargs):
            fig_list = self.container[afigure.get_label()]
            subcontainer = []

            search = FigureManager.findobj(afigure, 'neo')
            for item in search:
                item.set_label('processed')
                item.set_visible(False)
                subcontainer += [item]

            fig_list.append(subcontainer)
            # if you want the method coupled with cla being used in main, then it add_line is required for axes.

        def finish(self):
            print("Saving the GIF ....")
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
            for idx, a_fig in enumerate(self.watch_figs):
                ims = self.container[a_fig.get_label()]
                if ims:
                    ani = animation.ArtistAnimation(a_fig, ims,
                                                    interval=self.interval, blit=True, repeat_delay=1000)
                    self.fname = os.path.join(self.save_dir, f'{a_fig.get_label()}_{self.save_name}.gif')
                    ani.save(self.fname, writer=PillowWriter(fps=4))
                    # if you don't specify the writer, it goes to ffmpeg by default then try others if that is not
                    # available, resulting in behaviours that is not consistent across machines.
                    print(f"GIF Saved @", Path(self.fname).absolute().as_uri())
                else:
                    print(f"Nothing to be saved by GIF writer.")
                    return self.fname

    class GIFFileBased(GenericSave):
        def __init__(self, fps=4, dpi=100, bitrate=1800, _type='GIFFileBased', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBased':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBased':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBased':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'
            self.writer = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)
            assert self.watch_figs, "No figure was sent during instantiation of saver, therefore the writer cannot" \
                                    "be setup. Did you mean to use an autosaver?"
            self.writer.setup(fig=self.watch_figs[0], outfile=self.fname, dpi=dpi)

        def _save(self, afig, aname, **kwargs):
            self.writer.grab_frame(**kwargs)

        def finish(self):
            print('Saving results ...')
            self.writer.finish()
            print(f"Saved @", Path(self.fname).absolute().as_uri())
            return self.fname

    class GIFPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBased(GIFFileBased):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    """
    Parses the data automatically.
    For all subclasses, you need to provide a plotter class with animate method implemetend.
    You also need to have .fig attribute.
    """

    class GenericAuto(GenericSave):
        save_type = 'auto'

        def __init__(self, plotter_class, data, names_list=None, **kwargs):
            super().__init__(**kwargs)
            self.plotter_class = plotter_class
            self.data = data
            self.names_list = names_list
            self.kwargs = kwargs
            self.data_gen = None
            self.saver = None
            self.plotter = None

        def animate(self):
            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.data_gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **self.kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them
            Experimental.assert_package_installed("tqdm")
            from tqdm import tqdm
            for idx, datum in tqdm(enumerate(self.data_gen())):
                self.plotter.animate(datum)
                self.saver.add(names=[self.names_list[idx]])
            self.saver.finish()

    class GIFAuto(GenericAuto):
        def __init__(self, plotter_class, data, interval=500, extension='gif', fps=4, **kwargs):
            super().__init__(plotter_class, data, **kwargs)
            writer = None
            from matplotlib import animation
            if extension == 'gif':
                writer = animation.PillowWriter(fps=fps)
            elif extension == 'mp4':
                writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=2500)

            def gen_function():
                for i in zip(*self.data):
                    yield i

            self.gen = gen_function
            self.plotter = self.plotter_class(*[piece[0] for piece in self.data], **kwargs)
            plt.pause(self.delay * 0.001)  # give time for figures to show up before updating them
            self.ani = animation.FuncAnimation(self.plotter.fig, self.plotter.animate, frames=self.gen,
                                               interval=interval, repeat_delay=1500, fargs=None,
                                               cache_frame_data=True, save_count=10000)
            fname = f"{os.path.join(self.save_dir, self.save_name)}.{extension}"
            self.fname = fname
            self.ani.save(filename=fname, writer=writer)
            print(f"Saved @", Path(self.fname).absolute().as_uri())

    class PDFAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PDF(**kwargs)
            self.animate()

    class PNGAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.PNG(**kwargs)
            self.save_dir = self.saver.save_dir
            self.animate()
            self.fname = self.saver.fname

    class NullAuto(GenericAuto):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.saver = SaveType.Null(**kwargs)
            self.fname = self.saver.fname
            self.animate()

    class GIFFileBasedAuto(GenericAuto):
        def __init__(self, plotter_class, data, fps=4, dpi=150, bitrate=2500,
                     _type='GIFFileBasedAuto', **kwargs):
            super().__init__(**kwargs)
            from matplotlib.animation import ImageMagickWriter as Writer
            extension = '.gif'
            if _type == 'GIFPipeBasedAuto':
                from matplotlib.animation import ImageMagickFileWriter as Writer
            elif _type == 'MPEGFileBasedAuto':
                from matplotlib.animation import FFMpegFileWriter as Writer
                extension = '.mp4'
            elif _type == 'MPEGPipeBasedAuto':
                from matplotlib.animation import FFMpegWriter as Writer
                extension = '.mp4'

            self.saver = Writer(fps=fps, metadata=dict(artist='Alex Al-Saffar'), bitrate=bitrate)
            self.fname = os.path.join(self.save_dir, self.save_name + extension)

            def gen_function():
                for i in zip(*data):
                    yield i

            self.data = gen_function
            self.plotter = plotter_class(*[piece[0] for piece in data], **kwargs)
            plt.pause(0.5)  # give time for figures to show up before updating them
            Experimental.assert_package_installed("tqdm")
            from tqdm import tqdm
            with self.saver.saving(fig=self.plotter.fig, outfile=self.fname, dpi=dpi):
                for datum in tqdm(self.data()):
                    self.plotter.animate(datum)
                    self.saver.grab_frame()
                    plt.pause(self.delay * 0.001)
            print(f"Results saved successfully @ {self.fname}")

    class GIFPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGFileBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)

    class MPEGPipeBasedAuto(GIFFileBasedAuto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, _type=self.__class__.__name__, **kwargs)


class VisibilityViewer(FigureManager):
    artist = ['internal', 'external'][1]
    parser = ['internal', 'external'][1]
    stream = ['clear', 'accumulate', 'update'][1]
    """
    **Viewer Building Philosophy**: 
    
    Viewer should act as Saver and Browser:

    * How is the data viewed:
    
        * Can either be an artist himself, as in ImShow.
        * external artist is required to view the data (especially non image data)

    * Data parsing:

        * internal for loop to go through all the dataset passed.
            # Allows manual control over parsing.
        * external for loop. It should have add method.
            # Manual control only takes place after the external loop is over. #TODO parallelize this.

    * Refresh mechanism.

        * Clear the axis. (slowest, but easy on memory)
        * accumulate, using visibility to hide previous axes. (Fastest but memory intensive)
        * The artist has an update method. (best)

    The artist has to have:
    
        * fig, ax, txt attributes. ax and txt should be lists.
        * the ax and txt attributes should always belong to the same figure.
    
    Here and in all Visibility classes, the artist is assumed to be always creating new axes along the way.
    """

    def __init__(self, artist=None, hide_artist_axes=True):
        """
        This class works on hiding axes shown on a plot, so that a new plot can be drawn.
        Hiding is done via the method `add`.
        Thus, an external loop is required to parse through the plots one by one.
        Once the entire loop is finished, you can browse through the plots with the keyboard
        Animation is done bia method `animate`

        :param artist: A class that draws on one figure. It should have `.fig` attribute.
                       Can either be passed during instantiation, or everytime when `add` is called.
        :param hide_artist_axes:
        """
        super().__init__()
        self.index = -1
        self.index_max = 0
        self.current = None
        self.axes_repo = []
        self.texts_repo = []
        self.fig = None
        if artist:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)
            self.add(artist=artist, hide_artist_axes=hide_artist_axes)

    def add(self, artist=None, increment_index=True, hide_artist_axes=True):
        if artist is not None:
            self.artist = artist
        if self.fig is None:
            self.fig = artist.fig
            self.fig.canvas.mpl_connect('key_press_event', self.process_key)

        if increment_index:
            self.index += 1
            self.index_max += 1
        self.current = self.index
        self.axes_repo.append(self.artist.ax if type(self.artist.ax) is list else self.artist.ax.tolist())
        self.texts_repo.append(self.artist.txt if type(self.artist.txt) is list else self.artist.txt.tolist())
        print(f"VViewer added plot number {self.index}", end='\r')
        if hide_artist_axes:
            self.hide_artist_axes()

    def hide_artist_axes(self):
        for ax in self.artist.ax:
            ax.set_visible(False)
        for text in self.artist.txt:
            text.set_visible(False)

    def finish(self):  # simply: undo the last hiding
        self.current = self.index
        self.animate()

    def animate(self):
        # remove current axes and set self.index as visible.
        for ax in self.axes_repo[self.current]:
            ax.set_visible(False)
        for text in self.texts_repo[self.current]:
            text.set_visible(False)
        for ax in self.axes_repo[self.index]:
            ax.set_visible(True)
        for text in self.texts_repo[self.index]:
            text.set_visible(True)
        self.current = self.index
        self.fig.canvas.draw()


class VisibilityViewerAuto(VisibilityViewer):
    def __init__(self, data=None, artist=None, memorize=False, transpose=True, save_type=SaveType.Null,
                 save_dir=None, save_name=None, delay=1,
                 titles=None, legends=None, x_labels=None, pause=True, **kwargs):
        """
        The difference between this class and `VisibilityViewer` is that here the parsing of data is done
        internally, hence the suffix `Auto`.

        :param data: shoud be of the form [[ip1 list], [ip2 list], ...]
                     i.e. NumArgsPerPlot x NumInputsForAnimation x Input (possible points x signals)
        :param artist: an instance of a class that subclasses `Artist`
        :param memorize: if set to True, then axes are hidden and shown again, otherwise, plots constructed freshly
                         every time they're shown (axes are cleaned instead of hidden)
        """
        self.kwargs = kwargs
        self.memorize = memorize
        self.max_index_memorized = 0

        if transpose:
            data = np.array(list(zip(*data)))
        self.data = data
        self.legends = legends
        if legends is None:
            self.legends = [f"Curve {i}" for i in range(len(self.data))]
        self.titles = titles if titles is not None else np.arange(len(self.data))
        self.lables = x_labels

        if artist is None:
            artist = Artist(*self.data[0], title=self.titles[0], legends=self.legends, create_new_axes=True,
                            **kwargs)
        else:
            artist.plot(*self.data[0], title=self.titles[0], legends=self.legends)
            if memorize:
                assert artist.create_new_axes is True, "Auto Viewer is based on hiding and showing and requires new " \
                                                       "axes from the artist with every plot"
        self.artist = artist
        super().__init__(artist=self.artist, hide_artist_axes=False)
        self.index_max = len(self.data)
        self.pause = pause
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay)
        self.fname = None

    def animate(self):
        for i in range(self.index, self.index_max):
            datum = self.data[i]
            if self.memorize:  # ==> plot and use .add() method
                if self.index > self.max_index_memorized:  # a new plot never done before
                    self.hide_artist_axes()
                    self.artist.plot(*datum, title=self.titles[i], legends=self.legends)
                    self.add(increment_index=False, hide_artist_axes=False)  # index incremented via press_key manually
                    self.max_index_memorized += 1
                else:  # already seen this plot before ==> use animate method of parent class to hide and show,
                    # not plot and add
                    # print(f"current = {self.current}")
                    super().animate()
            else:
                self.fig.clf()  # instead of making previous axis invisible, delete it completely.
                self.artist.plot(*datum, title=self.titles[i], legends=self.legends)
                # replot the new data point on a new axis.
            self.saver.add()
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    @staticmethod
    def test():
        return VisibilityViewerAuto(data=np.random.randn(1, 10, 100, 3))


class ImShow(FigureManager):
    artist = ['internal', 'external'][0]
    parser = ['internal', 'external'][0]
    stream = ['clear', 'accumulate', 'update'][2]

    def __init__(self, *images_list: typing.Union[list, np.ndarray], sup_titles=None, sub_labels=None, labels=None,
                 save_type=SaveType.Null, save_name=None, save_dir=None, save_kwargs=None,
                 subplots_adjust=None, gridspec=None, tight=True, info_loc=None,
                 nrows=None, ncols=None, ax=None,
                 figsize=None, figname='im_show', figpolicy=FigurePolicy.add_new,
                 auto_brightness=True, delay=200, pause=False,
                 **kwargs):
        """
        :param images_list: arbitrary number of image lists separated by comma, say N.
        :param sup_titles: Titles for frames. Must have a length equal to number of images in each list, say M.
        :param sub_labels: Must have a length = M, and in each entry there should be N labels.
        :param labels: if labels are sent via this keyword, they will be repeated for all freames.
        :param save_type:
        :param save_dir:
        :param save_name:
        :param nrows:
        :param ncols:
        :param delay:
        :param kwargs: passed to imshow

        Design point: The expected inputs are made to be consistent with image_list passed. Labels and titles passed
        should have similar structure. The function internally process them.

        Tip: Use np.arrray_split to get sublists and have multiple plots per frame. Useful for very long lists.
        """
        super(ImShow, self).__init__(info_loc=info_loc)

        num_plots = len(images_list)  # Number of images in each plot
        self.num_plots = num_plots
        lengths = [len(images_list[i]) for i in range(num_plots)]
        self.index_max = min(lengths)
        nrows, ncols = self.get_nrows_ncols(num_plots, nrows, ncols)

        # Pad zero images for lists that have differnt length from the max.
        # images_list = list(images_list)  # now it is mutable, unlike tuple.
        # for i, a_list in enumerate(images_list):
        #     diff = self.num_images - len(a_list)
        #     if diff > 0:
        #         for _ in range(diff):
        #             if type(a_list) is list:
        #                 a_list.append(np.zeros_like(a_list[0]))
        #             else:  # numpy array
        #                 a_list = np.concatenate([a_list, [a_list[0]]])
        #     images_list[i] = a_list
        # # As far as labels are concerned, None is typed, if length is passed.

        # axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        # axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        # bnext = Button(axnext, 'Next')
        # bnext.on_clicked(callback.next)
        # bprev = Button(axprev, 'Previous')
        # bprev.on_clicked(callback.prev)

        if sup_titles is None:
            sup_titles = [str(i) for i in np.arange(self.index_max)]

        if labels:
            sub_labels = [[a_label for _ in np.arange(self.index_max)] for a_label in labels]
        elif sub_labels is None:
            sub_labels = [[str(i) for i in np.arange(self.index_max)] for _ in range(self.num_plots)]

        self.image_list = images_list
        self.sub_labels = sub_labels
        self.titles = sup_titles
        self.delay = delay
        self.fname = None
        self.kwargs = kwargs
        self.event = None
        self.cmaps = Cycle(plt.colormaps())
        self.cmaps.set('viridis')
        self.auto_brightness = auto_brightness
        if ax is None:
            self.figpolicy = figpolicy
            self.fig = self.get_fig(figname=figname,
                                    figsize=(14, 9) if figsize is None else figsize, facecolor='white')
            if figsize is None:
                plt.get_current_fig_manager().full_screen_toggle()
                # .window.showMaximized()  # state('zoom')
                # plt.get_current_fig_manager().window.setGeometry(800,70,1000,900)
            if gridspec is not None:
                gs = self.fig.add_gridspec(gridspec[0])
                self.ax = []
                for ags in gs[1:]:
                    self.ax.append(self.fig.add_subplot(gs[ags[0], ags[1]]))
            else:
                self.ax = self.fig.subplots(nrows=nrows, ncols=ncols)
        else:
            self.ax = ax
            try:
                self.fig = ax[0].figure
            except TypeError:  # Not subscriptable, single axis.
                self.fig = ax.figure

        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect("pick_event", self.annotate)
        if tight:
            self.fig.tight_layout()
        if subplots_adjust is not None:
            self.fig.subplots_adjust(**subplots_adjust)

        # if save_type.parser == "internal":
        #     raise TypeError("Requires external data parser")
        if save_kwargs is None:
            save_kwargs = {}
        self.saver = save_type(watch_figs=[self.fig], save_dir=save_dir, save_name=save_name,
                               delay=delay, fps=1000 / delay, **save_kwargs)
        if nrows == 1 and ncols == 1:
            self.ax = [self.ax]  # make a list out of it.
        else:
            self.ax = self.ax.ravel()  # make a 1D  list out of a 2D array.
        for an_ax in self.ax:
            # an_ax.set_xticks([])
            # an_ax.set_yticks([])
            self.toggle_ticks(an_ax, state=False)
        self.transposed_images = [images for images in zip(*images_list)]
        self.transposed_sublabels = [labels for labels in zip(*sub_labels)]

        self.ims = []  # container for images.
        self.pause = pause
        self.index = 0
        self.animate()

    def animate(self):
        for i in range(self.index, self.index_max):
            for j, (an_image, a_label, an_ax) in enumerate(zip(self.transposed_images[i],
                                                               self.transposed_sublabels[i],
                                                               self.ax)):
                # with zipping, the shortest of the three, will stop the loop.
                if i == 0 and self.ims.__len__() < self.num_plots:
                    im = an_ax.imshow(an_image, animated=True, **self.kwargs)
                    self.ims.append(im)
                else:
                    self.ims[j].set_data(an_image)
                if self.auto_brightness:
                    self.ims[j].norm.autoscale(an_image)
                an_ax.set_xlabel(f'{a_label}')
            self.fig.suptitle(self.titles[i], fontsize=8)
            self.saver.add(names=[self.titles[i]])
            if self.pause:
                break
            else:
                self.index = i
        if self.index == self.index_max - 1 and not self.pause:  # arrived at last image and not in manual mode
            self.fname = self.saver.finish()

    def annotate(self, event, axis=None, data=None):
        for ax in self.ax:
            super().annotate(event, axis=ax, data=ax.images[0].get_array())

    @classmethod
    def from_saved_images_path_lists(cls, *image_list, **kwargs):
        images = []
        sub_labels = []
        for alist in image_list:
            image_subcontainer = []
            label_subcontainer = []
            for an_image in alist:
                image_subcontainer.append(plt.imread(an_image))
                label_subcontainer.append(P(an_image).name)
            images.append(image_subcontainer)
            sub_labels.append(label_subcontainer)
        return cls(*images, sub_labels=sub_labels, **kwargs)

    @classmethod
    def from_directories(cls, *directories, extension='png', **kwargs):
        paths = []
        for a_dir in directories:
            paths.append(P(a_dir).search(f"*.{extension}", win_order=True))
        return cls.from_saved_images_path_lists(*paths, **kwargs)

    @classmethod
    def from_saved(cls, *things, **kwargs):
        example_item = things[0]
        if isinstance(example_item, list):
            return cls.from_saved_images_path_lists(*things, **kwargs)
        else:
            return cls.from_directories(*things, **kwargs)

    @staticmethod
    def cm(im, nrows=3, ncols=7, **kwargs):
        """ Useful for looking at one image in multiple cmaps

        :param im:
        :param nrows:
        :param ncols:
        :param kwargs:
        :return:
        """
        styles = plt.colormaps()
        colored = [plt.get_cmap(style)(im) for style in styles]
        splitted = np.array_split(colored, nrows * ncols)
        labels = np.array_split(styles, nrows * ncols)
        _ = ImShow(*splitted, nrows=nrows, ncols=ncols, sub_labels=labels, **kwargs)
        return colored

    @staticmethod
    def test():
        return ImShow(*np.random.randn(12, 101, 100, 100))

    @classmethod
    def complex(cls, data, pause=True, **kwargs):
        return cls(data.real, data.imag, np.angle(data), abs(data), labels=['Real Part', 'Imaginary Part',
                                                                            'Angle in Radians', 'Absolute Value'],
                   pause=pause, **kwargs)

    @staticmethod
    def resize(path, m, n):
        from skimage.transform import resize
        image = plt.imread(path)
        image_resized = resize(image, (m, n), anti_aliasing=True)
        plt.imsave(path, image_resized)


class Artist(FigureManager):
    def __init__(self, *args, ax=None, figname='Graph', title='', label='curve', style='seaborn',
                 create_new_axes=False, figpolicy=FigurePolicy.add_new, figsize=(7, 4), **kwargs):
        super().__init__(figpolicy=figpolicy)
        self.style = style
        # self.kwargs = kwargs
        self.title = title
        self.args = args
        self.line = self.cursor = self.check_b = None
        if ax is None:  # create a figure
            with plt.style.context(style=self.style):
                self.fig = self.get_fig(figname, figsize=figsize)
        else:  # use the passed axis
            self.ax = ax
            self.fig = ax[0].figure

        if len(args):  # if there's something to plot in the init
            if not ax:  # no ax sent but we need to plot, we need an ax, plot will soon call get_axes.
                self.create_new_axes = True  # just for the first time in this init method.
            self.plot(*self.args, label=label, title=title, **kwargs)
        else:  # nothing to be plotted in the init
            if not create_new_axes:  # are we going to ever create new axes?
                self.create_new_axes = True  # if not then let's create one now.
                self.get_axes()
        self.create_new_axes = create_new_axes
        self.visibility_ax = [0.01, 0.05, 0.2, 0.15]
        self.txt = []

    def plot(self, *args, **kwargs):
        self.get_axes()
        self.accessorize(*args, **kwargs)

    def accessorize(self, *args, legends=None, title=None, **kwargs):
        self.line = self.ax[0].plot(*args, **kwargs)
        if legends is not None:
            self.ax[0].legend(legends)
        if title is not None:
            self.ax[0].set_title(title)
        self.ax[0].grid('on')

    def get_axes(self):
        if self.create_new_axes:
            axis = self.fig.subplots()
            self.ax = np.array([axis])
        else:  # use same old axes
            pass

    def suptitle(self, title):
        self.txt = [self.fig.text(0.5, 0.98, title, ha='center', va='center', size=9)]

    def visibility(self):
        from matplotlib.widgets import CheckButtons
        self.fig.subplots_adjust(left=0.3)
        self.visibility_ax[-1] = 0.05 * len(self.ax.lines)
        rax = self.fig.add_axes(self.visibility_ax)
        labels = [str(line.get_label()) for line in self.ax.lines]
        visibility = [line.get_visible() for line in self.ax.lines]
        self.check_b = CheckButtons(rax, labels, visibility)

        def func(label):
            index = labels.index(label)
            self.ax.lines[index].set_visible(not self.ax.lines[index].get_visible())
            self.fig.canvas.draw()

        self.check_b.on_clicked(func)

    @staticmethod
    def styler(plot_gen):
        styles = plt.style.available
        for astyle in styles:
            with plt.style.context(style=astyle):
                plot_gen()
                plt.title(astyle)
                plt.pause(1)
                plt.cla()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
