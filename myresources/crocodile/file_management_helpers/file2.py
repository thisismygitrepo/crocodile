from pathlib import Path
import os
from typing import Any, Optional, TypeAlias, Literal


FILE_MODE: TypeAlias = Literal['r', 'w', 'x', 'a']
SHUTIL_FORMATS: TypeAlias = Literal["zip", "tar", "gztar", "bztar", "xztar"]


class Compression:
    @staticmethod
    def compress_folder(root_dir: str, op_path: str, base_dir: str, fmt: SHUTIL_FORMATS = 'zip', verbose: bool = False, **kwargs: Any) -> str:  # shutil works with folders nicely (recursion is done interally) # directory to be archived: root_dir\base_dir, unless base_dir is passed as absolute path. # when archive opened; base_dir will be found."""
        base_name = op_path[:-4] if op_path.endswith(".zip") else op_path  # .zip is added automatically by library, hence we'd like to avoid repeating it if user sent it.
        import shutil
        return shutil.make_archive(base_name=base_name, format=fmt, root_dir=root_dir, base_dir=base_dir, verbose=verbose, **kwargs)  # returned path possible have added extension.
    @staticmethod
    def zip_file(ip_path: str, op_path: str, arcname: Optional[str]= None, password: Optional[bytes] = None, mode: FILE_MODE = "w", **kwargs: Any):
        """arcname determines the directory of the file being archived inside the archive. Defaults to same as original directory except for drive.
        When changed, it should still include the file path in its end. If arcname = filename without any path, then, it will be in the root of the archive."""
        import zipfile
        with zipfile.ZipFile(op_path, mode=mode) as jungle_zip:
            if password is not None: jungle_zip.setpassword(pwd=password)
            jungle_zip.write(filename=str(ip_path), arcname=str(arcname) if arcname is not None else None, compress_type=zipfile.ZIP_DEFLATED, **kwargs)
        return Path(op_path)
    @staticmethod
    def unzip(ip_path: str, op_path: str, fname: Optional[str]= None, password: Optional[bytes] = None, memory: bool = False, **kwargs: Any) -> Path | dict[str, bytes] | bytes:
        import zipfile
        with zipfile.ZipFile(str(ip_path), 'r') as zipObj:
            if memory:
                return {name: zipObj.read(name) for name in zipObj.namelist()} if fname is None else zipObj.read(fname)
            if fname is None:
                zipObj.extractall(op_path, pwd=password, **kwargs)
                return Path(op_path)
            else:
                zipObj.extract(member=str(fname), path=str(op_path), pwd=password)
                return Path(op_path) / fname
    @staticmethod
    def gz(file: str, op_path: str):  # see this on what to use: https://stackoverflow.com/questions/10540935/what-is-the-difference-between-tar-and-zip
        import shutil
        import gzip
        with open(file, 'rb') as f_in:
            with gzip.open(op_path, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        return Path(op_path)
    @staticmethod
    def ungz(path: str, op_path: str):
        import gzip
        import shutil
        with gzip.open(path, 'r') as f_in, open(op_path, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        return Path(op_path)
    @staticmethod
    def unbz(path: str, op_path: str):
        import bz2
        import shutil
        with bz2.BZ2File(path, 'r') as fr, open(str(op_path), 'wb') as fw: shutil.copyfileobj(fr, fw)
        return Path(op_path)
    @staticmethod
    def xz(path: str, op_path: str):
        import lzma
        with lzma.open(op_path, "w") as f: f.write(Path(path).read_bytes())
    @staticmethod
    def unxz(ip_path: str, op_path: str):
        import lzma
        with lzma.open(ip_path) as file: Path(op_path).write_bytes(file.read())
    @staticmethod
    def tar(path: str, op_path: str):
        import tarfile
        with tarfile.open(op_path, "w:gz") as tar_: tar_.add(str(path), arcname=os.path.basename(path))
        return Path(op_path)
    @staticmethod
    def untar(path: str, op_path: str, fname: Optional[str]= None, mode: str = 'r', **kwargs: Any):
        import tarfile
        with tarfile.open(str(path), mode) as file:
            if fname is None: file.extractall(path=op_path, **kwargs)  # extract all files in the archive
            else: file.extract(fname, **kwargs)
        return Path(op_path)

