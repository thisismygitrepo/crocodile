from crocodile.core import randstr, install_n_import, List
from crocodile.file_management import P, OPLike, PLike
import logging
import sys
from typing import Union, Any, Optional, TextIO, IO, Literal, TypeVar, ParamSpec
from dataclasses import dataclass


_ = IO, TextIO
T = TypeVar('T')
PS = ParamSpec('PS')


@dataclass
class Scout:
    source_full: P
    source_rel2home: P
    exists: bool
    is_dir: bool
    files: Optional[List[P]]


def scout(source: PLike, z: bool = False, r: bool = False) -> Scout:
    source_full = P(source).expanduser().absolute()
    source_rel2home = source_full.collapseuser()
    exists = source_full.exists()
    is_dir = source_full.is_dir() if exists else False
    if z and exists:
        try: source_full = source_full.zip()
        except Exception as ex:
            raise Exception(f"Could not zip {source_full} due to {ex}") from ex  # type: ignore # pylint: disable=W0719
        source_rel2home = source_full.zip()
    files = source_full.search(folders=False, r=True).apply(lambda x: x.collapseuser()) if r and exists and is_dir else None
    return Scout(source_full=source_full, source_rel2home=source_rel2home, exists=exists, is_dir=is_dir, files=files)


class Log(logging.Logger):  #
    def __init__(self, dialect: Literal["colorlog", "logging", "coloredlogs"] = "colorlog",
                 name: Optional[str] = None, file: bool = False, file_path: OPLike = None, stream: bool = True, fmt: Optional[str] = None, sep: str = " | ",
                 s_level: int = logging.DEBUG, f_level: int = logging.DEBUG, l_level: int = logging.DEBUG, verbose: bool = False,
                 log_colors: Optional[dict[str, str]] = None):
        if name is None:
            name = randstr(noun=True)
            print("""
╔═════════════════════════ 📢 LOGGER WARNING 📢 ═════════════════════════╗
║ 🔔 Logger name not provided.                                            ║
║ ℹ️  Please provide a descriptive name for proper identification!        ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
        super().__init__(name, level=l_level)  # logs everything, finer level of control is given to its handlers
        print(f"🔧 Logger `{name}` from `{dialect}` initialized with level {l_level} 🔧")
        self.file_path = file_path  # proper update to this value by self.add_filehandler()
        if dialect == "colorlog":
            import colorlog
            module: Any = colorlog
            processed_fmt: Any = colorlog.ColoredFormatter(fmt or (r"%(log_color)s" + Log.get_format(sep)), datefmt="%d %H:%M:%S", log_colors=log_colors or {'DEBUG': 'bold_cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'fg_bold_red,bg_black', 'CRITICAL': 'thin_red', })  # see here for format: https://pypi.org/project/colorlog/
        else:
            module = logging
            processed_fmt = logging.Formatter(fmt or Log.get_format(sep))
        if file or file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.add_filehandler(file_path=file_path, fmt=processed_fmt, f_level=f_level)
        if stream: self.add_streamhandler(s_level, fmt=processed_fmt, module=module)
        self.specs = dict(dialect=dialect, name=self.name, file=file, file_path=self.file_path, stream=bool(self.get_shandler()), fmt=fmt, sep=sep, s_level=s_level, f_level=f_level, l_level=l_level, verbose=verbose, log_colors=log_colors)  # # this way of creating relative path makes transferrable across machines.
    def get_shandler(self): return List(handler for handler in self.handlers if "StreamHandler" in str(handler))
    def get_fhandler(self): return List(handler for handler in self.handlers if "FileHandler" in str(handler))
    def set_level(self, level: int, which: Literal["logger", "stream", "file", "all"] = "logger"):
        if which in {"logger", "all"}: self.setLevel(level)
        if which in {"stream", "all"}: self.get_shandler().setLevel(level)
        if which in {"file", "all"}: self.get_fhandler().setLevel(level)
    def __reduce_ex__(self, protocol: Any): _ = protocol; return self.__class__, tuple(self.specs.values())  # reduce_ex is enchanced reduce. It is lower than getstate and setstate. It uses init method to create an instance.
    def __repr__(self): return "".join([f"Logger {self.name} (level {self.level}) with handlers: \n"] + [repr(h) + "\n" for h in self.handlers])
    @staticmethod
    def get_format(sep: str = ' | ', datefmt: str = "%d %H:%M:%S"):
        _ = datefmt  # TODO: add datefmt to the format string
        return f"%(asctime)s{sep}%(name)s{sep}%(module)s{sep}%(funcName)s{sep}%(levelname)s(%(levelno)s){sep}%(message)s{sep}"  # Reference: https://docs.python.org/3/library/logging.html#logrecord-attributes logging.BASIC_FORMAT
    def manual_degug(self, path: PLike):
        _ = self
        sys.stdout = open(path, 'w', encoding="utf-8")
        sys.stdout.close()
        print(f"""
✅ ═════════════════ DEBUG OPERATION COMPLETED ═════════════════
📝 Output debug file located at: {path}
══════════════════════════════════════════════════════════════
""")
    @staticmethod
    def get_coloredlogs(name: Optional[str] = None, file: bool = False, file_path: OPLike = None, stream: bool = True, fmt: Optional[str] = None, sep: str = " | ", s_level: int = logging.DEBUG, f_level: int = logging.DEBUG, l_level: int = logging.DEBUG, verbose: bool = False):
        level_styles = {'spam': {'color': 'green', 'faint': True}, 'debug': {'color': 'white'}, 'verbose': {'color': 'blue'}, 'info': {'color': "green"}, 'notice': {'color': 'magenta'}, 'warning': {'color': 'yellow'}, 'success': {'color': 'green', 'bold': True},
                        'error': {'color': 'red', "faint": True, "underline": True}, 'critical': {'color': 'red', 'bold': True, "inverse": False}}  # https://coloredlogs.readthedocs.io/en/latest/api.html#available-text-styles-and-colors
        field_styles = {'asctime': {'color': 'green'}, 'hostname': {'color': 'magenta'}, 'levelname': {'color': 'black', 'bold': True}, 'path': {'color': 'blue'}, 'programname': {'color': 'cyan'}, 'username': {'color': 'yellow'}}
        if verbose:
            logger = install_n_import("verboselogs").VerboseLogger(name=name)
            logger.setLevel(l_level)  # https://github.com/xolox/python-verboselogs # verboselogs.install()  # hooks into logging module.
        else: logger = Log(name=name, dialect="logging", l_level=l_level, file=file, f_level=f_level, file_path=file_path, fmt=fmt or Log.get_format(sep), stream=stream, s_level=s_level)  # new step, not tested:
        install_n_import("coloredlogs").install(logger=logger, name="lol_different_name", level=logging.NOTSET, level_styles=level_styles, field_styles=field_styles, fmt=fmt or Log.get_format(sep), isatty=True, milliseconds=True)
        return logger
    def add_streamhandler(self, s_level: int = logging.DEBUG, fmt: Optional[Any] = None, module: Any = logging, name: str = "myStreamHandler"):
        shandler = module.StreamHandler()
        shandler.setLevel(level=s_level)
        shandler.setFormatter(fmt=fmt)
        shandler.set_name(name)
        self.addHandler(shandler)
        print(f"""
✅ ═════════════════ STREAM HANDLER CREATED ═════════════════
📊 Logger: {self.name}
🔢 Handler Level: {s_level}
═══════════════════════════════════════════════════════════
""")
    def add_filehandler(self, file_path: OPLike = None, fmt: Optional[Any] = None, f_level: int = logging.DEBUG, mode: str = "a", name: str = "myFileHandler"):
        filename = P.tmpfile(name=self.name, suffix=".log", folder="tmp_loggers") if file_path is None else P(file_path).expanduser()
        fhandler = logging.FileHandler(filename=filename, mode=mode)
        fhandler.setFormatter(fmt=fmt)
        fhandler.setLevel(level=f_level)
        fhandler.set_name(name)
        self.addHandler(fhandler)
        self.file_path = filename.collapseuser(strict=False)
        print(f"""
📁 ═════════════════ FILE HANDLER CREATED ═════════════════
📊 Logger: {self.name}
🔢 Level: {f_level} 
📋 Location: {P(filename).clickable()}
═══════════════════════════════════════════════════════════
""")
    def test(self):
        List([self.debug, self.info, self.warning, self.error, self.critical]).apply(lambda func: func(f"this is a {func.__name__} message"))
        for level in range(0, 60, 5): self.log(msg=f"This is a message of level {level}", level=level)
    def get_history(self, lines: int = 200, to_html: bool = False):
        assert isinstance(self.file_path, P), f"❌ Logger `{self.name}` does not have a file handler. Thus, no history is available."
        logs = "\n".join(self.file_path.expanduser().absolute().read_text().split("\n")[-lines:])
        return install_n_import("ansi2html").Ansi2HTMLConverter().convert(logs) if to_html else logs


@dataclass
class STD:
    stdin: str
    stdout: str
    stderr: str
    returncode: int


