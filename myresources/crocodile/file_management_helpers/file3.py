from crocodile.core import str2timedelta, Save
# from crocodile.file_management import P, Read, OPLike, PLike
from crocodile.file_management_helpers.file4 import Read, OPLike, PLike, P
from datetime import datetime, timedelta
import time
from typing import Any, Optional, Union, Callable, TypeVar, NoReturn, Protocol, Generic


T = TypeVar('T')
T2 = TypeVar('T2')
class PrintFunc(Protocol):
    def __call__(self, *args: str) -> Union[NoReturn, None]: ...


class Cache(Generic[T]):  # This class helps to accelrate access to latest data coming from expensive function. The class has two flavours, memory-based and disk-based variants."""
    # source_func: Callable[[], T]
    def __init__(self, source_func: Callable[[], T],
                 expire: Union[str, timedelta] = "1m", logger: Optional[PrintFunc] = None, path: OPLike = None,
                 saver: Callable[[T, PLike], Any] = Save.pickle, reader: Callable[[PLike], T] = Read.pickle, name: Optional[str] = None) -> None:
        self.cache: T
        self.source_func = source_func  # function which when called returns a fresh object to be frozen.
        self.path: Optional[P] = P(path) if path else None  # if path is passed, it will function as disk-based flavour.
        self.time_produced = datetime.now()  # if path is None else
        self.save = saver
        self.reader = reader
        self.logger = logger
        self.expire = str2timedelta(expire) if isinstance(expire, str) else expire
        self.name = name if isinstance(name, str) else str(self.source_func)
        self.last_call_is_fresh = False
    @property
    def age(self):
        """Throws AttributeError if called before cache is populated and path doesn't exists"""
        if self.path is None:  # memory-based cache.
            return datetime.now() - self.time_produced
        return datetime.now() - datetime.fromtimestamp(self.path.stat().st_mtime)
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.path = P.home() / self.path if self.path is not None else self.path
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["path"] = self.path.rel2home() if self.path is not None else state["path"]
        return state  # With this implementation, instances can be pickled and loaded up in different machine and still works.
    def __call__(self, fresh: bool = False) -> T:
        self.last_call_is_fresh = False
        if fresh or not hasattr(self, "cache"):  # populate cache for the first time
            if not fresh and self.path is not None and self.path.exists():
                age = datetime.now() - datetime.fromtimestamp(self.path.stat().st_mtime)
                msg1 = f"""
📦 ════════════════════ CACHE OPERATION ════════════════════
🔄 {self.name} cache: Reading cached values from `{self.path}`
⏱️  Lag = {age}
════════════════════════════════════════════════════════════
"""
                try:
                    self.cache = self.reader(self.path)
                except Exception as ex:
                    if self.logger:
                        msg2 = f"""
❌ ════════════════════ CACHE ERROR ════════════════════
⚠️  {self.name} cache: Cache file is corrupted
🔍 Error: {ex}
════════════════════════════════════════════════════════
"""
                        self.logger(msg1 + msg2)
                    self.cache = self.source_func()
                    self.last_call_is_fresh = True
                    self.time_produced = datetime.now()
                    if self.path is not None: self.save(self.cache, self.path)
                    return self.cache
                return self(fresh=False)  # may be the cache is old ==> check that by passing it through the logic again.
            else:
                if self.logger:
                    # Previous cache never existed or there was an explicit fresh order.
                    why = "There was an explicit fresh order." if fresh else "Previous cache never existed or is corrupted."
                    self.logger(f"""
🆕 ════════════════════ NEW CACHE ════════════════════
🔄 {self.name} cache: Populating fresh cache from source func
ℹ️  Reason: {why}
════════════════════════════════════════════════════════
""")
                self.cache = self.source_func()  # fresh data.
                self.last_call_is_fresh = True
                self.time_produced = datetime.now()
                if self.path is not None: self.save(self.cache, self.path)
        else:  # cache exists
            try: age = self.age
            except AttributeError:  # path doesn't exist (may be deleted) ==> need to repopulate cache form source_func.
                return self(fresh=True)
            if age > self.expire:
                if self.logger:
                    self.logger(f"""
🔄 ════════════════════ CACHE UPDATE ════════════════════
⚠️  {self.name} cache: Updating cache from source func
⏱️  Age = {age} > {self.expire}
════════════════════════════════════════════════════════
""")
                self.cache = self.source_func()
                self.last_call_is_fresh = True
                self.time_produced = datetime.now()
                if self.path is not None: self.save(self.cache, self.path)
            else:
                if self.logger: 
                    self.logger(f"""
✅ ════════════════════ USING CACHE ════════════════════
📦 {self.name} cache: Using cached values
⏱️  Lag = {age}
════════════════════════════════════════════════════════
""")
        return self.cache
    @staticmethod
    def as_decorator(expire: Union[str, timedelta] = "1m", logger: Optional[PrintFunc] = None, path: OPLike = None,
                     saver: Callable[[T2, PLike], Any] = Save.pickle,
                     reader: Callable[[PLike], T2] = Read.pickle,
                     name: Optional[str] = None):  # -> Callable[..., 'Cache[T2]']:
        def decorator(source_func: Callable[[], T2]) -> Cache['T2']:
            res = Cache(source_func=source_func, expire=expire, logger=logger, path=path, name=name, reader=reader, saver=saver)
            return res
        return decorator
    def from_cloud(self, cloud: str, rel2home: bool = True, root: Optional[str] = None):
        assert self.path is not None
        exists = self.path.exists()
        exists_but_old = exists and ((datetime.now() - datetime.fromtimestamp(self.path.stat().st_mtime)) > self.expire)
        if not exists or exists_but_old:
            returned_path = self.path.from_cloud(cloud=cloud, rel2home=rel2home, root=root)
            if returned_path is None and not exists:
                raise FileNotFoundError(f"❌ Failed to get @ {self.path}. Build the cache first with signed API.")
            elif returned_path is None and exists and self.logger is not None:
                self.logger(f"""
⚠️ ════════════════════ CLOUD FETCH WARNING ════════════════════
🔄 Failed to get fresh data from cloud 
📦 Using old cache @ {self.path}
════════════════════════════════════════════════════════════════
""")
        else:
            pass  # maybe we don't need to fetch it from cloud, if its too hot
        return self.reader(self.path)


class CacheV2(Generic[T]):
    def __init__(self, source_func: Callable[[], T],
                 expire: int = 60000, logger: Optional[PrintFunc] = None, path: OPLike = None,
                 saver: Callable[[T, PLike], Any] = Save.pickle, reader: Callable[[PLike], T] = Read.pickle, name: Optional[str] = None) -> None:
        self.cache: Optional[T] = None
        self.source_func = source_func
        self.path: Optional[P] = P(path) if path else None
        self.time_produced = time.time_ns() // 1_000_000
        self.save = saver
        self.reader = reader
        self.logger = logger
        self.expire = expire  # in milliseconds
        self.name = name if isinstance(name, str) else str(self.source_func)
    @property
    def age(self):
        if self.path is None:
            return time.time_ns() // 1_000_000 - self.time_produced
        return time.time_ns() // 1_000_000 - int(self.path.stat().st_mtime * 1000)
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.path = P.home() / self.path if self.path is not None else self.path
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["path"] = self.path.relative_to(P.home()) if self.path is not None else state["path"]
        return state
    def __call__(self, fresh: bool = False) -> T:
        if fresh or self.cache is None:
            if not fresh and self.path is not None and self.path.exists():
                age = time.time_ns() // 1_000_000 - int(self.path.stat().st_mtime * 1000)
                msg1 = f"""
📦 ════════════════════ CACHE V2 OPERATION ════════════════════
🔄 {self.name} cache: Reading cached values from `{self.path}`
⏱️  Lag = {age} ms
════════════════════════════════════════════════════════════════
"""
                try:
                    self.cache = self.reader(self.path)
                except Exception as ex:
                    if self.logger:
                        msg2 = f"""
❌ ════════════════════ CACHE V2 ERROR ════════════════════
⚠️  {self.name} cache: Cache file is corrupted
🔍 Error: {ex}
════════════════════════════════════════════════════════════
"""
                        self.logger(msg1 + msg2)
                    self.cache = self.source_func()
                    self.save(self.cache, self.path)
                    return self.cache
                return self(fresh=False)
            else:
                if self.logger:
                    self.logger(f"""
🆕 ════════════════════ NEW CACHE V2 ════════════════════
🔄 {self.name} cache: Populating fresh cache from source func
ℹ️  Reason: Previous cache never existed or there was an explicit fresh order
════════════════════════════════════════════════════════════
""")
                self.cache = self.source_func()
                if self.path is None:
                    self.time_produced = time.time_ns() // 1_000_000
                else:
                    self.save(self.cache, self.path)
        else:
            try:
                age = self.age
            except AttributeError:
                self.cache = None
                return self(fresh=fresh)
            if age > self.expire:
                if self.logger:
                    self.logger(f"""
🔄 ════════════════════ CACHE V2 UPDATE ════════════════════
⚠️  {self.name} cache: Updating cache from source func
⏱️  Age = {age} ms > {self.expire} ms
════════════════════════════════════════════════════════════
""")
                self.cache = self.source_func()
                if self.path is None:
                    self.time_produced = time.time_ns() // 1_000_000
                else:
                    self.save(self.cache, self.path)
            else:
                if self.logger:
                    self.logger(f"""
✅ ════════════════════ USING CACHE V2 ════════════════════
📦 {self.name} cache: Using cached values
⏱️  Lag = {age} ms
════════════════════════════════════════════════════════════
""")
        return self.cache
    @staticmethod
    def as_decorator(expire: int = 60000, logger: Optional[PrintFunc] = None, path: OPLike = None,
                     name: Optional[str] = None):
        def decorator(source_func: Callable[[], T2]) -> CacheV2['T2']:
            res = CacheV2(source_func=source_func, expire=expire, logger=logger, path=path, name=name)
            return res
        return decorator

