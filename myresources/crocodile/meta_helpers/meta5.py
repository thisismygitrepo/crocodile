from crocodile.core import install_n_import, Struct
from crocodile.file_management import P, PLike
import time
from typing import Any, Optional, Callable, Literal, TypeVar, ParamSpec

T = TypeVar('T')
PS = ParamSpec('PS')


def generate_readme(path: PLike, obj: Any = None, desc: str = '', save_source_code: bool = True, verbose: bool = True):  # Generates a readme file to contextualize any binary files by mentioning module, class, method or function used to generate the data"""
    import inspect
    text: str = "# Description\n" + desc + (separator := "\n" + "-" * 50 + "\n\n")
    obj_path = P(inspect.getfile(obj)) if obj is not None else None
    path = P(path)
    if obj_path is not None:
        text += f"# Source code file generated me was located here: \n`{obj_path.collapseuser().as_posix()}`\n" + separator
        try:
            repo = install_n_import(library="git", package="gitpython").Repo(obj_path.parent, search_parent_directories=True)
            text += f"# Last Commit\n{repo.git.execute(['git', 'log', '-1'])}{separator}# Remote Repo\n{repo.git.execute(['git', 'remote', '-v'])}{separator}"
            try: tmppp = obj_path.relative_to(repo.working_dir).as_posix()
            except Exception: tmppp = ""  # type: ignore
            text += f"# link to files: \n{repo.remote().url.replace('.git', '')}/tree/{repo.active_branch.commit.hexsha}/{tmppp}{separator}"
        except Exception as ex: text += f"Could not read git repository @ `{obj_path.parent}`\n{ex}.\n"
    if obj is not None:
        try: source_code = inspect.getsource(obj)
        except OSError: source_code = f"Could not read source code from `{obj_path}`."
        text += ("\n\n# Code to reproduce results\n\n```python\n" + source_code + "\n```" + separator)
    readmepath = (path / "README.md" if path.is_dir() else (path.with_name(path.trunk + "_README.md") if path.is_file() else path)).write_text(text, encoding="utf-8")
    if verbose: print(f"💽 SAVED {readmepath.name} @ {readmepath.absolute().as_uri()}")
    if save_source_code:
        if hasattr(obj, "__code__"):
            save_path = obj.__code__.co_filename
        else:
            module_maybe = inspect.getmodule(obj)
            if module_maybe is not None: save_path = module_maybe.__file__
            else: save_path = None
        if save_path is None:
            print(f"Could not find source code for {obj}.")
            return readmepath
        P(save_path).zip(path=readmepath.with_name(P(readmepath).trunk + "_source_code.zip"), verbose=False)
        print("💽 SAVED source code @ " + readmepath.with_name("source_code.zip").absolute().as_uri())
        return readmepath


class RepeatUntilNoException:
    """
    Repeat function calling if it raised an exception and/or exceeded the timeout, for a maximum of `retry` times.
    * Alternative: `https://github.com/jd/tenacity`
    """
    def __init__(self, retry: int, sleep: float, timeout: Optional[float] = None, scaling: Literal["linear", "exponential"] = "exponential"):
        self.retry = retry
        self.sleep = sleep
        self.timeout = timeout
        self.scaling: Literal["linear", "exponential"] = scaling
    def __call__(self, func: Callable[PS, T]) -> Callable[PS, T]:
        from functools import wraps
        if self.timeout is not None:
            func = install_n_import("wrapt_timeout_decorator").timeout(self.timeout)(func)
        @wraps(wrapped=func)
        def wrapper(*args: PS.args, **kwargs: PS.kwargs):
            t0 = time.time()
            for idx in range(self.retry):
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    match self.scaling:
                        case "linear":
                            sleep_time = self.sleep * (idx + 1)
                        case "exponential":
                            sleep_time = self.sleep * (idx + 1)**2
                    print(f"""💥 [RETRY] Function {func.__name__} call failed with error:
{ex}
Retry count: {idx}/{self.retry}. Sleeping for {sleep_time} seconds.
Total elapsed time: {time.time() - t0:0.1f} seconds.""")
                    print(f"""💥 Robust call of `{func}` failed with ```{ex}```.\nretrying {idx}/{self.retry} more times after sleeping for {sleep_time} seconds.\nTotal wait time so far {time.time() - t0: 0.1f} seconds.""")
                    time.sleep(sleep_time)
            raise RuntimeError(f"💥 Robust call failed after {self.retry} retries and total wait time of {time.time() - t0: 0.1f} seconds.\n{func=}\n{args=}\n{kwargs=}")
        return wrapper


def show_globals(scope: dict[str, Any], **kwargs: Any):
    # see print_dir
    return Struct(scope).filter(lambda k, v: "__" not in k and not k.startswith("_") and k not in {"In", "Out", "get_ipython", "quit", "exit", "sys"}).print(**kwargs)
