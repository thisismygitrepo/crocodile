
import time
import crocodile.toolbox as tb
from rich.progress import track


def expensive_function() -> tb.P:
    execution_time_in_seconds = 60 * 3
    print(f"Hello, I am an expensive function, and I just started running ...\n It will take me {execution_time_in_seconds} seconds to finish")
    a = 1 + 1

    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done

    print("I'm done, I crunched a lot of numbers. Next I'll save my results to a file and passing its directory back to the main process on the machine running me.")
    path = tb.P.tmpdir().joinpath("result.Struct.pkl")
    tb.S(a=a).save(path=path)
    return path.parent


def parallelize(idx_start: int, idx_end: int, idx_max: int, num_instances: int) -> tb.P:
    print(f"This script will execute ({(idx_max - idx_start) / idx_max * 100:.2f}%) of the work on this machine.")
    print(f"Splitting the work ({idx_start=}, {idx_end=}) among {num_instances} instances ...")
    kwargs_split = tb.L(range(idx_start, idx_end, 1)).split(to=num_instances).apply(lambda sub_list: dict(idx_start=sub_list[0], idx_end=sub_list[-1], idx_max=idx_max))
    for idx, x in enumerate(kwargs_split):
        tb.S(x).print(as_config=True, title=f"Instance {idx}")

    res = kwargs_split.apply(lambda kwargs: expensive_function_single_thread(**kwargs), jobs=num_instances)
    return tb.P(res[0]).parent


def expensive_function_single_thread(idx_start, idx_end, idx_max):
    print(f"Hello, I am an expensive function, and I just started running ...")
    execution_time_in_seconds = 60 * 3
    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done
    print("I'm done, I crunched numbers from {} to {}.".format(idx_start, idx_end))
    path = tb.P.tmp().joinpath(f"tmp_dirs/expensive_function_single_thread/trial_func_result_{idx_start}_{idx_end}.Struct.pkl")
    path.delete(sure=True)
    tb.S(a=1).save(path=path)
    return path


if __name__ == '__main__':
    pass
