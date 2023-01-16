
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


def expensive_function_parallel(idx_start: int, idx_end: int, idx_max: int, num_instances: int) -> tb.P:
    print(f"Splitting the work among {num_instances} instances ...")
    _ = tb.L(range(idx_start, idx_end, 1)).split(to=num_instances).apply(lambda sub_list: inner_func(sub_list[0], sub_list[-1], idx_max), jobs=num_instances)
    path = tb.P.tmpfile(suffix=".Struct.pkl")
    tb.S(sum=sum(range(idx_start, idx_end))).save(path=path)
    return path


def inner_func(idx_start, idx_end, idx_max):
    print(f"Hello, I am an expensive function, and I just started running ...")
    execution_time_in_seconds = 60 * 3
    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done
    print("I'm done, I crunched numbers from {} to {}.".format(idx_start, idx_end))


if __name__ == '__main__':
    pass
