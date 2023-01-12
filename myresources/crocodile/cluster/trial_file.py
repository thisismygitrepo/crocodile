
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


def expensive_function_parallel(start_idx: int, end_idx: int, num_threads: int, ram_gb: int) -> tb.P:
    print(f"Hello, I am an expensive function, and I just started running on {num_threads} threads, with {ram_gb} GB of RAM ...")
    time.sleep(2)
    print("I'm done, I crunched numbers from {} to {}.".format(start_idx, end_idx))
    return tb.S(sum=sum(range(start_idx, end_idx))).save(path=tb.P.tmpfile(suffix=".Struct.pkl"))


if __name__ == '__main__':
    pass
