
import time
import crocodile.toolbox as tb


def expensive_function() -> tb.P:
    length = 20
    print(f"Hello, I am an expensive function, and I just started running ...\n It will take me {length} seconds to finish")
    a = 1 + 1
    time.sleep(length)
    print("I'm done, I crunched a lot of numbers. Next I'll save my results to a file and passing its directory back to the main process on the machine running me.")
    return tb.S(a=a).save(path=tb.P.tmpdir().joinpath("result.Struct.pkl")).parent


def expensive_function_parallel(start_idx: int, end_idx: int, num_threads: int, ram_gb: int) -> tb.P:
    print(f"Hello, I am an expensive function, and I just started running on {num_threads} threads, with {ram_gb} GB of RAM ...")
    time.sleep(2)
    print("I'm done, I crunched numbers from {} to {}.".format(start_idx, end_idx))
    return tb.S(sum=sum(range(start_idx, end_idx))).save(path=tb.P.tmpfile(suffix=".Struct.pkl"))


if __name__ == '__main__':
    pass
