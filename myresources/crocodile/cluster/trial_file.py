
import time
import crocodile.toolbox as tb


def expensive_function() -> tb.P:
    print("Hello, I am an expensive function, and I just started running ...")
    a = 1 + 1
    time.sleep(2)
    print("I'm done, I crunched a lot of numbers.")
    return tb.S(a=a).save()


def expensive_function_parallel(start_idx: int, end_idx: int, num_threads: int, ram_gb: int) -> tb.P:
    print(f"Hello, I am an expensive function, and I just started running on {num_threads} threads, with {ram_gb} GB of RAM ...")
    time.sleep(2)
    print("I'm done, I crunched numbers from {} to {}.".format(start_idx, end_idx))
    return tb.S(sum=sum(range(start_idx, end_idx))).save()


if __name__ == '__main__':
    pass
