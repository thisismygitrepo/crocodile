
import time
import crocodile.toolbox as tb
from rich.progress import track


def expensive_function() -> tb.P:
    execution_time_in_seconds = 60 * 1.2
    print(f"Hello, I am an expensive function, and I just started running ...\n It will take me {execution_time_in_seconds} seconds to finish")
    a = 1 + 1

    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done

    print("I'm done, I crunched a lot of numbers. Next I'll save my results to a file and passing its directory back to the main process on the machine running me.")
    path = tb.P.tmpdir().joinpath("result.Struct.pkl")
    tb.S(a=a).save(path=path)
    return path.parent  # extra job details files will be added to this res_folder


# parallelizeBegins


def parallelize(idx_start: int, idx_end: int, idx_max: int, num_instances: int) -> tb.P:

    print(f"This script will execute ({(idx_end - idx_start) / idx_max * 100:.2f}%) of the work on this machine.")

    print(f"Splitting the work ({idx_start=}, {idx_end=}) equally among {num_instances} instances via `parallelize` of cluster.trial_file ...")

    kwargs_split = tb.L(range(idx_start, idx_end, 1)).split(to=num_instances).apply(lambda sub_list: dict(idx_start=sub_list[0], idx_end=sub_list[-1], idx_max=idx_max))
    for idx, x in enumerate(kwargs_split):
        tb.S(x).print(as_config=True, title=f"Instance {idx}")
    print("\n" * 2)

    save_dir_suffix = f"machine_{idx_start}_{idx_end}"

    res = kwargs_split.apply(lambda kwargs: expensive_function_single_thread(**kwargs, save_dir_suffix=save_dir_suffix), jobs=num_instances)
    return tb.P(res[0]).parent


# parallelizeEnds


# todo: consider making it a method of a class and user will subclass it and is forced to follow its interface


def expensive_function_single_thread(idx_start: int, idx_end: int, idx_max: int, save_dir_suffix: tb.P or str) -> tb.P:
    print(f"Hello, I am one thread of an expensive function, and I just started running ...")
    execution_time_in_seconds = 60 * 1
    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done
    print("I'm done, I crunched numbers from {} to {}.".format(idx_start, idx_end))
    _ = idx_max
    save_dir = tb.P.tmp().joinpath(f"tmp_dirs/expensive_function_single_thread").joinpath(save_dir_suffix, f"thread_{idx_start}_{idx_end}").create()
    tb.S(a=1).save(path=save_dir.joinpath(f"trial_func_result.Struct.pkl"))
    return save_dir


if __name__ == '__main__':
    pass
