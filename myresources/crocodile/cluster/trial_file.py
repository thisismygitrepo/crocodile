
import time
import crocodile.toolbox as tb
from rich.progress import track
from crocodile.cluster.remote_machine import WorkloadParams


def expensive_function(func_kwargs=None) -> tb.P:
    execution_time_in_seconds = 60 * 1.2
    print(f"Hello, I am an expensive function, and I just started running ...\n It will take me {execution_time_in_seconds} seconds to finish")
    a = 1 + 1
    print(f"Oh, I aslo recieved those arguments: {func_kwargs}")

    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done

    print("I'm done, I crunched a lot of numbers. Next I'll save my results to a file and passing its directory back to the main process on the machine running me.")
    path = tb.P.tmpdir().joinpath("result.Struct.pkl")
    tb.S(a=a).save(path=path)
    return path.parent  # extra job details files will be added to this res_folder


# todo: consider making it a method of a class and user will subclass it and is forced to follow its interface


def expensive_function_single_thread(workload_params: WorkloadParams,
                                     sim_dict=None) -> tb.P:
    print(f"Hello, I am one thread of an expensive function, and I just started running ...")
    print(f"Oh, I recieved this parameter: {sim_dict=}")
    execution_time_in_seconds = 60 * 1
    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done
    print("I'm done, I crunched numbers from {} to {}.".format(workload_params.idx_start, workload_params.idx_end))
    _ = workload_params.idx_max
    save_dir = tb.P.tmp().joinpath(f"tmp_dirs/expensive_function_single_thread").joinpath(workload_params.save_suffix, f"thread_{workload_params.idx_start}_{workload_params.idx_end}").create()
    tb.S(a=1).save(path=save_dir.joinpath(f"trial_func_result.Struct.pkl"))
    return save_dir


if __name__ == '__main__':
    pass
