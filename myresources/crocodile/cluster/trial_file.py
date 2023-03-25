
import time
import crocodile.toolbox as tb
from rich.progress import track
from crocodile.cluster.remote_machine import WorkloadParams


# todo: consider making it a method of a class and user will subclass it and is forced to follow its interface


def expensive_function(workload_params: WorkloadParams, sim_dict=None) -> tb.P:
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
