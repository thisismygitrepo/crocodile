
import time
import crocodile.toolbox as tb
from rich.progress import track
from crocodile.cluster.remote_machine import ThreadParams
from crocodile.cluster.distribute import Cluster, RemoteMachineConfig, ThreadLoadCalculator, MachineLoadCalculator


def expensive_function(thread_params: ThreadParams, sim_dict=None) -> tb.P:
    print(f"Hello, I am one thread of an expensive function, and I just started running ...")
    print(f"Oh, I recieved this parameter: {sim_dict=}")
    execution_time_in_seconds = 60 * 1
    steps = 100
    for _ in track(range(steps), description="Progress bar ..."):
        time.sleep(execution_time_in_seconds/steps)  # Simulate work being done
    print("I'm done, I crunched numbers from {} to {}.".format(thread_params.idx_start, thread_params.idx_end))
    _ = thread_params.idx_max
    save_dir = tb.P.tmp().joinpath(f"tmp_dirs/expensive_function_single_thread").joinpath(thread_params.save_suffix, f"thread_{thread_params.idx_start}_{thread_params.idx_end}").create()
    tb.S(a=1).save(path=save_dir.joinpath(f"trial_func_result.Struct.pkl"))
    return save_dir


class ExpensiveComputation:
    def __int__(self):
        pass

    def main_single_thread(self, single_thread_params):
        raise NotImplementedError

    def main(self, params: list):
        tb.L(params).apply(lambda single_thread_params: self.main_single_thread(single_thread_params), jobs=len(params))

    def run(self, ms):
        ic = ThreadLoadCalculator(multiplier=7, bottleneck_name="cpu", reference_machine="this_machine")
        mlc = MachineLoadCalculator(num_machines=len(ms), load_criterion="cpu")
        rm_config = RemoteMachineConfig(install_repo=False, copy_repo=True, update_essential_repos=True,
                                        ipython=False, interactive=False, pdb=False, wrap_in_try_except=False,
                                        lock_resources=True, kill_on_completion=False, parallelize=True
                                        )
        c = Cluster(func=self.main,
                    func_kwargs=dict(sim_config=1),
                    ssh_params=ms,
                    machine_load_calc=mlc,
                    thread_load_calc=ic,
                    remote_machine_config=rm_config,
                    base_dir=tb.P.home().joinpath("tmp_results").joinpath(f"bot_simulator/ark1_hp_tuning"),
                    )
        c.run(run=True, machines_per_tab=2, window_number=354)


if __name__ == '__main__':
    pass
