

import crocodile.toolbox as tb
# from crocodile.cluster.remote_machine import ThreadParams
from crocodile.cluster.distribute import Cluster, RemoteMachineConfig, ThreadLoadCalculator, MachineLoadCalculator


def run_on_remote():
    from crocodile.cluster.remote_machine import RemoteMachine, RemoteMachineConfig, WorkloadParams
    from crocodile.cluster.distribute import expensive_function
    config = RemoteMachineConfig(
        # connection
        ssh_params=dict(host="p51s"), description="Description of running an expensive function",  # job_id=, base_dir="",
        # data
        copy_repo=False, update_repo=False, install_repo=False, update_essential_repos=True, data=[], transfer_method="sftp",
        # remote machine behaviour
        open_console=True, notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut', kill_on_completion=False,
        # execution behaviour
        ipython=True, interactive=True, pdb=False, pudb=False, wrap_in_try_except=True,
        # resources
        lock_resources=True, max_simulataneous_jobs=2, parallelize=False, )
    m = RemoteMachine(func=expensive_function, func_kwargs=dict(sim_dict=dict(a=2, b=3), workload_params=WorkloadParams()),
                      config=config)
    m.run()
    return m


def try_run_on_remote():
    run_on_remote()
    run_on_remote()
    i3 = run_on_remote()
    i3.check_job_status()
    i3.download_results(r=True)
    i3.delete_remote_results()


def run_on_cluster():
    from crocodile.cluster.remote_machine import RemoteMachineConfig, WorkloadParams
    from crocodile.cluster.distribute import expensive_function, LoadCriterion
    config = RemoteMachineConfig(
        # connection
        ssh_params={}, description="Description of running an expensive function",  # job_id=, base_dir="",
        # data
        copy_repo=False, update_repo=False, install_repo=False, update_essential_repos=True, data=[], transfer_method="sftp",
        # remote machine behaviour
        open_console=True, notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut', kill_on_completion=False,
        # execution behaviour
        ipython=True, interactive=True, pdb=False, pudb=False, wrap_in_try_except=True,
        # resources
        lock_resources=True, max_simulataneous_jobs=2, parallelize=True, )
    ssh_params = [dict(host="thinkpad"), dict(host="p51s")]  # , dict(host="surface_wsl"), dict(port=2222)
    c = Cluster(func=expensive_function,
                func_kwargs=dict(sim_dict=dict(a=2, b=3)),
                ssh_params=ssh_params,
                remote_machine_config=config,
                thread_load_calc=ThreadLoadCalculator(multiplier=2, load_reference_value=3, load_criterion=LoadCriterion.cpu),  # if this machine can run 3 jobs at a time, how many can other machines do?
                )
    c.run(run=True, machines_per_tab=2)
    return c.job_id


def try_run_on_cluster():
    import time
    run_on_cluster()
    run_on_cluster()
    job_id = run_on_cluster()
    # later ...
    time.sleep(50)
    c = Cluster.load(job_id)
    c.open_mux(machines_per_tab=1)
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


class ExpensiveComputation:
    def __int__(self):
        pass

    def main_single_thread(self, single_thread_params):
        raise NotImplementedError

    def main(self, params: list):
        tb.L(params).apply(lambda single_thread_params: self.main_single_thread(single_thread_params), jobs=len(params))

    def run(self, ms):
        ic = ThreadLoadCalculator(multiplier=7, load_criterion="cpu", reference_machine="this_machine")
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
