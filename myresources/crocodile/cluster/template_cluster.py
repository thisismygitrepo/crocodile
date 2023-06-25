
import crocodile.toolbox as tb
from crocodile.cluster.distribute import Cluster, WorkloadParams


class ExpensiveComputation:
    @staticmethod
    def func_single_job(workload_params: WorkloadParams, *args, **kwargs) -> str:
        from crocodile.cluster.utils import expensive_function
        expensive_function(workload_params=workload_params, *args, **kwargs)
        return "Done"

    @staticmethod
    def submit():
        from crocodile.cluster.distribute import RemoteMachineConfig, LoadCriterion, Cluster, ThreadLoadCalculator
        config = RemoteMachineConfig(
            # connection
            ssh_params={}, description="Description of running an expensive function",  # job_id=, base_dir="",
            # data
            copy_repo=False, update_repo=False, install_repo=False, update_essential_repos=True, data=[],
            transfer_method="sftp",
            # remote machine behaviour
            open_console=True, notify_upon_completion=True, to_email='random@email.com', email_config_name='enaut',
            kill_on_completion=False,
            # execution behaviour
            ipython=True, interactive=True, pdb=False, pudb=False, wrap_in_try_except=True,
            # resources
            lock_resources=True, max_simulataneous_jobs=2, parallelize=True, )
        # ssh_params = [dict(host="thinkpad"), dict(host="p51s")]  # ,
        ssh_params = [dict(host="214676wsl"), dict(host="229234wsl")]
        c = Cluster(func=ExpensiveComputation.func_single_job,
                    func_kwargs=dict(sim_dict=dict(a=2, b=3)),
                    ssh_params=ssh_params,
                    remote_machine_config=config,
                    thread_load_calc=ThreadLoadCalculator(num_jobs=3, load_criterion=LoadCriterion.cpu),  # if this machine can run 3 jobs at a time, how many can other machines do?
                    )
        c.run(run=True, machines_per_tab=2, window_number=354)
        return c


def try_run_on_cluster():
    import time
    ExpensiveComputation.submit()
    ExpensiveComputation.submit()
    cluster = ExpensiveComputation.submit()
    # later ...
    time.sleep(50)
    c = Cluster.load(cluster.job_id)
    c.open_mux(machines_per_tab=1)
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


if __name__ == '__main__':
    pass
