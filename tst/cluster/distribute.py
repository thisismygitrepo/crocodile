
import crocodile.toolbox as tb
from crocodile.cluster.distribute import Cluster, ThreadLoadCalculator, RemoteMachineConfig
import time
from crocodile.cluster.trial_file import expensive_function


def run():
    config = RemoteMachineConfig(lock_resources=True, kill_on_completion=True, install_repo=False, parallelize=True)
    ssh_params = [dict(host="thinkpad"), dict(host="surface_wsl")]  # , dict(host="surface_wsl"), dict(port=2222)
    c = Cluster(func=expensive_function,
                func_kwargs=dict(sim_dict=dict(a=2, b=3)),
                ssh_params=ssh_params,
                thread_load_calc=ThreadLoadCalculator(multiplier=3, bottleneck_reference_value=8),
                remote_machine_config=config
                )
    c.run(run=True, machines_per_tab=2)
    return c.job_id


def try_it():
    run()
    run()
    job_id = run()

    # later ...
    time.sleep(50)
    c = Cluster.load(job_id)
    c.open_mux(machines_per_tab=1)
    c.check_job_status()
    c.download_results()
    tb.L(c.machines).delete_remote_results()
    return c


if __name__ == '__main__':
    # try_it()
    pass
