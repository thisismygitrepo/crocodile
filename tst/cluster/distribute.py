
import crocodile.toolbox as tb
from crocodile.cluster.distribute import Cluster, ThreadsWorkloadDivider
import time


def run():
    from crocodile.cluster.trial_file import expensive_function_single_thread
    machine_specs_list = [dict(host="thinkpad"), dict(host="p51s")]  # , dict(host="surface_wsl"), dict(port=2224)
    c = Cluster(func=expensive_function_single_thread,
                func_kwargs=dict(sim_dict=dict(a=2, b=3)),
                machine_specs_list=machine_specs_list, install_repo=False,
                thrd_load_calc=ThreadsWorkloadDivider(multiplier=3, bottleneck_reference_value=8),
                lock_resources=True,
                parallelize=True)
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
